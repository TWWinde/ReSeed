# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the ReSeed port. No GPU, no checkpoints, no HF.

    python evaluation/reseed/test_reseed.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.reseed.adapter import FirstStepGate, WanBlockResidualBias, load_wan_residual_bias_adapter  # noqa: E402
from evaluation.reseed.flow import (  # noqa: E402
    boundary_aligned_teacher_grid, euler_step_via_x0, secant_velocity, solve_teacher_interval, student_velocity,
)

T_LIST = [0.999, 0.937, 0.833, 0.624, 0.0]
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'OK ' if ok else 'BAD'}] {name:<52s} {detail}")


# ---- a stand-in for fastgen's RFNoiseSchedule (x_t = (1-t)x0 + t eps) -------
class MiniRF:
    t_precision = torch.float64

    def forward_process(self, x, eps, t):
        t = t.to(torch.float64).view(-1, *([1] * (x.ndim - 1)))
        return ((1 - t) * x.double() + t * eps.double()).to(x.dtype)

    def x0_to_eps(self, xt, x0, t):
        t = t.to(torch.float64).view(-1, *([1] * (xt.ndim - 1)))
        return ((xt.double() - (1 - t) * x0.double()) / t).to(xt.dtype)


# ---- a stand-in for a Wan transformer: .blocks with attn1.to_out[0] ----------
class FakeBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn1 = nn.Module()
        self.attn1.to_out = nn.ModuleList([nn.Linear(d, d)])
        self.lin = nn.Linear(d, d)

    def forward(self, h, *rest):
        return self.lin(h)


class FakeTransformer(nn.Module):
    def __init__(self, n=4, d=16):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(d) for _ in range(n)])

    def forward(self, h):
        for b in self.blocks:
            h = b(h)
        return h


class FakeNet(nn.Module):
    """Mimics Wan.forward(x_t, t, condition=...) for the gate test."""

    def __init__(self, tr):
        super().__init__()
        self.transformer = tr

    def forward(self, x_t, t, condition=None):
        return self.transformer(x_t)


def main() -> int:
    torch.manual_seed(0)

    # ---------------------------------------------------------------- grid
    # Production passes config.model.sample_t_cfg.t_list -- a Python list -- so
    # the boundaries are exact float64. (A float32 tensor would carry
    # 0.99900001 and the pin would be to *that*, which is still correct.)
    g = boundary_aligned_teacher_grid(T_LIST, 0, 13, 5.0)
    check("grid has m+1 = 14 points", g.numel() == 14, f"{g.numel()}")
    check("grid endpoints pinned to t_list", float(g[0]) == 0.999 and float(g[-1]) == 0.937,
          f"{float(g[0]):.4f} .. {float(g[-1]):.4f}")
    g32 = boundary_aligned_teacher_grid(torch.tensor(T_LIST), 0, 13, 5.0)
    check("fp32 t_list pins to its own values", float(g32[0]) == float(torch.tensor(0.999)), "")
    check("grid strictly decreasing", bool(torch.all(g[:-1] > g[1:])))
    check("grid tracks native teacher (2nd pt 0.9951)", abs(float(g[1]) - 0.9951) < 5e-4, f"{float(g[1]):.4f}")
    check("grid interior != naive linspace in t", abs(float(g[1]) - (0.999 - 0.062 / 13)) > 1e-4,
          "(interpolated in unshifted space)")
    try:
        boundary_aligned_teacher_grid(torch.tensor([0.5, 0.9]), 0, 3, 5.0)
        check("rejects increasing t_list", False)
    except ValueError:
        check("rejects increasing t_list", True, "ValueError")

    # ---------------------------------------------------------------- Euler identity
    rf = MiniRF()
    B, shape = 2, (2, 3, 4, 5, 6)
    x = torch.randn(shape); x0 = torch.randn(shape)
    t = torch.full((B,), 0.999, dtype=torch.float64); tn = torch.full((B,), 0.937, dtype=torch.float64)
    stepped = euler_step_via_x0(rf, x, x0, t, tn)
    closed = x * (0.937 / 0.999) + x0 * (1 - 0.937 / 0.999)
    check("euler_step_via_x0 == x*(t'/t) + x0*(1-t'/t)", torch.allclose(stepped, closed, atol=1e-6),
          f"max|diff|={float((stepped - closed).abs().max()):.2e}")
    v = student_velocity(x, x0, t)
    check("student_velocity == (x - x0)/t", torch.allclose(v, (x - x0) / 0.999, atol=1e-6))
    euler_v = x + (0.937 - 0.999) * v
    check("x + dt*v reproduces the x0-path step", torch.allclose(euler_v, stepped, atol=1e-5),
          f"max|diff|={float((euler_v - stepped).abs().max()):.2e}")
    # secant of a single Euler step must recover that step's velocity
    sec = secant_velocity(x, stepped, 0.999, 0.937)
    check("secant(one Euler step) == its velocity", torch.allclose(sec, v, atol=1e-5),
          f"max|diff|={float((sec - v).abs().max()):.2e}")

    # ---------------------------------------------------------------- teacher solver
    # A teacher that always predicts x0 = 0 is a pure contraction; after the
    # grid the state is x * prod(t_{k+1}/t_k) = x * (t_end/t_start).
    end = solve_teacher_interval(lambda cur, tb: torch.zeros_like(cur), rf, x, g)
    check("solve_teacher_interval telescopes t_end/t_start", torch.allclose(end, x * (0.937 / 0.999), atol=1e-5),
          f"max|diff|={float((end - x * (0.937 / 0.999)).abs().max()):.2e}")
    check("solver output has no grad", not end.requires_grad)

    # ---------------------------------------------------------------- adapter
    tr = FakeTransformer(n=4, d=16)
    h = torch.randn(2, 7, 16)
    ref = tr(h).clone()
    ad = WanBlockResidualBias(tr, range(4))
    check("zero-init adapter is bit-identical", torch.equal(tr(h), ref))
    check("param count = blocks * width", sum(p.numel() for p in ad.parameters()) == 4 * 16)
    with torch.no_grad():
        ad.biases["3"].fill_(0.25)
    out_on = tr(h)
    check("last-block bias shifts output by exactly bias", torch.allclose(out_on, ref + 0.25, atol=1e-6))
    ad.disable()
    check("disabled adapter is bit-identical again", torch.equal(tr(h), ref))
    ad.enable()
    # gradient reaches the bias, and only the bias
    tr.requires_grad_(False)
    loss = tr(h).pow(2).mean(); loss.backward()
    check("bias receives gradient", ad.biases["3"].grad is not None and ad.biases["3"].grad.abs().sum() > 0)
    check("frozen base receives no gradient", all(p.grad is None for p in tr.parameters()))
    with tempfile.TemporaryDirectory() as td:
        ad.save(Path(td), step=7)
        tr2 = FakeTransformer(n=4, d=16); tr2.load_state_dict(tr.state_dict())
        ad2, payload = load_wan_residual_bias_adapter(tr2, td)
        check("save/load round-trips values", torch.equal(ad2.biases["3"], ad.biases["3"]) and payload["step"] == 7)
    ad.close()
    check("close() removes hooks", torch.equal(tr(h), ref))
    ad2.close()

    # ---------------------------------------------------------------- gate
    tr3 = FakeTransformer(n=2, d=16); net = FakeNet(tr3)
    ad3 = WanBlockResidualBias(tr3, [0, 1])
    with torch.no_grad():
        ad3.biases["1"].fill_(1.0)
    gate = FirstStepGate(ad3, T_LIST[0], T_LIST[1]).install(net)
    base = tr3(h)  # adapter currently active -> shifted
    y1 = net(h, torch.tensor([0.999]))
    check("gate: t=0.999 -> active", gate.last_active is True and torch.allclose(y1, base))
    y2 = net(h, torch.tensor([0.937]))
    check("gate: t=0.937 -> inactive", gate.last_active is False and not torch.allclose(y2, base))
    y3 = net(h, torch.tensor([0.624]))
    check("gate: t=0.624 -> inactive", gate.last_active is False and torch.allclose(y3, y2))
    y4 = net(h, t=torch.tensor([0.999]))
    check("gate: kwargs t also works", gate.last_active is True and torch.allclose(y4, base))
    gate.remove(); ad3.close()

    bad = [n for n, ok in results if not ok]
    print()
    if bad:
        print(f"FAILED: {bad}")
        return 1
    print(f"ALL {len(results)} RESEED TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
