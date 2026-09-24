# SPDX-License-Identifier: Apache-2.0
"""Offline tests for eval_config (no GPU): run with `python -m pytest` or directly."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.tools.eval_config import load_config, model_spec  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "config.yaml"


def test_model_spec_exact_and_alias():
    os.environ["DFD_RESEED_ADAPTER"] = "/x/checkpoint-500"
    cfg = load_config(CFG)
    assert model_spec(cfg, "dmd2")["base_model"] == "dmd2"
    s = model_spec(cfg, "dmd2_reseed_ckpt500")
    assert s["base_model"] == "dmd2_reseed"            # longest prefix, not dmd2
    assert s["reseed_adapter"] == "/x/checkpoint-500"
    assert s["checkpoint"] == cfg["models"]["dmd2"]["checkpoint"]
    assert model_spec(cfg, "dfd_something")["base_model"] == "dfd"
    for bad in ("dmd2reseed", "dmd", "teacher2", "x_dmd2"):
        try:
            model_spec(cfg, bad)
        except KeyError:
            pass
        else:
            raise AssertionError(f"{bad!r} should not resolve")


def test_nested_default_expansion():
    from evaluation.tools.eval_config import _expand_str
    for k in ("A_SET", "A_UNSET", "R"):
        os.environ.pop(k, None)
    os.environ["A_SET"] = "/venv/bin/python"
    os.environ["R"] = "/root"
    s = "${A_SET:-${R:-.}/x/bin/python}"
    assert _expand_str(s) == "/venv/bin/python", _expand_str(s)
    s = "${A_UNSET:-${R:-.}/x/bin/python}"
    assert _expand_str(s) == "/root/x/bin/python", _expand_str(s)
    os.environ.pop("R")
    assert _expand_str(s) == "./x/bin/python"
    assert _expand_str("${A_UNSET:-}") == ""
    assert _expand_str("plain") == "plain"


if __name__ == "__main__":
    test_model_spec_exact_and_alias()
    test_nested_default_expansion()
    print("test_eval_config: OK")
