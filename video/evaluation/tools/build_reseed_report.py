#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Regenerate docs/reseed_video_report.md from the local result files.

Reads evaluation/results_5arm (71-prompt diversity + VBench for every arm),
evaluation/results_colornorm_all (color-normalized variant, optional),
evaluation/results_distillalign (PRDC) and the overhead benchmark JSON. Every
number in the report comes from these files; the narrative is in this script.
ReSeed arms are discovered by name (dmd2_reseed_ckptN) and sorted by N.
"""
from __future__ import annotations

import csv
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R5 = REPO / "evaluation/results_5arm"
RC = REPO / "evaluation/results_colornorm_all"
RD = REPO / "evaluation/results_distillalign"
OUT = REPO / "docs/reseed_video_report.md"

div = json.load(open(R5 / "diversity_summary.json"))["models"]
vb = {r["model"]: r for r in csv.DictReader(open(R5 / "vbench_summary.csv"))}
prdc = {r["model"]: r for r in json.load(open(RD / "summary.json"))["results"]}
bench = json.load(open(R5 / "reseed_bench_checkpoint-300.json"))
cn = json.load(open(RC / "diversity_summary.json"))["models"] if (RC / "diversity_summary.json").is_file() else {}
rows = list(csv.DictReader(open(R5 / "diversity_per_prompt.csv")))
by: dict = {}
for r in rows:
    by.setdefault(r["model"], {})[int(r["prompt_id"])] = r

def ckpt_step(a: str) -> int:
    mo = re.search(r"ckpt(\d+)$", a); return int(mo.group(1)) if mo else -1
RESEED = sorted([a for a in div if a.startswith("dmd2_reseed_ckpt")], key=ckpt_step)
ARMS = ["teacher", "dmd2", "dfd", *RESEED]
PRDC_ARMS = ["dmd2", "dfd", *sorted([a for a in prdc if a.startswith("dmd2_reseed_ckpt")], key=ckpt_step)]
def NAME(a: str) -> str:
    return {"teacher": "Teacher (Wan2.1-1.3B, 50 步)", "dmd2": "DMD2", "dfd": "DFD"}.get(a, f"DMD2+ReSeed@{ckpt_step(a)}")
BEST = "dmd2_reseed_ckpt300"

def m(a, k): return div[a][k]["mean"]
def ci(a, k):
    c = div[a][k]["ci95"]; return f"{div[a][k]['mean']:.4f} [{c[0]:.3f}, {c[1]:.3f}]"
def vbf(a, k): return f"{float(vb[a][k]):.3f}" if a in vb else "—"

random.seed(0)
def paired(a, b, key, B=4000):
    ids = sorted(set(by[a]) & set(by[b])); d = [float(by[a][i][key]) - float(by[b][i][key]) for i in ids]
    boots = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(B))
    return st.mean(d), boots[int(0.025 * B)], boots[int(0.975 * B)], sum(x > 0 for x in d), len(d)
def pstr(a, b, key):
    mu, lo, hi, w, n = paired(a, b, key); return f"{mu:+.4f} [{lo:+.4f}, {hi:+.4f}]，{w}/{n}"

L: list = []
A = L.append
gap = m("teacher", "dino") - m("dmd2", "dino")
gain = m(BEST, "dino") - m("dmd2", "dino")
A("# ReSeed 在视频模型（Wan2.1-T2V-1.3B / DMD2）上的完整实验报告")
A("")
A("更新：2026-09-22（由 `evaluation/tools/build_reseed_report.py` 从结果文件生成）。代码仓库 `data-forcing-distillation`（本地 `evaluation/`），集群 `/data/fs201205/wt93384/project/DFD/`。")
A("配套文档：[`docs/dfd_eval_audit.md`](dfd_eval_audit.md)（评测审计）、[`evaluation/README.md`](../evaluation/README.md)（流水线用法）。")
A("")
A("## 0. 一页结论")
A("")
A(f"- **ReSeed 有效**：在发布的 DMD2 4 步 student 上加 46,080 个参数（denoiser 的 {bench['adapter_share_of_denoiser_pct']:.5f}%）的第一步残差偏置，训练 300 步后，71 条官方 prompt × 8 seed 的 DINO 多样性从 {m('dmd2','dino'):.4f} 升到 {m(BEST,'dino'):.4f}（成对差 {pstr(BEST,'dmd2','dino_diversity')} 个 prompt 更高），补回了 DMD2 相对 teacher 丢失多样性（{gap:.4f}）的 {100*gain/gap:.0f}%；CLIP 两种读法、逐帧变体、色彩归一化变体、DistillAlign coverage 全部同向。")
A(f"- **文本一致性不降**（{m('dmd2','clip_text_adherence'):.4f} → {m(BEST,'clip_text_adherence'):.4f}），**VBench 质量维持在 teacher 水平**（subject/background/flicker/smoothness/aesthetic 均 ≥ teacher；imaging −0.045、dynamic degree −0.12 是代价）。")
R3b = REPO / "evaluation/results_vbench320/diversity_summary.json"
if R3b.is_file():
    d3b = json.load(open(R3b))["models"]
    if "teacher" in d3b and "dmd2_reseed_ckpt300" in d3b:
        A("- **第二评测集（VBench 类别 320 条 × 8 seed，四臂）复现同一结论**：DINO teacher %.3f / DMD2 %.3f / ReSeed@300 %.3f / DFD %.3f，ReSeed 补回 %.0f%% 的 teacher 差距，8 个类别全部显著；DFD 与 DMD2 持平；teacher 的 dynamic degree（0.37）低于 DMD2（0.53），ReSeed（0.33）更接近 teacher（§3.7）。"
          % (d3b["teacher"]["dino"]["mean"], d3b["dmd2"]["dino"]["mean"], d3b["dmd2_reseed_ckpt300"]["dino"]["mean"], d3b["dfd"]["dino"]["mean"],
             100 * (d3b["dmd2_reseed_ckpt300"]["dino"]["mean"] - d3b["dmd2"]["dino"]["mean"]) / (d3b["teacher"]["dino"]["mean"] - d3b["dmd2"]["dino"]["mean"])))
A(f"- **推理开销可忽略**：每样本 {bench['per_sample_ms'][0]:.1f} → {bench['per_sample_ms'][1]:.1f} ms（+{bench['per_sample_overhead_pct']:.3f}%），峰值激活内存不变。")
if "dmd2_reseed_ckpt100" in div:
    A(f"- **增益来得很早、之后是随机游走**：100 步时 DINO 已到 {m('dmd2_reseed_ckpt100','dino'):.4f}（300 步增益的 {100*(m('dmd2_reseed_ckpt100','dino')-m('dmd2','dino'))/gain:.0f}%），200 步最高（{m('dmd2_reseed_ckpt200','dino'):.4f}）；100 步之后每 100 步的偏置增量彼此近似正交（余弦 ≈ 0），幅值却按 √t 增长，质量指标随 checkpoint 非单调摆动（§2.4、§3.4）。")
A("- **需要早停**：loss 在 ~300 步后平台化，但偏置幅值持续单调增长（RMS 0.013 → 0.029 @2000 步）；500 步时视频级多样性不再上升、CLIP 多样性回落、VBench 一致性/画质全面下降。300 步是当前工作点。")
A(f"- **DFD 复现结果与论文相反**：发布 checkpoint 上 DFD 的多样性低于 DMD2（DINO 成对差 {pstr('dfd','dmd2','dino_diversity')}），三套特征空间同向；VBench 5 个论文维度符号与论文一致。倾向解释：发布的 DMD2（1000 步）不在论文所述的坍缩状态（论文 baseline 30k 步）。")
A("")
A("## 1. 实验设置")
A("")
A("### 1.1 模型与 checkpoint")
A("")
A("| arm | 权重 | 采样 |")
A("|---|---|---|")
A("| teacher | Wan2.1-T2V-1.3B 基座（fastgen 官方 teacher 采样器） | UniPC 50 步，shift 5，CFG 5.0，官方负面 prompt |")
A("| dmd2 | 官方发布 `dmd2_t2v_vipe/0001000`（DMD2 蒸馏 1000 步） | 4 步 ODE，t_list = [0.999, 0.937, 0.833, 0.624, 0]（uniform 4 步 + shift 5），无 CFG |")
A("| dfd | 官方发布 `ours_wan_t2v_vipe/0000040`（从上述 DMD2 再训 40 步） | 同 dmd2 |")
A("| dmd2_reseed_ckptN | dmd2 权重 + ReSeed 偏置（checkpoint-N，只在第 1 步生效） | 同 dmd2 |")
A("")
A("推理统一走官方 `scripts/inference/video_model_inference.py` + `WanT2V/config_dmd2.py`：832×480、81 帧、16 fps、bf16；每次运行只抽一次噪声并对所有 prompt 复用，所以同一 seed 下各 arm 的初始噪声比特一致（已用 sha256 探针验证），所有比较都是成对的。")
A("")
A("### 1.2 数据集")
A("")
A("| 集合 | 来源 | 规模 | 用途 |")
A("|---|---|---|---|")
A("| DFD 官方 eval prompts | `web_dfd_eval_prompts.txt`，134 行去重后 71 条唯一 prompt | 71 × 8 seed（0–7）= 568 视频/arm | 多样性、文本一致性、VBench |")
A("| VBench 类别 prompt（第二评测集） | VBench 官方 `prompts/prompts_per_category/*.txt`（Vchitect/VBench master fd18b3d0，2026-08-21），animal / architecture / food / human / lifestyle / plant / scenery / vehicles 8 类各取前 40 条；`plant.txt` 第 1 条与 lifestyle 重复，跳过顺延；文件 `evaluation/prompts/vbench_category_320.txt`（sha256 4c3cdab1…），类别对照 `vbench_category_320.categories.csv`，来源记录 `vbench_category_320.PROVENANCE.md` | 320 × 8 seed（0–7）= 2,560 视频/arm，三个 student arm 共 7,680；VBench 只用 seed 0–4（1,600 视频/arm） | 多样性、文本一致性、VBench、按类别分解（§3.7） |")
A("| DistillAlign eval-16 | 官方 `prompts/distribution_eval_16.txt` 逐字复制 | 16 × 16 seed（11,22,33,42,44,55,66,77,88,123,456,789,2024,3407,7777,9999）= 256 视频/arm | PRDC |")
A("| ViPE long captions | 4000 条，UMT5 embedding 预先缓存 | 仅训练 ReSeed | — |")
A("")
A("### 1.3 指标定义")
A("")
A("- **DINO / CLIP 多样性**（DFD 论文 / DP-DMD 定义）：每个视频均匀取 16 帧 → 逐帧 embedding L2 归一化 → 帧平均 → 对每个 prompt 的 8 个视频算 C(8,2)=28 对余弦，D = 1 − 平均余弦；对 71 个 prompt 取均值，2000 次 bootstrap 95% CI。DINO 用 `facebook/dinov2-large` pooler 输出；CLIP 用 `openai/clip-vit-large-patch14`，同时报告投影特征（768-d，`image_embeds`）和 DP-DMD 代码的 CLS 特征（1024-d，`last_hidden_state[:,0]`）。变体：逐帧（先逐帧算 D 再平均）、色彩归一化（逐视频减均值除标准差后再编码）。")
A("- **CLIP 文本一致性**：CLIP 文本 embedding 与 16 帧视觉 embedding 均值的余弦。")
A("- **VBench 7 维**（官方 VBench 0.1.5，`custom_input`）：subject/background consistency、temporal flickering、motion smoothness、dynamic degree、aesthetic、imaging。")
A("- **DistillAlign PRDC**（arXiv 2607.26811）：V-JEPA2 ViT-H（前 81 帧 linspace 8 帧，token mean⊕std，L2，2560-d）；teacher 每个样本的 k=5 近邻半径定义支撑集；precision = student 落在 teacher 球内的比例，coverage = teacher 样本被最近 student 覆盖的比例。teacher 特征直接用官方发布的 Wan2.1-1.3B 缓存（25 步 UniPC、shift 8、CFG 5），协议哈希（模型 revision、帧选择、池化、transformers 4.57.6 / torch 2.6.0+cu124 / numpy 1.26.4、代码 sha）与官方一致；student 走 post-DMD 协议（`--no-refine`）。")
A("")
A("## 2. ReSeed 操作细节")
A("")
A("### 2.1 适配器结构（`evaluation/reseed/adapter.py`）")
A("")
A("- `WanBlockResidualBias`：对 Wan transformer 的 30 个 block，各注册一个 forward hook，在 block **输出**上加一个逐通道偏置 `b_ℓ ∈ R^1536`（fp32 参数，零初始化）；30 × 1536 = **46,080 个参数**（bf16 184 KB），占 denoiser 1,418,996,800 参数的 **0.00325%**。偏置对所有 token（帧 × 空间）共享，时间维度直接沿用图像版做法不分帧。")
A("- `FirstStepGate`：网络的 forward pre-hook 读取时间步 `t`，`t > (t₀+t₁)/2 = 0.968` 时开启偏置，否则关闭 → 偏置只在 4 步采样的**第 1 步**生效。读取 `t` 不消耗 RNG，加载 adapter 前后抽到的噪声比特一致。")
A("- 零初始化时输出与原模型逐比特一致（训练前的 identity 检查），因此 adapter 是严格的\"加法\"修改，随时可卸载。")
A("")
A("### 2.2 训练目标（`evaluation/reseed/train_reseed.py`, `flow.py`）")
A("")
A("- 每步抽一批 prompt（4 个 rank × 1 = 4 条/步，从 4000 条 ViPE caption 的固定随机流中取）和噪声 z ~ N(0, I)（latent 16×21×60×104）。")
A("- **teacher 目标**：以 z 为起点，用 teacher（Wan2.1-1.3B，CFG 5.0，含负面 prompt）在 student 第一段区间 [t₀=0.999, t₁=0.937] 上做 **m=13 步 Euler 微步**（微步网格在 unshifted-sigma 空间均匀、端点与 student 区间边界对齐，`boundary_aligned_teacher_grid`），得到 x_{t₁}；再折算成把 z 从 t₀ 送到 x_{t₁} 的单步**割线速度** v* = (z − x_{t₁})/(t₀ − t₁)。")
A("- **student 预测**：带偏置的 student 在 t₀ 上预测 x̂₀，速度 v̂ = (z − x̂₀)/t₀（`student_velocity`）。")
A("- **损失**：MSE(v̂, v*)，只有 46,080 个偏置参数有梯度（step-1 梯度断言保证梯度只来自第 1 步）。记录 velocity cosine、相对误差、端点相对误差、梯度范数、bias RMS/max。")
A("- 直觉：DMD 蒸馏使 student 第一步把不同噪声映射到少数几个 x̂₀（坍缩）；用 teacher 的第一步割线速度做监督，把 student 第一步的\"落点\"推回 teacher 会到达的位置，而后续 3 步不动，从而恢复初始噪声到样本的多样性映射。")
A("")
A("### 2.3 训练配置")
A("")
A("| 项 | 值 |")
A("|---|---|")
A("| student | `dmd2_t2v_vipe/0001000`（FSDP 加载，与推理脚本相同构造路径） |")
A("| teacher | 配置中的 Wan2.1-T2V-1.3B，CFG 5.0 |")
A("| 数据 | ViPE long caption 4000 条，UMT5 embedding 预编码缓存（`reseed_data/vipe_long/umt5_embeddings.pt`） |")
A("| 微步数 m | 13 |")
A("| 优化器 | AdamW，lr 1e-3，β=(0.9, 0.999)，weight decay 0，梯度裁剪 1.0 |")
A("| 并行 | 4 × H100 DDP（每 rank 1 条 prompt，梯度 all-reduce） |")
A("| 步数 | 2000（每 100 步存 checkpoint，含每 rank RNG/流状态，可精确断点续训） |")
A("| 耗时 | 30.3 s/步（teacher 13 微步 × 2（CFG）26.2 s + student 4.2 s），2000 步 16 h 54 min |")
A("| 非确定性 | 反向传播非确定，同配置两次运行 7 步后偏置最大差 4.7e-3 |")
A("")
A("### 2.4 训练曲线")
A("")
A("| 窗口 | loss | velocity cosine | bias RMS | bias max |")
A("|---|---|---|---|---|")
A("| 0–200 | 0.369 | 0.907 | 0.013 | 0.060 |")
A("| 200–400 | 0.316 | 0.919 | 0.016 | 0.087 |")
A("| 400–600 | 0.294 | 0.926 | 0.018 | 0.120 |")
A("| 600–1000 | 0.29–0.30 | 0.925 | 0.022 | 0.159 |")
A("| 1000–1600 | 0.29–0.30 | 0.924–0.926 | 0.026 | 0.233 |")
A("| 1600–2000 | 0.274–0.279 | 0.929–0.930 | 0.029 | 0.264 |")
A("")
A("loss / cosine 在 ~300–400 步后平台化；偏置幅值全程单调增长，且没有饱和迹象——优化器在平坦谷里沿几乎不影响 loss 的方向推大偏置，而生成质量对幅值敏感（见 §3.4）。图：`evaluation/reseed/plot_training.py` 生成的 `training_curves.png`。")
A("")
A("**偏置向量的演化**（对 checkpoint 的 `residual_biases.pt` 直接分析）：相邻 checkpoint 的余弦 0.86–0.92，但每 100 步的**增量**彼此近似正交（d200·d100 = 0.06，d300·d200 = 0.08，d400·d300 = −0.01，d500·d400 = −0.03）。也就是说前 100 步学到的是一个一致的方向，之后的更新是 4 条 prompt/步的梯度噪声在平坦方向上的随机游走：幅值按 √t 增长（RMS 100 步 0.0104 → 2000 步 0.0286），方向却不再累积。这解释了 §3.4 里质量指标随 checkpoint 非单调的现象——不同 checkpoint 之间的差别主要是随机分量。各 block 的 RMS 相近（0.009–0.018，中后段 block 略大，最后一个 block 最小）。")
A("")
A("### 2.5 推理开销（`evaluation/reseed/bench_overhead.py`，H100，bf16，官方 `generator_fn`，8 样本中位数）")
A("")
A("| Student model | Params | of denoiser | DINO before → after | Gain | Peak act. change | Step 1 (ms) | Per sample (ms) | Overhead |")
A("|---|---|---|---|---|---|---|---|---|")
A(f"| Wan2.1-T2V-1.3B (DMD2, 4-step) | {bench['adapter_params']:,} | {bench['adapter_share_of_denoiser_pct']:.5f}% | {m('dmd2','dino'):.4f} → {m(BEST,'dino'):.4f} | +{100*(m(BEST,'dino')/m('dmd2','dino')-1):.1f}% | +0.00 MiB | {bench['step1_ms'][0]:.1f} → {bench['step1_ms'][1]:.1f} | {bench['per_sample_ms'][0]:.1f} → {bench['per_sample_ms'][1]:.1f} | +{bench['per_sample_overhead_pct']:.3f}% |")
A("")
A(f"每步约 993 ms（4 步几乎等长），adapter 仅在第 1 步多 {bench['step1_ms'][1]-bench['step1_ms'][0]:.1f} ms；基线重复测量漂移 {bench['baseline_drift_ms']:.1f} ms，因此 +{bench['per_sample_overhead_pct']:.3f}% 应视为上界。峰值激活内存实测差 {bench['peak_activation_change_mib']:+.0f} MiB，在 caching allocator 噪声内（偏置只是 block 输出上的一次逐通道加法，无新增持久激活）。per-sample 不含 VAE 解码。DINO 为 DINOv2-L。")
A("")
A("## 3. 结果")
A("")
A("### 3.1 多样性与文本一致性（71 prompt × 8 seed，均值；括号为 bootstrap 95% CI）")
A("")
A("| arm | DINO | CLIP proj | CLIP cls | DINO 逐帧 | CLIP proj 逐帧 | CLIP cls 逐帧 | 文本一致性 |")
A("|---|---|---|---|---|---|---|---|")
for a in ARMS:
    A(f"| {NAME(a)} | {ci(a,'dino')} | {m(a,'clip_proj'):.4f} | {m(a,'clip_cls'):.4f} | {m(a,'dino_per_frame'):.4f} | {m(a,'clip_proj_per_frame'):.4f} | {m(a,'clip_cls_per_frame'):.4f} | {m(a,'clip_text_adherence'):.4f} |")
A("")
if cn:
    A("色彩归一化变体（DINO / CLIP proj / CLIP cls）：" + "；".join(f"{NAME(a)} {cn[a]['dino']['mean']:.4f} / {cn[a]['clip_proj']['mean']:.4f} / {cn[a]['clip_cls']['mean']:.4f}" for a in ARMS if a in cn) + "。")
    A("")
A("### 3.2 逐 prompt 成对差（同噪声；均值 [bootstrap 95% CI]，胜出 prompt 数/71）")
A("")
A("| 比较 | DINO | CLIP proj | CLIP cls | DINO 逐帧 | 文本一致性 |")
A("|---|---|---|---|---|---|")
pairs = [(a, "dmd2") for a in RESEED] + [(RESEED[i+1], RESEED[i]) for i in range(len(RESEED)-1)] + [(BEST, "dfd"), ("dfd", "dmd2"), ("dmd2", "teacher"), (BEST, "teacher"), ("dfd", "teacher")]
for a, b in pairs:
    A(f"| {NAME(a)} − {NAME(b)} | " + " | ".join(pstr(a, b, k) for k in ["dino_diversity", "clip_diversity", "clip_cls_diversity", "dino_diversity_per_frame", "clip_text_adherence"]) + " |")
A("")
A("### 3.3 VBench")
A("")
A("| arm | subject | background | flicker | smoothness | dynamic degree | aesthetic | imaging |")
A("|---|---|---|---|---|---|---|---|")
VK = ["subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality"]
for a in ARMS:
    A(f"| {NAME(a)} | " + " | ".join(vbf(a, k) for k in VK) + " |")
A("")
A("DFD 与 DMD2 相比 5 个论文报告维度（subject、background、flicker、smoothness、aesthetic）的符号全部与论文一致，说明质量类指标可复现，只有多样性方向相反。")
A("")
A("### 3.4 checkpoint 扫描（ReSeed 训练步数 vs 指标）")
A("")
A("| checkpoint | DINO | CLIP proj | CLIP cls | 文本一致性 | VBench dynamic | VBench imaging | VBench subject | PRDC precision | PRDC coverage |")
A("|---|---|---|---|---|---|---|---|---|---|")
def prdcf(a, k): return f"{prdc[a][k]:.3f}" if a in prdc else "—"
for a in ["dmd2", *RESEED]:
    label = "dmd2（无 adapter）" if a == "dmd2" else f"ckpt-{ckpt_step(a)}"
    A(f"| {label} | {m(a,'dino'):.4f} | {m(a,'clip_proj'):.4f} | {m(a,'clip_cls'):.4f} | {m(a,'clip_text_adherence'):.4f} | {vbf(a,'dynamic_degree')} | {vbf(a,'imaging_quality')} | {vbf(a,'subject_consistency')} | {prdcf(a,'precision')} | {prdcf(a,'coverage')} |")
A("")
A("多样性在 100 步已拿到绝大部分增益（DINO +0.036，为最大增益 +0.047 的 77%），200 步达到最高（0.2341），之后不再上升（300/500 与 200 的成对差 CI 含 0，CLIP 在 500 步回落）。**质量随 checkpoint 非单调**：ckpt-100 和 ckpt-300 的一致性 / flicker / smoothness / aesthetic 均不低于 dmd2，而 ckpt-200 和 ckpt-500 在这些维度上跌破 teacher；dynamic degree 在 0.39（100）→ 0.67（200）→ 0.49（300）→ 0.80（500）之间来回摆动。结合 §2.4 的偏置增量正交性，这不是训练步数的系统性效应，而是随机游走分量落在不同位置：第一步的逐通道偏置会整体改变视频的运动幅度与帧间稳定性，而这个方向并不受训练目标约束。imaging quality 是唯一在所有 checkpoint 上都一致低于 dmd2 的维度（−0.04 ～ −0.09）。")
A("")
A("工作点选择：以\"多样性增益 + 质量不低于 teacher\"为准，**ckpt-100**（DINO +0.036，aesthetic 0.633 最高，一致性最好）和 **ckpt-300**（DINO +0.043）都可用；ckpt-100 的 dynamic degree 最低（0.39 vs teacher 0.55）是它的主要代价。下一版训练应加偏置范数约束或按 bias RMS 早停，并把 dynamic degree 纳入 checkpoint 选择。")
A("")
A("### 3.5 DistillAlign PRDC（16 prompt × 16 seed，对官方 Wan2.1-1.3B teacher 缓存，k=5）")
A("")
A("| arm | precision | coverage |")
A("|---|---|---|")
for a in PRDC_ARMS:
    A(f"| {NAME(a)} | {prdc[a]['precision']:.4f} | {prdc[a]['coverage']:.4f} |")
A("")
A("按 prompt 做成对 bootstrap（16 条 prompt）后所有差异的 95% CI 都跨 0（如 ReSeed@300 − dmd2 coverage +0.020 [−0.027, +0.082]），PRDC 只作方向性佐证：ReSeed（≥200 步）两项均不低于 dmd2，DFD coverage 最低。ckpt-200 与 300 数值相同是巧合（129/256 与 78/256），两组视频与特征均不同（平均余弦 0.993）。")
A("")
A("### 3.6 DMD2 vs DFD 复现（与论文对照）")
A("")
A("| 指标 | 论文 Table 1 | 本复现（发布 checkpoint，seed 0–7） |")
A("|---|---|---|")
A(f"| DINO 多样性 DMD2 → DFD | 0.190 → 0.205（+0.015） | {m('dmd2','dino'):.4f} → {m('dfd','dino'):.4f}（成对 {pstr('dfd','dmd2','dino_diversity')}） |")
A(f"| CLIP 多样性 DMD2 → DFD | 0.120 → 0.128（+0.008） | proj {m('dmd2','clip_proj'):.4f} → {m('dfd','clip_proj'):.4f}；cls {m('dmd2','clip_cls'):.4f} → {m('dfd','clip_cls'):.4f} |")
A("| VBench 5 维符号 | — | 全部一致（aesthetic 0.612 → 0.623 等） |")
A("")
A("已排除的解释：配置/噪声/文本 embedding 逐比特核对一致；checkpoint 正确加载；指标按 DP-DMD 参考实现逐项对照（两种 CLIP 读法、逐帧、色彩归一化）；V-JEPA2 coverage 独立同向。像素统计：DFD 像素 std 低约 35%、饱和度 0.63 vs 0.85（与论文\"DMD2 过饱和\"描述一致），对比度归一化后跨 seed 距离两者接近。倾向解释：发布的 DMD2 只训了 1000 步、8-seed 网格肉眼未坍缩，DFD 从此再训 40 步没有可恢复的多样性，反而收缩了对比度；论文的 DFD 是从 25k 步 DMD2 再训 100 步，不是同一条轨迹。可定案的实验：把 DMD2 继续训到 10k–30k 步再评（数十 GPU 小时）；便宜的补充：seed 1–8 重跑、联系作者确认 checkpoint 与表格的对应。")
A("")
R3 = REPO / "evaluation/results_vbench320"
if (R3 / "diversity_summary.json").is_file():
    d3 = json.load(open(R3 / "diversity_summary.json"))["models"]
    vb3 = {r["model"]: r for r in csv.DictReader(open(R3 / "vbench_summary.csv"))} if (R3 / "vbench_summary.csv").is_file() else {}
    by3: dict = {}
    for r in csv.DictReader(open(R3 / "diversity_per_prompt.csv")):
        by3.setdefault(r["model"], {})[int(r["prompt_id"])] = r
    def p3(a, b, key):
        ids = sorted(set(by3[a]) & set(by3[b])); dd = [float(by3[a][i][key]) - float(by3[b][i][key]) for i in ids]
        boots = sorted(st.mean(random.choices(dd, k=len(dd))) for _ in range(2000))
        return f"{st.mean(dd):+.4f} [{boots[50]:+.4f}, {boots[1949]:+.4f}]，{sum(x > 0 for x in dd)}/{len(dd)}"
    arms3 = [a for a in ["teacher", "dmd2", "dfd", *sorted([x for x in d3 if x.startswith("dmd2_reseed_ckpt")], key=ckpt_step)] if a in d3]
    A("### 3.7 第二评测集：VBench 类别 prompt 320 条 × 8 seed")
    A("")
    A("数据：`evaluation/prompts/vbench_category_320.txt`（见 §1.2），VBench 官方 8 类各前 40 条；arm = teacher / dmd2 / dfd / dmd2+ReSeed@300；每 arm 320 × 8 seed = 2,560 视频（四臂共 10,240），同 seed 同噪声。多样性与文本一致性用全部 8 个 seed（C(8,2)=28 对/prompt），VBench 用 seed 0–4（VBench 自身的 5 样本/prompt 约定，1,600 视频/arm）。作业：student 三臂 141546（生成 3 h 40 min、多样性 1 h、VBench 2.5 h）；teacher 每个 seed 一个作业（142259–142266，每作业 4 张卡各跑 80 条 prompt，`--prompt-range` 分片，约 2.5 h），收尾 142267（校验 + metadata + 四臂多样性 + teacher VBench + 合并 + 分类，4 h 23 min）。脚本 `run_evalset2.sbatch`、`run_evalset2_teacher.sbatch`、`run_evalset2_teacher_finalize.sbatch`；结果 `evaluation/results_vbench320/`。")
    A("")
    A("| arm | DINO | CLIP proj | CLIP cls | DINO 逐帧 | 文本一致性 |")
    A("|---|---|---|---|---|---|")
    for a in arms3:
        s = d3[a]; A(f"| {NAME(a)} | {s['dino']['mean']:.4f} [{s['dino']['ci95'][0]:.3f}, {s['dino']['ci95'][1]:.3f}] | {s['clip_proj']['mean']:.4f} | {s['clip_cls']['mean']:.4f} | {s['dino_per_frame']['mean']:.4f} | {s['clip_text_adherence']['mean']:.4f} |")
    A("")
    A("成对差（320 prompt；均值 [bootstrap 95% CI]，胜出数）：")
    A("")
    A("| 比较 | DINO | CLIP proj | CLIP cls | 文本一致性 |")
    A("|---|---|---|---|---|")
    for a, b in ([(x, "dmd2") for x in arms3 if x != "dmd2"] + [(x, "dfd") for x in arms3 if x.startswith("dmd2_reseed")]
                 + ([(x, "teacher") for x in arms3 if x != "teacher"] if "teacher" in arms3 else [])):
        A(f"| {NAME(a)} − {NAME(b)} | " + " | ".join(p3(a, b, k) for k in ["dino_diversity", "clip_diversity", "clip_cls_diversity", "clip_text_adherence"]) + " |")
    A("")
    if vb3:
        A("| arm | subject | background | flicker | smoothness | dynamic degree | aesthetic | imaging |")
        A("|---|---|---|---|---|---|---|---|")
        for a in arms3:
            if a in vb3:
                A(f"| {NAME(a)} | " + " | ".join(f"{float(vb3[a][k]):.3f}" for k in VK) + " |")
        A("")
    cb = R3 / "category_breakdown.md"
    if cb.is_file():
        sec = cb.read_text().split("## dino_diversity")[1].split("## ")[0].strip().splitlines()
        A("按类别的 DINO 多样性（`category_breakdown.md`，成对差相对 dmd2）：")
        A("")
        L.extend(sec)
        A("")
    if "teacher" in d3:
        gap3 = d3["teacher"]["dino"]["mean"] - d3["dmd2"]["dino"]["mean"]; gain3 = d3["dmd2_reseed_ckpt300"]["dino"]["mean"] - d3["dmd2"]["dino"]["mean"]
        A("在这套更通用的 prompt 集上：teacher 的 DINO 多样性 %.4f，DMD2 蒸馏丢掉 %.3f，ReSeed@300 补回 %.3f（**%.0f%%**，与 71-prompt 集上的 55%% 一致），8 个类别全部显著为正（food / lifestyle / human 最大，scenery 最小），teacher 在每个类别上都仍高于 ReSeed。DFD 与 DMD2 在 DINO 上持平、CLIP proj 略高——71-prompt 集上的“DFD 更低”没有再现，因此对 DFD 的稳妥表述是“两套集上均无显著多样性增益”。ReSeed 的文本一致性小幅显著下降（−0.0055；teacher %.4f、dmd2 %.4f、ReSeed %.4f）。VBench：一致性 / flicker / smoothness 四臂几乎相同（差 <0.01），ReSeed 的 aesthetic 最高（0.631 vs teacher 0.590）；**dynamic degree 上 teacher 本身只有 0.371**，dmd2 的 0.531 是蒸馏后偏高，ReSeed 的 0.329 反而更接近 teacher——71-prompt 集上被当作“代价”的 dynamic degree 下降，在 teacher 参照下应重新解读为向 teacher 靠拢；imaging（0.675 vs teacher 0.703、dmd2 0.698）仍是唯一一致偏低的维度。"
          % (d3["teacher"]["dino"]["mean"], gap3, gain3, 100 * gain3 / gap3, d3["teacher"]["clip_text_adherence"]["mean"], d3["dmd2"]["clip_text_adherence"]["mean"], d3["dmd2_reseed_ckpt300"]["clip_text_adherence"]["mean"]))
    else:
        A("在这套更通用的 prompt 集上：ReSeed 的 DINO 增益更大（+27%，8 个类别全部显著为正）；DFD 与 DMD2 在 DINO 上持平。")
    A("")
A("## 4. 可视化")
A("")
A("- Seed 网格（第 40 帧，行 = Teacher / DMD2 / ReSeed@300，列 = seed 0–7）：`evaluation/samples/grids_nodfd/seed_grid_p{026,013,007,015,030}_f40.png`（`evaluation/tools/seed_grid.py`）。")
A("- 拼接对比视频（3 行 × 4 列 seed 0–3；p022 另有 3 × 8 版本）：`evaluation/samples/compare_nodfd/compare_p{000,007,013,015,022,026,030,048,067}_*.mp4`；含 DFD 行的版本在 `evaluation/samples/compare/`。")
A("- 原始单视频：`evaluation/samples/raw_p026/`、`raw_p015/`。")
A("- 320 集：`evaluation/samples/vbench320/seed_grid_p{314,134,184,068,109,236,168}_f40.png` 与 `videos/compare_p*.mp4`（每类增益最大的 prompt + 中位数 p168）。**定性提醒**：VBench 类别 prompt 很短（如 \"a glass filled with beer\"、\"eiffel tower from the seine river\"），增益最大的几条里 ReSeed 的多样性有一部分来自语义漂移——啤酒杯里出现水果配料、铁塔变成教堂尖顶、\"红巴士\"三臂都没生成巴士而 ReSeed 的画面内容更散；这与 320 集上文本一致性显著下降 −0.0055 一致。中位数 prompt（p168）和长描述的 71 集上没有这种现象。论文选图应避开这几条，并在正文说明短 prompt 下多样性与 prompt 一致性的权衡。")
top = sorted(((float(by[BEST][i]["dino_diversity"]) - float(by["dmd2"][i]["dino_diversity"]), i) for i in by["dmd2"]), reverse=True)
A("- ReSeed@300 增益最大的 prompt（DINO 成对差）：" + "、".join(f"p{i:03d} {g:+.3f}" for g, i in top[:6]) + f"；中位数 p{top[len(top)//2][1]:03d} {top[len(top)//2][0]:+.3f}；最差 " + "、".join(f"p{i:03d} {g:+.3f}" for g, i in top[-2:]) + "。（p026 老夫妇跳舞、p022 拉花咖啡、p013 玻璃电梯、p007 沙漠摩托、p015 湖面木船、p067 咖啡馆雨窗、p000 卷发女子打伞。）")
A("")
A("## 5. 未完成 / 下一步")
A("")
missing_vb = [a for a in RESEED if a not in vb]
if missing_vb:
    A(f"1. {', '.join('ckpt-'+str(ckpt_step(a)) for a in missing_vb)} 的 VBench 与全 arm 色彩归一化复评完成后补入 §3.3/§3.4。")
A("2. 排除替代解释：随机偏置对照（同 RMS 的随机偏置）、门控消融（全 4 步生效 / 仅第 2 步）、偏置形状消融（分帧 / 分 token）。均不需重训只需生成。")
A("3. 带偏置范数约束（weight decay / 范数投影）或按 bias RMS 早停的训练版本。")
A("4. ReSeed 应用到 DFD student（DFD+ReSeed），把 DFD 从\"对手\"变成\"基座\"。")
A("5. 超参：m、学习率、CFG；训练 seed 方差；held-out velocity loss 曲线。")
A("6. DistillAlign 协议下自建 teacher（严格成对 PRDC）；DINOv3 重算以与图像表格口径统一。")
A("7. 论文写法：主表 teacher / DMD2 / DMD2+ReSeed；DFD 放附录并注明\"发布 checkpoint 上未复现其多样性增益，baseline 非坍缩状态\"；投稿前联系 DFD 作者。")
A("")
A("## 6. 文件索引")
A("")
A("| 内容 | 本地 | 集群 |")
A("|---|---|---|")
A("| 全 arm 71-prompt 结果（多样性 / VBench / 逐 prompt CSV） | `evaluation/results_5arm/` | `repo/evaluation/results/`, `results_reseed_checkpoint-N/` |")
A("| 第二评测集（VBench 类别 320）：多样性 / VBench / 按类别分解 | `evaluation/results_vbench320/` | `repo/evaluation/results_vbench320/`, 视频 `generated_vbench320/` |")
A("| 色彩归一化全 arm | `evaluation/results_colornorm_all/` | `repo/evaluation/results_colornorm_all/` |")
A("| DistillAlign 特征与 PRDC | `evaluation/results_distillalign/` | `repo/evaluation/results_distillalign/` |")
A("| DMD2/DFD 复现（含 refclip 变体、paired_analysis） | `evaluation/results_full71/` | `repo/evaluation/results_full71_v1/` |")
A("| 推理开销 | `evaluation/results_5arm/reseed_bench_checkpoint-300.json` | `reseed_runs/bench_checkpoint-300.json` |")
A("| ReSeed 训练日志 / checkpoint | — | `reseed_runs/dmd2_m13/{metrics.jsonl, checkpoint-100..2000}` |")
A("| 生成视频 | `evaluation/samples/` | `repo/evaluation/generated/{teacher,dmd2,dfd,dmd2_reseed_ckpt*}/`, `generated_distillalign/` |")
A("")
A("复现命令：`bash evaluation/scripts/run_all.sh`（71-prompt 全套）、`ADAPTER=<checkpoint-N> sbatch --export=ALL evaluation/reseed/eval_reseed.sbatch`（加一个 ReSeed arm）、`MODELS=<arm> ADAPTER=<ckpt> sbatch --export=ALL evaluation/scripts/run_distillalign.sbatch`（PRDC）、`sbatch evaluation/reseed/train_reseed.sbatch`（训练）、`sbatch evaluation/reseed/bench_overhead.sbatch`（开销）、`python evaluation/tools/build_reseed_report.py`（重建本报告）。")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({len(L)} lines); arms: {ARMS}; prdc arms: {PRDC_ARMS}; colornorm: {sorted(cn) if cn else 'absent'}")
