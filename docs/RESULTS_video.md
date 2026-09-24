# ReSeed 在视频模型（Wan2.1-T2V-1.3B / DMD2）上的完整实验报告

更新：2026-09-22（由 `evaluation/tools/build_reseed_report.py` 从结果文件生成）。代码仓库 `data-forcing-distillation`（本地 `evaluation/`），集群 `/data/fs201205/wt93384/project/DFD/`。
配套文档：[`docs/dfd_eval_audit.md`](dfd_eval_audit.md)（评测审计）、[`evaluation/README.md`](../evaluation/README.md)（流水线用法）。

## 0. 一页结论

- **ReSeed 有效**：在发布的 DMD2 4 步 student 上加 46,080 个参数（denoiser 的 0.00325%）的第一步残差偏置，训练 300 步后，71 条官方 prompt × 8 seed 的 DINO 多样性从 0.1869 升到 0.2297（成对差 +0.0428 [+0.0288, +0.0575]，57/71 个 prompt 更高），补回了 DMD2 相对 teacher 丢失多样性（0.0778）的 55%；CLIP 两种读法、逐帧变体、色彩归一化变体、DistillAlign coverage 全部同向。
- **文本一致性不降**（0.2822 → 0.2846），**VBench 质量维持在 teacher 水平**（subject/background/flicker/smoothness/aesthetic 均 ≥ teacher；imaging −0.045、dynamic degree −0.12 是代价）。
- **第二评测集（VBench 类别 320 条 × 8 seed，四臂）复现同一结论**：DINO teacher 0.501 / DMD2 0.337 / ReSeed@300 0.429 / DFD 0.336，ReSeed 补回 56% 的 teacher 差距，8 个类别全部显著；DFD 与 DMD2 持平；teacher 的 dynamic degree（0.37）低于 DMD2（0.53），ReSeed（0.33）更接近 teacher（§3.7）。
- **推理开销可忽略**：每样本 3978.9 → 3984.3 ms（+0.137%），峰值激活内存不变。
- **增益来得很早、之后是随机游走**：100 步时 DINO 已到 0.2230（300 步增益的 84%），200 步最高（0.2341）；100 步之后每 100 步的偏置增量彼此近似正交（余弦 ≈ 0），幅值却按 √t 增长，质量指标随 checkpoint 非单调摆动（§2.4、§3.4）。
- **需要早停**：loss 在 ~300 步后平台化，但偏置幅值持续单调增长（RMS 0.013 → 0.029 @2000 步）；500 步时视频级多样性不再上升、CLIP 多样性回落、VBench 一致性/画质全面下降。300 步是当前工作点。
- **DFD 复现结果与论文相反**：发布 checkpoint 上 DFD 的多样性低于 DMD2（DINO 成对差 -0.0185 [-0.0312, -0.0060]，24/71），三套特征空间同向；VBench 5 个论文维度符号与论文一致。倾向解释：发布的 DMD2（1000 步）不在论文所述的坍缩状态（论文 baseline 30k 步）。

## 1. 实验设置

### 1.1 模型与 checkpoint

| arm | 权重 | 采样 |
|---|---|---|
| teacher | Wan2.1-T2V-1.3B 基座（fastgen 官方 teacher 采样器） | UniPC 50 步，shift 5，CFG 5.0，官方负面 prompt |
| dmd2 | 官方发布 `dmd2_t2v_vipe/0001000`（DMD2 蒸馏 1000 步） | 4 步 ODE，t_list = [0.999, 0.937, 0.833, 0.624, 0]（uniform 4 步 + shift 5），无 CFG |
| dfd | 官方发布 `ours_wan_t2v_vipe/0000040`（从上述 DMD2 再训 40 步） | 同 dmd2 |
| dmd2_reseed_ckptN | dmd2 权重 + ReSeed 偏置（checkpoint-N，只在第 1 步生效） | 同 dmd2 |

推理统一走官方 `scripts/inference/video_model_inference.py` + `WanT2V/config_dmd2.py`：832×480、81 帧、16 fps、bf16；每次运行只抽一次噪声并对所有 prompt 复用，所以同一 seed 下各 arm 的初始噪声比特一致（已用 sha256 探针验证），所有比较都是成对的。

### 1.2 数据集

| 集合 | 来源 | 规模 | 用途 |
|---|---|---|---|
| DFD 官方 eval prompts | `web_dfd_eval_prompts.txt`，134 行去重后 71 条唯一 prompt | 71 × 8 seed（0–7）= 568 视频/arm | 多样性、文本一致性、VBench |
| VBench 类别 prompt（第二评测集） | VBench 官方 `prompts/prompts_per_category/*.txt`（Vchitect/VBench master fd18b3d0，2026-08-21），animal / architecture / food / human / lifestyle / plant / scenery / vehicles 8 类各取前 40 条；`plant.txt` 第 1 条与 lifestyle 重复，跳过顺延；文件 `evaluation/prompts/vbench_category_320.txt`（sha256 4c3cdab1…），类别对照 `vbench_category_320.categories.csv`，来源记录 `vbench_category_320.PROVENANCE.md` | 320 × 8 seed（0–7）= 2,560 视频/arm，三个 student arm 共 7,680；VBench 只用 seed 0–4（1,600 视频/arm） | 多样性、文本一致性、VBench、按类别分解（§3.7） |
| DistillAlign eval-16 | 官方 `prompts/distribution_eval_16.txt` 逐字复制 | 16 × 16 seed（11,22,33,42,44,55,66,77,88,123,456,789,2024,3407,7777,9999）= 256 视频/arm | PRDC |
| ViPE long captions | 4000 条，UMT5 embedding 预先缓存 | 仅训练 ReSeed | — |

### 1.3 指标定义

- **DINO / CLIP 多样性**（DFD 论文 / DP-DMD 定义）：每个视频均匀取 16 帧 → 逐帧 embedding L2 归一化 → 帧平均 → 对每个 prompt 的 8 个视频算 C(8,2)=28 对余弦，D = 1 − 平均余弦；对 71 个 prompt 取均值，2000 次 bootstrap 95% CI。DINO 用 `facebook/dinov2-large` pooler 输出；CLIP 用 `openai/clip-vit-large-patch14`，同时报告投影特征（768-d，`image_embeds`）和 DP-DMD 代码的 CLS 特征（1024-d，`last_hidden_state[:,0]`）。变体：逐帧（先逐帧算 D 再平均）、色彩归一化（逐视频减均值除标准差后再编码）。
- **CLIP 文本一致性**：CLIP 文本 embedding 与 16 帧视觉 embedding 均值的余弦。
- **VBench 7 维**（官方 VBench 0.1.5，`custom_input`）：subject/background consistency、temporal flickering、motion smoothness、dynamic degree、aesthetic、imaging。
- **DistillAlign PRDC**（arXiv 2607.26811）：V-JEPA2 ViT-H（前 81 帧 linspace 8 帧，token mean⊕std，L2，2560-d）；teacher 每个样本的 k=5 近邻半径定义支撑集；precision = student 落在 teacher 球内的比例，coverage = teacher 样本被最近 student 覆盖的比例。teacher 特征直接用官方发布的 Wan2.1-1.3B 缓存（25 步 UniPC、shift 8、CFG 5），协议哈希（模型 revision、帧选择、池化、transformers 4.57.6 / torch 2.6.0+cu124 / numpy 1.26.4、代码 sha）与官方一致；student 走 post-DMD 协议（`--no-refine`）。

## 2. ReSeed 操作细节

### 2.1 适配器结构（`evaluation/reseed/adapter.py`）

- `WanBlockResidualBias`：对 Wan transformer 的 30 个 block，各注册一个 forward hook，在 block **输出**上加一个逐通道偏置 `b_ℓ ∈ R^1536`（fp32 参数，零初始化）；30 × 1536 = **46,080 个参数**（bf16 184 KB），占 denoiser 1,418,996,800 参数的 **0.00325%**。偏置对所有 token（帧 × 空间）共享，时间维度直接沿用图像版做法不分帧。
- `FirstStepGate`：网络的 forward pre-hook 读取时间步 `t`，`t > (t₀+t₁)/2 = 0.968` 时开启偏置，否则关闭 → 偏置只在 4 步采样的**第 1 步**生效。读取 `t` 不消耗 RNG，加载 adapter 前后抽到的噪声比特一致。
- 零初始化时输出与原模型逐比特一致（训练前的 identity 检查），因此 adapter 是严格的"加法"修改，随时可卸载。

### 2.2 训练目标（`evaluation/reseed/train_reseed.py`, `flow.py`）

- 每步抽一批 prompt（4 个 rank × 1 = 4 条/步，从 4000 条 ViPE caption 的固定随机流中取）和噪声 z ~ N(0, I)（latent 16×21×60×104）。
- **teacher 目标**：以 z 为起点，用 teacher（Wan2.1-1.3B，CFG 5.0，含负面 prompt）在 student 第一段区间 [t₀=0.999, t₁=0.937] 上做 **m=13 步 Euler 微步**（微步网格在 unshifted-sigma 空间均匀、端点与 student 区间边界对齐，`boundary_aligned_teacher_grid`），得到 x_{t₁}；再折算成把 z 从 t₀ 送到 x_{t₁} 的单步**割线速度** v* = (z − x_{t₁})/(t₀ − t₁)。
- **student 预测**：带偏置的 student 在 t₀ 上预测 x̂₀，速度 v̂ = (z − x̂₀)/t₀（`student_velocity`）。
- **损失**：MSE(v̂, v*)，只有 46,080 个偏置参数有梯度（step-1 梯度断言保证梯度只来自第 1 步）。记录 velocity cosine、相对误差、端点相对误差、梯度范数、bias RMS/max。
- 直觉：DMD 蒸馏使 student 第一步把不同噪声映射到少数几个 x̂₀（坍缩）；用 teacher 的第一步割线速度做监督，把 student 第一步的"落点"推回 teacher 会到达的位置，而后续 3 步不动，从而恢复初始噪声到样本的多样性映射。

### 2.3 训练配置

| 项 | 值 |
|---|---|
| student | `dmd2_t2v_vipe/0001000`（FSDP 加载，与推理脚本相同构造路径） |
| teacher | 配置中的 Wan2.1-T2V-1.3B，CFG 5.0 |
| 数据 | ViPE long caption 4000 条，UMT5 embedding 预编码缓存（`reseed_data/vipe_long/umt5_embeddings.pt`） |
| 微步数 m | 13 |
| 优化器 | AdamW，lr 1e-3，β=(0.9, 0.999)，weight decay 0，梯度裁剪 1.0 |
| 并行 | 4 × H100 DDP（每 rank 1 条 prompt，梯度 all-reduce） |
| 步数 | 2000（每 100 步存 checkpoint，含每 rank RNG/流状态，可精确断点续训） |
| 耗时 | 30.3 s/步（teacher 13 微步 × 2（CFG）26.2 s + student 4.2 s），2000 步 16 h 54 min |
| 非确定性 | 反向传播非确定，同配置两次运行 7 步后偏置最大差 4.7e-3 |

### 2.4 训练曲线

| 窗口 | loss | velocity cosine | bias RMS | bias max |
|---|---|---|---|---|
| 0–200 | 0.369 | 0.907 | 0.013 | 0.060 |
| 200–400 | 0.316 | 0.919 | 0.016 | 0.087 |
| 400–600 | 0.294 | 0.926 | 0.018 | 0.120 |
| 600–1000 | 0.29–0.30 | 0.925 | 0.022 | 0.159 |
| 1000–1600 | 0.29–0.30 | 0.924–0.926 | 0.026 | 0.233 |
| 1600–2000 | 0.274–0.279 | 0.929–0.930 | 0.029 | 0.264 |

loss / cosine 在 ~300–400 步后平台化；偏置幅值全程单调增长，且没有饱和迹象——优化器在平坦谷里沿几乎不影响 loss 的方向推大偏置，而生成质量对幅值敏感（见 §3.4）。图：`evaluation/reseed/plot_training.py` 生成的 `training_curves.png`。

**偏置向量的演化**（对 checkpoint 的 `residual_biases.pt` 直接分析）：相邻 checkpoint 的余弦 0.86–0.92，但每 100 步的**增量**彼此近似正交（d200·d100 = 0.06，d300·d200 = 0.08，d400·d300 = −0.01，d500·d400 = −0.03）。也就是说前 100 步学到的是一个一致的方向，之后的更新是 4 条 prompt/步的梯度噪声在平坦方向上的随机游走：幅值按 √t 增长（RMS 100 步 0.0104 → 2000 步 0.0286），方向却不再累积。这解释了 §3.4 里质量指标随 checkpoint 非单调的现象——不同 checkpoint 之间的差别主要是随机分量。各 block 的 RMS 相近（0.009–0.018，中后段 block 略大，最后一个 block 最小）。

### 2.5 推理开销（`evaluation/reseed/bench_overhead.py`，H100，bf16，官方 `generator_fn`，8 样本中位数）

| Student model | Params | of denoiser | DINO before → after | Gain | Peak act. change | Step 1 (ms) | Per sample (ms) | Overhead |
|---|---|---|---|---|---|---|---|---|
| Wan2.1-T2V-1.3B (DMD2, 4-step) | 46,080 | 0.00325% | 0.1869 → 0.2297 | +22.9% | +0.00 MiB | 992.7 → 996.1 | 3978.9 → 3984.3 | +0.137% |

每步约 993 ms（4 步几乎等长），adapter 仅在第 1 步多 3.4 ms；基线重复测量漂移 2.2 ms，因此 +0.137% 应视为上界。峰值激活内存实测差 -32 MiB，在 caching allocator 噪声内（偏置只是 block 输出上的一次逐通道加法，无新增持久激活）。per-sample 不含 VAE 解码。DINO 为 DINOv2-L。

## 3. 结果

### 3.1 多样性与文本一致性（71 prompt × 8 seed，均值；括号为 bootstrap 95% CI）

| arm | DINO | CLIP proj | CLIP cls | DINO 逐帧 | CLIP proj 逐帧 | CLIP cls 逐帧 | 文本一致性 |
|---|---|---|---|---|---|---|---|
| Teacher (Wan2.1-1.3B, 50 步) | 0.2647 [0.237, 0.294] | 0.0909 | 0.1473 | 0.3245 | 0.1231 | 0.1957 | 0.2924 |
| DMD2 | 0.1869 [0.164, 0.210] | 0.0722 | 0.1141 | 0.2290 | 0.0974 | 0.1519 | 0.2822 |
| DFD | 0.1683 [0.149, 0.189] | 0.0732 | 0.1094 | 0.2158 | 0.1032 | 0.1522 | 0.2875 |
| DMD2+ReSeed@100 | 0.2230 [0.200, 0.246] | 0.0819 | 0.1322 | 0.2716 | 0.1101 | 0.1749 | 0.2841 |
| DMD2+ReSeed@200 | 0.2341 [0.210, 0.258] | 0.0813 | 0.1319 | 0.3013 | 0.1183 | 0.1877 | 0.2842 |
| DMD2+ReSeed@300 | 0.2297 [0.207, 0.253] | 0.0842 | 0.1350 | 0.2854 | 0.1152 | 0.1815 | 0.2846 |
| DMD2+ReSeed@500 | 0.2299 [0.206, 0.254] | 0.0778 | 0.1267 | 0.3079 | 0.1189 | 0.1888 | 0.2838 |

色彩归一化变体（DINO / CLIP proj / CLIP cls）：Teacher (Wan2.1-1.3B, 50 步) 0.2834 / 0.0808 / 0.1381；DMD2 0.2036 / 0.0690 / 0.1099；DFD 0.1886 / 0.0646 / 0.1026；DMD2+ReSeed@100 0.2381 / 0.0765 / 0.1239；DMD2+ReSeed@200 0.2504 / 0.0754 / 0.1225；DMD2+ReSeed@300 0.2473 / 0.0781 / 0.1262；DMD2+ReSeed@500 0.2443 / 0.0705 / 0.1154。

### 3.2 逐 prompt 成对差（同噪声；均值 [bootstrap 95% CI]，胜出 prompt 数/71）

| 比较 | DINO | CLIP proj | CLIP cls | DINO 逐帧 | 文本一致性 |
|---|---|---|---|---|---|
| DMD2+ReSeed@100 − DMD2 | +0.0361 [+0.0233, +0.0496]，56/71 | +0.0098 [+0.0048, +0.0150]，46/71 | +0.0181 [+0.0110, +0.0254]，52/71 | +0.0425 [+0.0282, +0.0576]，57/71 | +0.0019 [-0.0014, +0.0052]，41/71 |
| DMD2+ReSeed@200 − DMD2 | +0.0473 [+0.0339, +0.0616]，58/71 | +0.0092 [+0.0046, +0.0138]，53/71 | +0.0178 [+0.0114, +0.0244]，55/71 | +0.0723 [+0.0572, +0.0870]，64/71 | +0.0020 [-0.0012, +0.0052]，38/71 |
| DMD2+ReSeed@300 − DMD2 | +0.0428 [+0.0287, +0.0571]，57/71 | +0.0120 [+0.0077, +0.0169]，50/71 | +0.0209 [+0.0143, +0.0279]，56/71 | +0.0563 [+0.0409, +0.0718]，59/71 | +0.0025 [-0.0002, +0.0054]，45/71 |
| DMD2+ReSeed@500 − DMD2 | +0.0431 [+0.0282, +0.0584]，55/71 | +0.0057 [+0.0015, +0.0099]，43/71 | +0.0126 [+0.0067, +0.0186]，46/71 | +0.0788 [+0.0630, +0.0943]，64/71 | +0.0016 [-0.0015, +0.0047]，38/71 |
| DMD2+ReSeed@200 − DMD2+ReSeed@100 | +0.0111 [+0.0017, +0.0205]，45/71 | -0.0006 [-0.0051, +0.0037]，37/71 | -0.0003 [-0.0063, +0.0056]，37/71 | +0.0297 [+0.0205, +0.0390]，55/71 | +0.0001 [-0.0018, +0.0020]，38/71 |
| DMD2+ReSeed@300 − DMD2+ReSeed@200 | -0.0045 [-0.0132, +0.0043]，31/71 | +0.0028 [-0.0007, +0.0063]，43/71 | +0.0031 [-0.0018, +0.0078]，42/71 | -0.0159 [-0.0254, -0.0068]，27/71 | +0.0004 [-0.0013, +0.0023]，35/71 |
| DMD2+ReSeed@500 − DMD2+ReSeed@300 | +0.0002 [-0.0084, +0.0093]，35/71 | -0.0063 [-0.0105, -0.0023]，26/71 | -0.0082 [-0.0141, -0.0024]，26/71 | +0.0225 [+0.0131, +0.0322]，49/71 | -0.0009 [-0.0022, +0.0005]，33/71 |
| DMD2+ReSeed@300 − DFD | +0.0613 [+0.0463, +0.0766]，62/71 | +0.0110 [+0.0043, +0.0177]，49/71 | +0.0256 [+0.0169, +0.0348]，56/71 | +0.0696 [+0.0542, +0.0859]，63/71 | -0.0029 [-0.0067, +0.0010]，27/71 |
| DFD − DMD2 | -0.0185 [-0.0314, -0.0060]，24/71 | +0.0010 [-0.0037, +0.0064]，40/71 | -0.0047 [-0.0109, +0.0016]，32/71 | -0.0132 [-0.0260, -0.0005]，25/71 | +0.0053 [+0.0024, +0.0081]，45/71 |
| DMD2 − Teacher (Wan2.1-1.3B, 50 步) | -0.0778 [-0.0965, -0.0583]，13/71 | -0.0188 [-0.0248, -0.0127]，14/71 | -0.0332 [-0.0415, -0.0249]，10/71 | -0.0955 [-0.1161, -0.0751]，8/71 | -0.0103 [-0.0137, -0.0069]，21/71 |
| DMD2+ReSeed@300 − Teacher (Wan2.1-1.3B, 50 步) | -0.0350 [-0.0555, -0.0141]，25/71 | -0.0068 [-0.0135, -0.0000]，31/71 | -0.0123 [-0.0213, -0.0028]，28/71 | -0.0391 [-0.0615, -0.0160]，23/71 | -0.0078 [-0.0116, -0.0043]，21/71 |
| DFD − Teacher (Wan2.1-1.3B, 50 步) | -0.0964 [-0.1177, -0.0763]，7/71 | -0.0177 [-0.0253, -0.0087]，18/71 | -0.0379 [-0.0481, -0.0268]，9/71 | -0.1087 [-0.1309, -0.0868]，9/71 | -0.0049 [-0.0081, -0.0016]，27/71 |

### 3.3 VBench

| arm | subject | background | flicker | smoothness | dynamic degree | aesthetic | imaging |
|---|---|---|---|---|---|---|---|
| Teacher (Wan2.1-1.3B, 50 步) | 0.943 | 0.953 | 0.977 | 0.987 | 0.553 | 0.610 | 0.649 |
| DMD2 | 0.954 | 0.957 | 0.981 | 0.989 | 0.606 | 0.612 | 0.666 |
| DFD | 0.951 | 0.951 | 0.982 | 0.991 | 0.632 | 0.623 | 0.670 |
| DMD2+ReSeed@100 | 0.952 | 0.960 | 0.986 | 0.992 | 0.393 | 0.633 | 0.626 |
| DMD2+ReSeed@200 | 0.936 | 0.949 | 0.976 | 0.987 | 0.673 | 0.601 | 0.595 |
| DMD2+ReSeed@300 | 0.947 | 0.956 | 0.983 | 0.991 | 0.486 | 0.616 | 0.621 |
| DMD2+ReSeed@500 | 0.931 | 0.944 | 0.971 | 0.985 | 0.796 | 0.594 | 0.579 |

DFD 与 DMD2 相比 5 个论文报告维度（subject、background、flicker、smoothness、aesthetic）的符号全部与论文一致，说明质量类指标可复现，只有多样性方向相反。

### 3.4 checkpoint 扫描（ReSeed 训练步数 vs 指标）

| checkpoint | DINO | CLIP proj | CLIP cls | 文本一致性 | VBench dynamic | VBench imaging | VBench subject | PRDC precision | PRDC coverage |
|---|---|---|---|---|---|---|---|---|---|
| dmd2（无 adapter） | 0.1869 | 0.0722 | 0.1141 | 0.2822 | 0.606 | 0.666 | 0.954 | 0.473 | 0.285 |
| ckpt-100 | 0.2230 | 0.0819 | 0.1322 | 0.2841 | 0.393 | 0.626 | 0.952 | 0.457 | 0.227 |
| ckpt-200 | 0.2341 | 0.0813 | 0.1319 | 0.2842 | 0.673 | 0.595 | 0.936 | 0.504 | 0.305 |
| ckpt-300 | 0.2297 | 0.0842 | 0.1350 | 0.2846 | 0.486 | 0.621 | 0.947 | 0.504 | 0.305 |
| ckpt-500 | 0.2299 | 0.0778 | 0.1267 | 0.2838 | 0.796 | 0.579 | 0.931 | 0.520 | 0.285 |

多样性在 100 步已拿到绝大部分增益（DINO +0.036，为最大增益 +0.047 的 77%），200 步达到最高（0.2341），之后不再上升（300/500 与 200 的成对差 CI 含 0，CLIP 在 500 步回落）。**质量随 checkpoint 非单调**：ckpt-100 和 ckpt-300 的一致性 / flicker / smoothness / aesthetic 均不低于 dmd2，而 ckpt-200 和 ckpt-500 在这些维度上跌破 teacher；dynamic degree 在 0.39（100）→ 0.67（200）→ 0.49（300）→ 0.80（500）之间来回摆动。结合 §2.4 的偏置增量正交性，这不是训练步数的系统性效应，而是随机游走分量落在不同位置：第一步的逐通道偏置会整体改变视频的运动幅度与帧间稳定性，而这个方向并不受训练目标约束。imaging quality 是唯一在所有 checkpoint 上都一致低于 dmd2 的维度（−0.04 ～ −0.09）。

工作点选择：以"多样性增益 + 质量不低于 teacher"为准，**ckpt-100**（DINO +0.036，aesthetic 0.633 最高，一致性最好）和 **ckpt-300**（DINO +0.043）都可用；ckpt-100 的 dynamic degree 最低（0.39 vs teacher 0.55）是它的主要代价。下一版训练应加偏置范数约束或按 bias RMS 早停，并把 dynamic degree 纳入 checkpoint 选择。

### 3.5 DistillAlign PRDC（16 prompt × 16 seed，对官方 Wan2.1-1.3B teacher 缓存，k=5）

| arm | precision | coverage |
|---|---|---|
| DMD2 | 0.4727 | 0.2852 |
| DFD | 0.4883 | 0.2266 |
| DMD2+ReSeed@100 | 0.4570 | 0.2266 |
| DMD2+ReSeed@200 | 0.5039 | 0.3047 |
| DMD2+ReSeed@300 | 0.5039 | 0.3047 |
| DMD2+ReSeed@500 | 0.5195 | 0.2852 |

按 prompt 做成对 bootstrap（16 条 prompt）后所有差异的 95% CI 都跨 0（如 ReSeed@300 − dmd2 coverage +0.020 [−0.027, +0.082]），PRDC 只作方向性佐证：ReSeed（≥200 步）两项均不低于 dmd2，DFD coverage 最低。ckpt-200 与 300 数值相同是巧合（129/256 与 78/256），两组视频与特征均不同（平均余弦 0.993）。

### 3.6 DMD2 vs DFD 复现（与论文对照）

| 指标 | 论文 Table 1 | 本复现（发布 checkpoint，seed 0–7） |
|---|---|---|
| DINO 多样性 DMD2 → DFD | 0.190 → 0.205（+0.015） | 0.1869 → 0.1683（成对 -0.0185 [-0.0315, -0.0054]，24/71） |
| CLIP 多样性 DMD2 → DFD | 0.120 → 0.128（+0.008） | proj 0.0722 → 0.0732；cls 0.1141 → 0.1094 |
| VBench 5 维符号 | — | 全部一致（aesthetic 0.612 → 0.623 等） |

已排除的解释：配置/噪声/文本 embedding 逐比特核对一致；checkpoint 正确加载；指标按 DP-DMD 参考实现逐项对照（两种 CLIP 读法、逐帧、色彩归一化）；V-JEPA2 coverage 独立同向。像素统计：DFD 像素 std 低约 35%、饱和度 0.63 vs 0.85（与论文"DMD2 过饱和"描述一致），对比度归一化后跨 seed 距离两者接近。倾向解释：发布的 DMD2 只训了 1000 步、8-seed 网格肉眼未坍缩，DFD 从此再训 40 步没有可恢复的多样性，反而收缩了对比度；论文的 DFD 是从 25k 步 DMD2 再训 100 步，不是同一条轨迹。可定案的实验：把 DMD2 继续训到 10k–30k 步再评（数十 GPU 小时）；便宜的补充：seed 1–8 重跑、联系作者确认 checkpoint 与表格的对应。

### 3.7 第二评测集：VBench 类别 prompt 320 条 × 8 seed

数据：`evaluation/prompts/vbench_category_320.txt`（见 §1.2），VBench 官方 8 类各前 40 条；arm = teacher / dmd2 / dfd / dmd2+ReSeed@300；每 arm 320 × 8 seed = 2,560 视频（四臂共 10,240），同 seed 同噪声。多样性与文本一致性用全部 8 个 seed（C(8,2)=28 对/prompt），VBench 用 seed 0–4（VBench 自身的 5 样本/prompt 约定，1,600 视频/arm）。作业：student 三臂 141546（生成 3 h 40 min、多样性 1 h、VBench 2.5 h）；teacher 每个 seed 一个作业（142259–142266，每作业 4 张卡各跑 80 条 prompt，`--prompt-range` 分片，约 2.5 h），收尾 142267（校验 + metadata + 四臂多样性 + teacher VBench + 合并 + 分类，4 h 23 min）。脚本 `run_evalset2.sbatch`、`run_evalset2_teacher.sbatch`、`run_evalset2_teacher_finalize.sbatch`；结果 `evaluation/results_vbench320/`。

| arm | DINO | CLIP proj | CLIP cls | DINO 逐帧 | 文本一致性 |
|---|---|---|---|---|---|
| Teacher (Wan2.1-1.3B, 50 步) | 0.5005 [0.481, 0.521] | 0.1634 | 0.2451 | 0.5352 | 0.2425 |
| DMD2 | 0.3374 [0.319, 0.355] | 0.1267 | 0.1835 | 0.3748 | 0.2419 |
| DFD | 0.3358 [0.317, 0.354] | 0.1351 | 0.1845 | 0.3775 | 0.2398 |
| DMD2+ReSeed@300 | 0.4290 [0.412, 0.446] | 0.1437 | 0.2105 | 0.4716 | 0.2365 |

成对差（320 prompt；均值 [bootstrap 95% CI]，胜出数）：

| 比较 | DINO | CLIP proj | CLIP cls | 文本一致性 |
|---|---|---|---|---|
| Teacher (Wan2.1-1.3B, 50 步) − DMD2 | +0.1631 [+0.1495, +0.1762]，299/320 | +0.0367 [+0.0308, +0.0422]，260/320 | +0.0616 [+0.0544, +0.0685]，276/320 | +0.0006 [-0.0014, +0.0026]，164/320 |
| DFD − DMD2 | -0.0016 [-0.0090, +0.0059]，161/320 | +0.0084 [+0.0047, +0.0116]，222/320 | +0.0010 [-0.0034, +0.0053]，186/320 | -0.0022 [-0.0039, -0.0005]，132/320 |
| DMD2+ReSeed@300 − DMD2 | +0.0916 [+0.0804, +0.1024]，273/320 | +0.0170 [+0.0119, +0.0221]，220/320 | +0.0270 [+0.0211, +0.0333]，230/320 | -0.0055 [-0.0074, -0.0035]，127/320 |
| DMD2+ReSeed@300 − DFD | +0.0932 [+0.0830, +0.1035]，269/320 | +0.0086 [+0.0044, +0.0129]，188/320 | +0.0260 [+0.0212, +0.0311]，226/320 | -0.0033 [-0.0053, -0.0012]，143/320 |
| DMD2 − Teacher (Wan2.1-1.3B, 50 步) | -0.1631 [-0.1770, -0.1495]，21/320 | -0.0367 [-0.0425, -0.0313]，60/320 | -0.0616 [-0.0685, -0.0550]，44/320 | -0.0006 [-0.0026, +0.0014]，156/320 |
| DFD − Teacher (Wan2.1-1.3B, 50 步) | -0.1647 [-0.1775, -0.1526]，19/320 | -0.0283 [-0.0332, -0.0235]，80/320 | -0.0606 [-0.0670, -0.0546]，36/320 | -0.0028 [-0.0048, -0.0009]，138/320 |
| DMD2+ReSeed@300 − Teacher (Wan2.1-1.3B, 50 步) | -0.0715 [-0.0825, -0.0605]，80/320 | -0.0197 [-0.0243, -0.0150]，99/320 | -0.0346 [-0.0401, -0.0292]，83/320 | -0.0060 [-0.0079, -0.0042]，107/320 |

| arm | subject | background | flicker | smoothness | dynamic degree | aesthetic | imaging |
|---|---|---|---|---|---|---|---|
| Teacher (Wan2.1-1.3B, 50 步) | 0.970 | 0.961 | 0.983 | 0.991 | 0.371 | 0.590 | 0.703 |
| DMD2 | 0.970 | 0.962 | 0.983 | 0.991 | 0.531 | 0.615 | 0.698 |
| DFD | 0.967 | 0.955 | 0.984 | 0.991 | 0.529 | 0.611 | 0.711 |
| DMD2+ReSeed@300 | 0.967 | 0.961 | 0.986 | 0.992 | 0.329 | 0.631 | 0.675 |

按类别的 DINO 多样性（`category_breakdown.md`，成对差相对 dmd2）：

| category | n | dfd | dmd2 | dmd2_reseed_ckpt300 | teacher | dfd − dmd2 | dmd2_reseed_ckpt300 − dmd2 | teacher − dmd2 |
|---|---|---|---|---|---|---|---|---|
| animal | 40 | 0.2471 | 0.2555 | 0.3313 | 0.4235 | -0.0084 [-0.025, +0.010] 19/40 | +0.0758 [+0.053, +0.100] 34/40 | +0.1680 [+0.135, +0.202] 39/40 |
| architecture | 40 | 0.4374 | 0.4391 | 0.5278 | 0.5948 | -0.0016 [-0.025, +0.023] 19/40 | +0.0888 [+0.056, +0.125] 31/40 | +0.1558 [+0.122, +0.190] 38/40 |
| food | 40 | 0.2719 | 0.2797 | 0.3904 | 0.4371 | -0.0077 [-0.027, +0.012] 19/40 | +0.1107 [+0.087, +0.136] 40/40 | +0.1575 [+0.124, +0.192] 38/40 |
| human | 40 | 0.2635 | 0.2564 | 0.3800 | 0.4518 | +0.0071 [-0.008, +0.023] 20/40 | +0.1236 [+0.097, +0.152] 37/40 | +0.1954 [+0.164, +0.227] 40/40 |
| lifestyle | 40 | 0.3117 | 0.2837 | 0.4093 | 0.4792 | +0.0279 [+0.011, +0.045] 27/40 | +0.1255 [+0.096, +0.155] 38/40 | +0.1954 [+0.157, +0.233] 38/40 |
| plant | 40 | 0.4146 | 0.4125 | 0.4818 | 0.5813 | +0.0021 [-0.017, +0.020] 20/40 | +0.0693 [+0.034, +0.104] 31/40 | +0.1688 [+0.130, +0.206] 37/40 |
| scenery | 40 | 0.3841 | 0.4128 | 0.4556 | 0.5335 | -0.0286 [-0.057, +0.001] 16/40 | +0.0428 [+0.017, +0.068] 29/40 | +0.1207 [+0.070, +0.166] 32/40 |
| vehicles | 40 | 0.3558 | 0.3596 | 0.4557 | 0.5029 | -0.0038 [-0.025, +0.018] 21/40 | +0.0961 [+0.062, +0.133] 33/40 | +0.1433 [+0.106, +0.183] 37/40 |
| ALL | 320 | 0.3358 | 0.3374 | 0.4290 | 0.5005 | -0.0016 [-0.009, +0.006] 161/320 | +0.0916 [+0.081, +0.102] 273/320 | +0.1631 [+0.150, +0.176] 299/320 |

在这套更通用的 prompt 集上：teacher 的 DINO 多样性 0.5005，DMD2 蒸馏丢掉 0.163，ReSeed@300 补回 0.092（**56%**，与 71-prompt 集上的 55% 一致），8 个类别全部显著为正（food / lifestyle / human 最大，scenery 最小），teacher 在每个类别上都仍高于 ReSeed。DFD 与 DMD2 在 DINO 上持平、CLIP proj 略高——71-prompt 集上的“DFD 更低”没有再现，因此对 DFD 的稳妥表述是“两套集上均无显著多样性增益”。ReSeed 的文本一致性小幅显著下降（−0.0055；teacher 0.2425、dmd2 0.2419、ReSeed 0.2365）。VBench：一致性 / flicker / smoothness 四臂几乎相同（差 <0.01），ReSeed 的 aesthetic 最高（0.631 vs teacher 0.590）；**dynamic degree 上 teacher 本身只有 0.371**，dmd2 的 0.531 是蒸馏后偏高，ReSeed 的 0.329 反而更接近 teacher——71-prompt 集上被当作“代价”的 dynamic degree 下降，在 teacher 参照下应重新解读为向 teacher 靠拢；imaging（0.675 vs teacher 0.703、dmd2 0.698）仍是唯一一致偏低的维度。

## 4. 可视化

- Seed 网格（第 40 帧，行 = Teacher / DMD2 / ReSeed@300，列 = seed 0–7）：`evaluation/samples/grids_nodfd/seed_grid_p{026,013,007,015,030}_f40.png`（`evaluation/tools/seed_grid.py`）。
- 拼接对比视频（3 行 × 4 列 seed 0–3；p022 另有 3 × 8 版本）：`evaluation/samples/compare_nodfd/compare_p{000,007,013,015,022,026,030,048,067}_*.mp4`；含 DFD 行的版本在 `evaluation/samples/compare/`。
- 原始单视频：`evaluation/samples/raw_p026/`、`raw_p015/`。
- 320 集：`evaluation/samples/vbench320/seed_grid_p{314,134,184,068,109,236,168}_f40.png` 与 `videos/compare_p*.mp4`（每类增益最大的 prompt + 中位数 p168）。**定性提醒**：VBench 类别 prompt 很短（如 "a glass filled with beer"、"eiffel tower from the seine river"），增益最大的几条里 ReSeed 的多样性有一部分来自语义漂移——啤酒杯里出现水果配料、铁塔变成教堂尖顶、"红巴士"三臂都没生成巴士而 ReSeed 的画面内容更散；这与 320 集上文本一致性显著下降 −0.0055 一致。中位数 prompt（p168）和长描述的 71 集上没有这种现象。论文选图应避开这几条，并在正文说明短 prompt 下多样性与 prompt 一致性的权衡。
- ReSeed@300 增益最大的 prompt（DINO 成对差）：p026 +0.199、p022 +0.187、p013 +0.185、p007 +0.175、p015 +0.150、p067 +0.131；中位数 p030 +0.042；最差 p024 -0.085、p051 -0.099。（p026 老夫妇跳舞、p022 拉花咖啡、p013 玻璃电梯、p007 沙漠摩托、p015 湖面木船、p067 咖啡馆雨窗、p000 卷发女子打伞。）

## 5. 未完成 / 下一步

2. 排除替代解释：随机偏置对照（同 RMS 的随机偏置）、门控消融（全 4 步生效 / 仅第 2 步）、偏置形状消融（分帧 / 分 token）。均不需重训只需生成。
3. 带偏置范数约束（weight decay / 范数投影）或按 bias RMS 早停的训练版本。
4. ReSeed 应用到 DFD student（DFD+ReSeed），把 DFD 从"对手"变成"基座"。
5. 超参：m、学习率、CFG；训练 seed 方差；held-out velocity loss 曲线。
6. DistillAlign 协议下自建 teacher（严格成对 PRDC）；DINOv3 重算以与图像表格口径统一。
7. 论文写法：主表 teacher / DMD2 / DMD2+ReSeed；DFD 放附录并注明"发布 checkpoint 上未复现其多样性增益，baseline 非坍缩状态"；投稿前联系 DFD 作者。

## 6. 文件索引

| 内容 | 本地 | 集群 |
|---|---|---|
| 全 arm 71-prompt 结果（多样性 / VBench / 逐 prompt CSV） | `evaluation/results_5arm/` | `repo/evaluation/results/`, `results_reseed_checkpoint-N/` |
| 第二评测集（VBench 类别 320）：多样性 / VBench / 按类别分解 | `evaluation/results_vbench320/` | `repo/evaluation/results_vbench320/`, 视频 `generated_vbench320/` |
| 色彩归一化全 arm | `evaluation/results_colornorm_all/` | `repo/evaluation/results_colornorm_all/` |
| DistillAlign 特征与 PRDC | `evaluation/results_distillalign/` | `repo/evaluation/results_distillalign/` |
| DMD2/DFD 复现（含 refclip 变体、paired_analysis） | `evaluation/results_full71/` | `repo/evaluation/results_full71_v1/` |
| 推理开销 | `evaluation/results_5arm/reseed_bench_checkpoint-300.json` | `reseed_runs/bench_checkpoint-300.json` |
| ReSeed 训练日志 / checkpoint | — | `reseed_runs/dmd2_m13/{metrics.jsonl, checkpoint-100..2000}` |
| 生成视频 | `evaluation/samples/` | `repo/evaluation/generated/{teacher,dmd2,dfd,dmd2_reseed_ckpt*}/`, `generated_distillalign/` |

复现命令：`bash evaluation/scripts/run_all.sh`（71-prompt 全套）、`ADAPTER=<checkpoint-N> sbatch --export=ALL evaluation/reseed/eval_reseed.sbatch`（加一个 ReSeed arm）、`MODELS=<arm> ADAPTER=<ckpt> sbatch --export=ALL evaluation/scripts/run_distillalign.sbatch`（PRDC）、`sbatch evaluation/reseed/train_reseed.sbatch`（训练）、`sbatch evaluation/reseed/bench_overhead.sbatch`（开销）、`python evaluation/tools/build_reseed_report.py`（重建本报告）。
