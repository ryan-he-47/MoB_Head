# 28. MoB 头在大词表小模型上的训练、评估与对比研究——阶段性收官报告

**日期**：2026-09-20　**状态**：推理加速臂阶段性收尾（docs/16 §8 裁定）　**配套**：评测细节 docs/27、计划 docs/26、撞车分析 docs/25、代码架构 docs/13

---

## 摘要

针对端侧小模型输出层的带宽瓶颈（V=248,320 时 lm_head 每步读取 1017 MiB），本工作将 LLM 的反嵌入层替换为**语义二叉树码 + 混合伯努利（MoB）密度头**：词表中每个 token 映射到 18 维超立方体顶点，一个小型门控 FFN 头输出 K=128 个逐位伯努利分量与混合权重，构成全词表归一化密度。在 Qwen3.5-2B（指令版）上以四源 2B token 混合语料做精确教师加权蒸馏（92% 进度），并与当前最强的免训练粗-细加速头 CFHead（IEEE TC 2026）做了三套口径的严格对打。**结论：在等带宽预算（~95 MiB/步）下，CFHead 在 top-1 保真、候选集分布质量两项以 1~2pt / 2~8pt 胜出且免训练；MoB 在紧预算（32 候选）下以逐位概率排名结构赢 EN/ZH +2.5~3.1pt，并提供对方结构性缺席的全词表归一化密度与无原头可用性；双方 CPU-offload 拆分延迟同量级（~4.4ms/步）。** 推理加速臂据此阶段性停止，资产归档，主线转向端到端蒸馏。

## 1. 模型与训练

### 1.1 语义二叉树码本

词表 V=248,320 → M=⌈log2 V⌉=18 位二进制码（18 维单位超立方体的顶点，2^18=262,144 含 13,824 个未用顶点）。码的分配通过在模型自有 tied 嵌入表上自顶向下递归二分：**L2 归一化到单位球面 → 2-means（余弦，3 重启+最远点初始化）取质心差方向 → 按该方向投影的中位数平衡切分**（裸 2-means 切簇在锥形嵌入上系统性歪斜，docs/04 §3.1——只取方向不切簇是修正）。语义邻近的 token 获得 Hamming 距离近的码（Hamming-余弦相关实测 + 最近邻距离对照，docs/04）。码本 `data/codebook_qwen35.npz`（18 位，含 vertex_id 映射）。

### 1.2 MoB 头架构

SwiGLU 门控 FFN 头（与 LLM 内置 FFN 同构，赛马冠军架构 docs/16 §5）：h(2048) → pre-norm RMSNorm → W3(SiLU(xW1)·(xW2))（hidden=7300）残差块 ×1 → 终 norm → 线性读出 K×M=128×18 位 logits + K=128 个 π logits。**参数量 49.9M（fp16 95.0 MiB ≈ CFHead-T4 索引图 95.3 MiB——等带宽对照的基准）**。密度：p(v)=Σ_k π_k Π_m Ber(v_m; σ(z_km))，对全部 2^18 顶点归一化。

### 1.3 蒸馏配置与实施

- 教师/基座：Qwen3.5-2B（指令版，Gated DeltaNet 混合架构，hidden_states[-1] 为 lm_head 输入点，TAP_OK 验证 max rel 0.39%），全程冻结。
- 数据：四源 2.0B token 文档级交错——EN fineweb-edu 1.00B / ZH CCI3-HQ 0.60B / Code stack-v3 0.30B / 指令 smoltalk 0.10B（50/30/15/5），四源 held-out dev 各 ~4M token。
- 损失：教师 top-1024 下一词分布的**精确概率加权 NLL**（无采样无噪声；loss-chunk 2048 控制 V×topk logits 显存），π 熵正则，DAEM 温度 2→1 退火，k-means-顶点初始化位 logits（V 边缘化采样，规避 multinomial 2^24 上限）。
- 日程：余弦 1e-3（batch-16 配方，warmup 4000），fp16+GradScaler（V100），head-latest 每 20k 步轮转归档。
- **进度：224k/244k 步（92%）**，3 天 slurm 时限到期停止。终 panel：round 解码 dev top-1 一致率 0.7505、dev NLL 3.256、n_eff 76.5/128、0 死分量。**本报告所有 MoB 数字均为该 92% 进度头的下界。**

### 1.4 解码路径族（推理实现）

| 路径 | 机制 | 计算级别 |
|---|---|---|
| round-32 | top-π 分量逐位取整（去重） | 零 GEMM |
| roundflip (v3) | 32 分量模式 + 全分量不确定位翻转（cap 8/comp，~85 候选） | 零码本 GEMM |
| adaptive | π 质量/比率动态分量域 + 翻转 | 零码本 GEMM |
| **scoreflip** | π 质量 99% 域 + 方差序翻转格点 + **精确混合概率重打分按分截断 256**；尖锐位自截断 90-105 候选 | 无排名 GEMM |
| mixture_topk | **界剪枝精确排名**（max_k A ≤ log p ≤ max_k A+log K，幸存者才做精确 logsumexp；与全排名逐位一致） | [V,19]@[19,K] GEMM |
| 级联重排 | 任意提议 + 原嵌入表候选行精确重打分（32-256 行 gather） | ~131KB-1MB/步 |

CPU-offload 拆分：GPU 跑粗筛（头前向 / CFHead 索引图），KB 级结果过 PCIe，CPU（numba JIT，packed uint32 码本 1.3MB）跑提议 + 从 RAM 驻留词表取行 + GEMV——**词表矩阵完全不占显存**。

## 2. 评估口径（定义）

1. **教师强制保真**（held-out 四源 dev，各 16,384 位配对测量）：recall/containment@k（原生 argmax ∈ 候选集）、级联/细阶段后 top-1 一致率（含 bf16 并列翻转）、候选质量占比、**全词表归一化 PPL 比（MoB vs 原生）**、原生 top-1 概率四分位分桶。
2. **Benchmark 任务**：MGSM(en/zh)/MMLU/GSM8K/ARC-C，贪心（对方 Table VI 口径），accuracy + token 一致率 + containment，**配对翻转分析**（赢/输/双对/双错——下游 accuracy 差的去噪）。
3. **长文自由生成**：6 prompts×8192 步×4 臂，逐步原生全头 shadow 打分，复读/新词分桶（防轨迹口径灌水）。
4. **预算扫描**：k∈{8…8192} 的 containment/recall 曲线 + 预算对齐（32 vs 256）行。
5. **采样形状**：mass(C)（兜住的原生总质量=采样命中率=TV 距离）、覆盖曲线、max 单点漏失（worst casualty）、分桶漏失、nucleus(0.9) 完整包含率、T=1.0/0.7。
6. **延迟**：朴素实现存档 + 算法复杂度核算（python/实现开销按裁定豁免）+ offload 拆分实测。

## 3. 结果

### 3.1 教师强制保真（92% 进度头）

| 源 | 密度 PPL 比 | mixture r@32 | round r@32 | 级联一致（32） | CFHead-T4 cont@32 / @256 |
|---|---|---|---|---|---|
| EN | 2.17× | 0.9667 | 0.9482 | 0.9516 | 0.9415 / 0.9916 |
| ZH | 3.66× | 0.9420 | 0.9159 | 0.9304 | 0.9114 / 0.9879 |
| Code | 2.30× | 0.9503 | 0.9202 | 0.9450 | 0.9623 / 0.9921 |
| Instruct | 1.47× | 0.9845 | 0.9712 | 0.9753 | 0.9907 / 0.9989 |

预算对齐 k=32：**MoB 混合在 EN/ZH 领先 CFHead 最好配置 +2.5/+3.1pt，Code/Instruct 被反超 ~1pt**；弥散位（q1 桶）MoB 全源领先 +5-6pt。候选集 miss 时 99.7-100% 兜住原生 top-8（优雅降级）。

### 3.2 预算解锁 256：解码策略与天花板

scoreflip 全胜便宜提议且尖锐源自截断；varflip（分量级方差序）证伪；**但精确上限 mixture_topk@256 仍差 CFHead-T4@256 一至两点（四源一致）——解码侧挖尽，差距在密度**。采样形状：CFHead-T4 全指标领先（mass(C) +2~8pt、max 单点漏失小 3-5×、nucleus 包含 +10~15pt），精确上限同样全输，T=0.7 不救。

### 3.3 Benchmark（贪心，their Table VI 口径）

CFHead-Q（k=256）≈ 原生（±噪声，其两个"胜"经配对翻转证实为轨迹彩票）；**CFHead-M（论文 benchmark 头条配置 T8/C4/k8192）全线崩塌 -6.3~-25.8pt**（原生轨迹 containment 仅 81-85%，候选饿死）；MoB 级联 -1.6~-5.2pt（1/3 预算）。

### 3.4 长文自由生成（their Table II 口径）

四臂全部 ≥99.87%（hit）。机制实锤：**原生贪心轨迹 90.3% 的步数处于复读循环**（该桶内包含率平凡成立）——论文的"100% hit"没有造假，但测在自条件化高置信区；同配置换 held-out 人类文本即降至 89-99%。CFHead 全文无教师强制 held-out 指标，本报告 §2/§3.1 即该缺失指标的补测。

### 3.5 延迟与等带宽核算

带宽平价成立（T4 INT6 理论 95.3 MiB ≈ 头 fp16 95.0 MiB；uint8/fp32 运行时口径双方各涨）。Offload 拆分单头分解：**mob 1.80+0.16+2.44 = 4.40 ms/步 vs CFHead 3.61+0.16+0.63 = 4.40 ms/步（镜像成本分布：其贵在 GPU 索引游走，我贵在 CPU 提议）**；e2e（2B/4060）：native-gpu 10.43 / cfhead-offload 10.27 / mob-offload 9.96 / naive-cpu 5.06 tok/s。scoreflip 在真实解码分布上 CPU 均值仅 2.4ms（自截断；对抗分布才有 42ms）。

## 4. 与 CFHead 的对比判定

1. **轨迹/benchmark 口径**：人人 ~100% / ±噪声（CFHead-Q 确实稳）；M 档在 Qwen 不可用。
2. **教师强制、预算对齐**：k=32 我方 EN/ZH +2.5~3.1pt、Code/Instruct -1pt；k=256 被反超 1~2pt（连精确上限）。
3. **采样形状**：全输（mass、worst-casualty、nucleus）。
4. **延迟**：offload 拆分同量级平手；纯 GPU 粗筛其实现内核成熟（vendored Triton 已移植，7.8×）。
5. **训练成本**：CFHead 免训练（离线 k-means 分钟级）；MoB 需蒸馏（~35 GPU·时/2B）——对方结构性优势。
6. **我方独占**：全词表归一化密度（采样/校准/截断语义的前提）、无原头可用性（替换模式零 1017MiB 常驻）、可原生预训练（Rung-3 判据通过）、码接口（空余顶点/约束解码）。

**总判定**：在"加速现有模型推理"的赛道上，同预算三轴（延迟、top-1、集合质量）均败于免训练的 CFHead-T4，幅度不大但构成明确的技术天花板；精度基线已高、追赶的边际效益不乐观。推理加速臂阶段性停止（docs/16 §8），训练侧反超（密度质量）保留为开放问题，主线转向端到端蒸馏——无原头独立生成是对方无法进入、我方密度独占的场景。

## 5. 资产与复现

- 产物：`runs/qwen35_head_latest.pt`（92% 头）；`runs/cfhead_{teacher_forced,benchmarks,longform}/`、`runs/proposal256/`、`runs/sampling_shape/`、`runs/bench_offload.json`（全部评测数据）；`runs/cfhead_cache_qwen35/`（三档缓存）。
- 代码：`tools/build_cfhead_cache.py`、`eval/cfhead_{rt,kernels,eval,benchmarks,longform}.py`、`eval/{proposal_256,sampling_shape}_eval.py`、`mob/cpu_proposals.py`、`tools/bench_offload.py`。
- 复现成本（本地 4060 8GB 实测）：缓存 ~22s/档；教师强制四源 ~30min；benchmarks ~6h（断点续跑）；长文 ~7h；采样形状 ~25min；offload ~2min。

## 6. 诚实边界

92% 训练进度（余弦尾未跑完，MoB 全部数字为下界）；stride 抽样 ±0.6pt（臂间配对有效）；benchmark 样本 128-250（±4pt 内视为噪声，以配对翻转为准）；延迟数字为朴素实现存档（双方实现开销按裁定豁免）；MGSM 数据事件已修复在案（answer 列 NaN → answer_number，EN-CoT 示范）；未测口径：采样实际生成分布对照（P2）、T4 档 benchmark 臂（教师强制已含 T4）。

---
*阶段收尾。下一阶段：端到端蒸馏。*
