# 26. CFHead vs MoB 对打计划（Qwen3.5-2B 试验台，2026-09-18）

> 背景：docs/25 判定 CFHead 为最强撞车（免训练粗-细两段，greedy 100% 保真，但全文无 teacher-forced held-out 指标；附录 C/D 记录口径疑点）。用户裁定：**借 Qwen3.5-2B 蒸馏结果做交叉验证——两个方法都过两套口径**。本文档是对打方案 + 决策记录。复现侧（Llama-3.2-1B-Instruct）已完成，见 `E:\CFHead\CFHead\repro\REPRO_REPORT.md`。

## 1. 试验台

- 模型：Qwen3.5-2B（instruct，scratch `/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B`），V=248,320，h=2048，tied embeddings（CFHead 的 lm_head = embed_tokens 权重）。
- MoB 头：`runs/p9_qwen35_swiglu/head_latest.pt`（swiglu K=128 m=18，224k/244k 步，3 天时限到期停止；终 panel agree 0.7505 / dev_nll 3.256 / n_eff 76.5 / 0 死）。
- CFHead：**免训练**，按仓库 `algorithm/CFHead.py` 语义在 Qwen tied 权重上建缓存（seed 99、iters 10、逐 tile randperm 初始化、cdist+argmin+均值）。双配置（用户拍板）：
  - **CFHead-Q（质量档）**：T=8, C=64, 默认 k=256；
  - **CFHead-M（存储档）**：T=8, C=4, 默认 k=8192；
  - k 为运行时参数，同缓存上扫 k∈{8,16,32,64,128,256,512,2048,8192} 出预算-保真曲线。
- 数据：四源 held-out dev（`cache/qwen35_dev_{en,zh,code,instruct}_tokens.npy`，各 ~4M tokens）+ mix dev。

## 2. 口径 B：teacher-forced held-out（论文缺失指标，我方主口径）

每源 16,384 位置，所有臂**同批位置配对**。原生参考（argmax / top-8 / top-1 概率 / 熵）只算一次。

| 臂 | 指标 |
|---|---|
| CFHead-Q / CFHead-M | **containment@k**（原生 argmax ∈ 粗筛 top-k）全 k 扫描；细阶段精确 ⇒ 重排后一致率 = containment（并列翻转单独统计） |
| MoB round (32) / roundflip (~36-122) / mixture (精确排名) / adaptive | recall@32、级联重排一致率、hit@top8、候选质量占比 |
| MoB 独有 | **全词表归一化密度**：dev NLL / 标准化 PPL（CFHead 结构上 N/A——候选集外无分数） |
| CFHead 独有 | 免训练、零蒸馏成本（记账项，非指标） |

分桶（附录 C 对策）：按原生 top-1 概率/熵分桶（尖锐 vs 弥散），按语源分解（中/代码长尾 = CFHead 疑似软肋，其 MGSM -3.9pt 暗示 + 我们码覆盖≠训练覆盖的教训互为镜像）。**预算对齐行**：CFHead k=32 vs MoB 32 候选，同预算正面交锋。

## 3. 口径 A：自回归 + benchmark（对方主口径）

**长文**：6 prompts（ShareGPT 2 / LMSYS 2 / LongBench hotpotqa+samsum 2，输入固定 256 token，协议同复现）× **8192 步**（用户拍板，与论文 Table II 同规模）× 臂：native / CFHead-Q(k256) / CFHead-M(k8192) / **MoB 级联（roundflip+原表重排）**。每步用原生全头 shadow 打分：hit（原生 argmax ∈ 候选）+ 逐步一致率 + 复读段分桶（近窗重复检测）+ 轨迹漂移分析。**所有臂对称享受/承受"自选上下文"偏差**。

**独立生成行裁撤（用户裁定 2026-09-18）**：不报 MoB 独立生成。理由记录在案：独立生成的主战场是端到端蒸馏（下一阶段工作）；有级联重排的场景下不用级联是学术故弄玄虚，且不对应任何实际生产环境。

**Benchmark**：MGSM（en+zh，主攻——论文最大掉点 -3.9pt 的疑点行）、MMLU（100 样本 5-shot，协议同复现）、GSM8K（128 样本）、ARC-C（128 样本）。臂：original / CFHead-Q / CFHead-M / MoB 级联。指标：accuracy + token agreement + containment。

## 4. 账本行（Table 1 素材）

| | 原表常驻 | 每步流量 | 候选预算 | 归一化密度 | 无原头可用 |
|---|---|---|---|---|---|
| native | 1017 MiB | 1017 MiB | — | 有 | — |
| CFHead-Q | **必需** | ~61 MiB (uint8 索引图；理论 INT6 45 MiB) + 1 MiB gather | 256 | 无 | 否 |
| CFHead-M | **必需** | ~8 MiB (INT2) + 32 MiB gather | 8192 | 无 | 否 |
| MoB 级联 | 仅重排 32 行 (~131KB) | ~95 MiB 头权重 | 32 | **有** | **是** |

## 5. 作业布局（slurm）

- **J1** `tools/build_cfhead_cache.py` + `slurm_cfhead_cache.sbatch`：双配置缓存构建（256 tiles k-means，GPU 约半小时），产物落 scratch `/gpfs1/scratch/yihe47/qwen35/cfhead_cache/`，格式与仓库兼容（{metadata, codebook, index_map}）。
- **J2** `eval/cfhead_eval.py` + sbatch：口径 B 配对评测（4 源 × 16,384 位置）。
- **J3** `eval/cfhead_benchmarks.py` + sbatch：四 benchmark × 四臂（数据集从复现侧上传 scratch）。
- **J4** `eval/cfhead_longform.py` + sbatch：长文对打。

## 6. 预期假设（诚实记录，结果如实报）

1. 预算对齐 k=32：CFHead containment 预计 ~96-97%（其 Table XII k=32=96.48% 的轨迹口径），MoB recall@32 预计 95%+ —— 同预算交锋可能接近打平，**差异化在密度行与账本行**。
2. CFHead 默认 k=256 在 EN 接近 100%；**ZH/Code 源上的退化幅度是核心观测量**（MGSM 疑点 + 长尾词表）。
3. 自回归口径两臂都接近 100% hit —— 该口径分辨力弱是附录 C 的既有结论，复读分桶数据用于量化这一点。
4. MoB 头 92% 训练进度（余弦尾未跑完）是已知劣势，在报告中标注。

## 7. 风险与边界

- CFHead k-means 语义严格按仓库（iters=10；论文 Fig 5 用 256 簇大码本属另一口径，不混入）。
- Qwen3.5 为 Gated DeltaNet 混合架构——CFHead 只碰输出层，与骨干无关，无适配风险。
- MGSM prompt 模板：论文未发布，按复现侧已下载的原始数据 + 标准 8-shot 模板自建，与论文不可比之处如实标注。
- 计时/加速比非本计划目标（复现已证明本机不可复现其内核加速；带宽账按理论字节数计）。
