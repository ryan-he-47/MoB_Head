# 23. 文献调研查缺补漏与撞车风险彻查（第 2 轮，2026-09-15）

> 归档说明：本报告对 `docs/18-academic-positioning.md` 与 `docs/19-dynamic-vocabulary-deepdive.md`（2026-09-07 归档）做第 2 轮查缺补漏。方法：针对 docs/18/19 **未覆盖或覆盖薄弱**的 6 个方向做定向 live 检索（伯努利混合头直接撞车检查、大记忆/因子化访问层、蒸馏提议器、多 token 预测、扩散式 LM 家族、1-bit 权重范式、低秩/因子化输出层），所有**新增**文献均经 arXiv/NeurIPS/ACL Anthology 页面核实；docs/18/19 已收录的 2026 年文献维持原归档结论（其中标注 [待核实] 的投稿前仍需复核）。
>
> **一句话结论：本轮检索未发现任何工作覆盖本项目核心组合（冻结骨干蒸馏 + 混合因子化伯努利头 + 语义极小超立方体码 + 候选集原头精确重排 + 原生预训练可行性）；撞车风险结论维持 docs/18 §3。但发现 5 个必须补入 related work 的缺口，其中"扩散式语言模型（dLLM）家族"是最大的定位级缺口——2025-2026 年它已成为"逃离自回归 softmax 串行瓶颈"的最热替代范式，评审必问，必须显式划界。**

---

## 1. 缺口清单（docs/18/19 未覆盖，按重要性排序）

### G1【定位级缺口】扩散式语言模型家族（dLLMs：LLaDA / Dream / Mercury / Gemini Diffusion / DiffusionGemma / Seed Diffusion）

**这是什么**：2025 年爆发的"掩码离散扩散"文本生成家族——用整块并行去噪替代自回归逐 token softmax 解码。LLaDA（8B，2.3T token 从头训练，MMLU 逼近 LLaMA3-8B，arXiv:2502.09992）证明可行性；Dream（arXiv:2505.14831，自 AR 初始化）改进训练；**Mercury（Inception Labs，2025-06）商用化，单 H100 ~1109 tok/s，比同质量 AR 快近 10 倍**；Google Gemini Diffusion（I/O 2025，~1479 tok/s）；ByteDance Seed Diffusion（H20 上 2146 tok/s）；2026-06 Google+NVIDIA 开源 DiffusionGemma（26B-A4B，Apache 2.0，1000+ tok/s 但质量仍逊 AR 版）；LLaDA-MoE、LLaDA2.0（100B）持续推 scaling。

**为什么必须引**：docs/18 只收了 BitLM/BitDance 两个"二进制码扩散"工作，**漏掉了整个 dLLM 主干家族**。在 2026 年的评审语境里，"如何不同于扩散式 LM" 将成为与"如何不同于 FlashHead""如何不同于 BitLM"同等必答的问题——dLLM 家族和 MoB 头在最高层竞争同一个目标（**逃离自回归 softmax 的串行/带宽瓶颈**），只是机制完全不同：

| 维度 | dLLM 家族 | 本项目 MoB |
|---|---|---|
| 范式 | 非自回归，整块并行去噪（多步迭代） | 保持单步自回归（与现有生态全兼容） |
| 每步分布 | 掩码位置的多项式概率（仍是 V 维 softmax 头） | **无 V 维 softmax**：17 位因子化伯努利密度 |
| 质量（2026 现状） | 26B 级质量仍落后同族 AR（Google 官方承认） | 蒸馏口径 1.78-2.0× PPL；原生 1.89×（未收敛） |
| 适配现有模型 | 需整体重训（Dream 式 AR→diffusion 转换是研究课题本身） | **冻结骨干 drop-in 蒸馏**（核心卖点不受影响） |
| 输出结构可编程性 | 无码结构 | 语义超立方体码接口（D2） |
| 理论卖点 | 双向上下文、any-order、中途纠错 | 候选集精确重排、截断采样等价（docs/19 §1） |

**结论**：dLLM 是"威胁性邻居"而非竞品——它反而**抬高**了"输出层结构值得重新设计"这一命题的学界热度（Google 亲自把 Gemma 的 AR 头换成扩散头，等于工业界承认 lm_head 是可替换部件）。写作策略：intro 用 1 段引 dLLM 家族 + BitLM，划清"整范式替换 vs 输出表示替换"的界线，把 MoB 定位为"**不改解码范式、只换输出表示**"的保守-可落地路线。dLLM 的多步去噪与 MoB 的单步密度是互补而非互斥（理论上 MoB 密度头也可服务掩码式生成，留作 future work 钩子）。

### G2【概念级缺口】Product-Key Memory（Lample et al., NeurIPS 2019, arXiv:1907.05242）

大记忆层：把 query 切成两半分别在两个子键集做精确 top-k（√|K| 成本），再取笛卡尔积——**用因子化分解对超大键空间做次线性精确访问**。12 层 + PKM 超过 24 层 baseline 且推理快 2 倍。

**为什么必须引**：它是"**因子化码 + 次线性访问大输出空间**"这一思想的最近先例之一，且证明了大容量查表结构与 transformer 可以端到端共同训练（对轴 B 叙事有利）。与本项目差异明确：①PKM 是**中间记忆层**不是输出分布（不建模词表概率）；②访问方式是精确 NN 检索不是密度建模；③键无语义码结构；④不解码成 token、不蒸馏。归入 docs/18 支流二（输出层效率）的"大容量结构"小节即可。

### G3【赛道 B 强化】DistillSpec 及后续（Zhou et al., ICLR 2024, arXiv:2310.08461；AdaSPEC NeurIPS 2025；EDD ACL 2025；域内 draft 蒸馏最佳实践 arXiv:2503.07807）

DistillSpec 用白盒 KD 蒸馏与目标模型对齐的 draft 模型（on-policy 数据 + 散度选择），SD 加速再提 10-45%；AdaSPEC 进一步指出"逐 token KL 对齐"与"SD 接受率"目标错位，改做选择性蒸馏，接受率再 +15%。

**影响**：这组工作**强化** docs/19 §4 赛道 B 的"最近邻"论证——"蒸馏出的小网络当提议器 + 大网络验证"的形状在 SD 家族已成熟。但精确形态差异不变：SD 系提议**串行 token 序列**（draft 本身就是个小 LM/多头，验证是串行 autoregressive verify），MoB 级联提议**逐位置的平面候选集** + gather-32 精确重排（无额外验证前向）。**写作时必须显式引 DistillSpec**（比 S2T 更早、更经典的"learned+distilled proposer"先例），否则评审会拿它当"已知做法"。

### G4【支流一补强】Multi-Token Prediction（Gloeckle et al., ICML 2024, arXiv:2404.19737；DeepSeek-V3 采用；CP 分解变体 arXiv:2410.17765）

n 个独立输出头共享 trunk 预测未来 n token，提升样本效率且推理最多 3× 加速。docs/18 支流一只有 Medusa/EAGLE 系，漏了 MTP 这条"训练期改造"主线。与项目关系：正交可叠加（MoB 替换单步头，MTP 预测未来位），但其"n 个 h×V 头"恰好**反证 h×V 投影的带宽问题会随多头放大**（docs/18 §4.3 Medusa 论点同源）。CP 分解变体（rank-r 混合因子化联合分布）是"因子化联合分布"的相邻思想，可作一句引用。

### G5【名词防线】BitNet b1.58（Ma et al., arXiv:2402.17764）与 1-bit 权重范式

权重三值化 {-1,0,1}，3B 起匹配 FP16（PPL 与下游），原生 1-bit 训练定义了新 scaling law；bitnet.cpp 使 CPU 原生推理可用。**与本项目正交（权重二进制 ≠ 词表码二进制）**，但：①"Bit" 名词在 2026 年已被 1-bit 权重占坑，论文命名/摘要需主动划清；②更重要的——BitNet 是"**原生训练进受限表示优于训练后压缩**"这一论点的最强先例，可直接为轴 B（MoB 原生预训练 vs 蒸馏挂载）的方法论立场背书；③Mercury 的架构也叫 BitNet（Inception Labs），三处 "bit" 撞名，intro 一句话理清三者（权重 1-bit / 扩散 dLLM / 词表二进制码）。

### G6【已排查，无新威胁】低秩/因子化 lm_head 与词表效率线

定向核查低秩输出层、SVD 因子化 lm_head、词表/tokenizer 效率（GroupReduce NeurIPS 2019、ReTok arXiv:2410.04335、tokenizer 跨尺度评估 arXiv:2506.03101）：均为**训练期或压缩期**技术，推理仍 O(V) 或需要全量 logits，与 MoB 的"解码时消除 h×V"不在同一攻击面；VQ-Logits（已收录）仍是该谱系唯一推理期竞品。**结论：无新增威胁，docs/18 支流二覆盖充分，补 GroupReduce（NeurIPS 2019，频率分组块级低秩压缩 softmax 层）一句引用即可。**

---

## 2. 撞车风险彻查结论（更新版）

| 检查项 | 结果 |
|---|---|
| "混合因子化伯努利头 + LLM 词表码"直接撞车 | **未发现**。live 检索"Bernoulli mixture head language model binary code output layer"仅返回通用深度学习教材、MDN 其他域应用与图生成模型中的 mixture-of-Bernoulli 边缘用例（arXiv:2303.03293，图边预测，非词表、非 LLM）——不构成本领域先例，反而再次确认 TCMBN/Oda 仍是唯一同构/最近邻 |
| 输出头/检索式赛道 | 维持 docs/18 §2.5：FlashHead/HNSW/CSV-Decode 仍为最强应用竞品，本项目不打"纯加速"叙事 |
| 二进制码 LM 赛道 | BitLM 维持概念正上方邻居；**新增 dLLM 家族为同目标不同范式的最热邻居（G1）** |
| 提议-验证/候选-重排赛道 | 维持 docs/19 §4 结论，**最近邻名单加入 DistillSpec 家族（G3）**——"蒸馏提议器"概念在 SD 家族已存在，本项目差异在"逐位置候选集 + 同模型原头 gather 重排 + 码结构" |
| 原生预训练/结构可行性 | 维持：无 softmax 替代论文做过从零结构检验；G2（PKM 与 transformer 共同训练）与 G5（BitNet 原生 1-bit scaling law）是** supportive 先例** |
| 码即接口（D2） | 维持：本轮无任何新工作触及语义码接口维度 |

**综合判定：创新点 I1-I5（docs/18 §5.1）全部维持有效；"组合式增量"的诚实定位不变，但组合的对手盘更新为——评审现在会同时拿 FlashHead（加速）、BitLM+dLLM 家族（范式）、DistillSpec（蒸馏提议器）、Oda/ECOC（位码）四个方向夹击，intro 的 related work 必须四线全防。**

---

## 3. 新增必引文献清单（本轮核实）

| 文献 | 出处 | 归入 |
|---|---|---|
| LLaDA (Large Language Diffusion Models) | arXiv:2502.09992, 2025 | dLLM 家族代表（G1） |
| Dream 7B | arXiv:2505.14831, 2025 | dLLM 家族代表（G1） |
| Mercury (Inception Labs) | 2025-06 技术报告 | dLLM 商用速度锚（G1） |
| DiffusionGemma | 2026-06（Google+NVIDIA 开源） | dLLM 工业界承认 lm_head 可替换（G1） |
| Product-Key Memory | arXiv:1907.05242, NeurIPS 2019 | 支流二·大容量因子化访问（G2） |
| DistillSpec | arXiv:2310.08461, ICLR 2024 | 赛道 B 最近邻（G3） |
| AdaSPEC | NeurIPS 2025 | 赛道 B 最近邻（G3） |
| MTP (Better & Faster LLMs via Multi-Token Prediction) | arXiv:2404.19737, ICML 2024 | 支流一·训练期改造主线（G4） |
| MTP-CP 分解变体 | arXiv:2410.17765 | 因子化联合分布相邻思想（G4） |
| BitNet b1.58 | arXiv:2402.17764 | 名词防线 + 原生受限表示先例（G5） |
| GroupReduce | NeurIPS 2019 | 低秩 softmax 压缩谱系一句引用（G6） |

**两篇综述（教授任务）的遴选与分析见 `docs/24-advisor-meeting-pack.md`。**
