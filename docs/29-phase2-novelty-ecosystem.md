# 29. Phase 2（端到端蒸馏/原生训练替代解嵌层）创新性、撞车风险与学术生态位全面调研

> 归档说明：本报告由独立调研会话于 2026-09-21 完成，针对 **Phase 2 = 端到端训练替代传统解嵌层**（联合微调 2a / 从零预训练 2b / 出生即蒸馏 2c）做全面新颖性与撞车风险评估——Phase 1 的调研（docs/18、docs/23、docs/25）主要覆盖推理加速臂，本报告**不重复其结论，只在其上叠加 Phase 2 特有的攻击面**。方法：5 路并行检索子代理（端到端非 softmax 输出层 / 密度头与连续潜变量 / 跨输出空间蒸馏 / 理论表达力 / 2025-2026 实时扫雷），所有新增文献经 arXiv/OpenReview/Semantic Scholar/OpenAlex 页面核实；标注 [子代理核实] 的条目由子代理本轮抓取原文页核实、主代理抽查复核；最高威胁条目（TDLM、CoBit、CALM、EBT、梯度瓶颈文、撤稿 ECOC 投稿）由主代理逐条亲自复核（arXiv 摘要页 / OpenReview API）。
>
> **一句话结论：Phase 2 的核心组合——"端到端训练的 LM，唯一输出层 = 语义极小二进制码上的归一化因子化伯努利混合密度，训练与推理全程无 V 维 softmax"——在 2024-2026 全部可查文献中为空集，五要素交集无人占据；蒸馏进"不同分布族"的输出头（含出生即蒸馏）同样为空。但赛道正在急剧升温：BitLM 已到 8B 从零预训练、TDLM 用语义树码做端到端扩散 LM 并主打训练显存减半、一篇 ICLR 2026 撤稿论文做过"ECOC 码替换 softmax 端到端"且审稿人亲口建议"用语义码"——留给我们的时间窗是真实存在的，估计 6-12 个月。Phase 2 的论文必须抢在窗口内，并把叙事锚定在三件别人给不了的东西上：归一化单步密度、受控双生子折扣测量、码即接口。**

---

## 1. Phase 2 待检验主张（判决对象）

| 编号 | 主张 | 对应实验线 |
|---|---|---|
| **A** | 首个端到端训练（联合微调/从零预训练）的 LM，其输出层是 ⌈log2 V⌉ 位**语义**二进制码上的**归一化混合密度**（K 分量因子化伯努利），完全替代 softmax——训练与推理均无 V 维 softmax | P4 联合微调、rung-3 双生子、0.8B 计划 |
| **B** | 首个把 softmax 教师蒸馏进**不同分布族**学生输出头的工作（跨分布族 KD；含 2c 出生即蒸馏：学生从不拥有 softmax 头） | 2c 出生即蒸馏管线（docs/18 §9.3 条件 2） |
| **C** | 训练侧收益：消除 V 维 softmax 带来的训练计算/参数/带宽节省（不限于 logits 显存） | rung-3 twin 每步 1.6× 观察（docs/16 §1.4） |
| **D** | 首个对"输出层**结构**"做同算力双生子/scaling 折扣测量的工作（折扣可约性轨迹） | rung-3 twin（1.893× @0.49B token，单调收拢）+ Q1 规模阶梯 |

## 2. 撞车风险总表（威胁分级）

| 级别 | 工作 | 撞什么 | 存活要素 |
|---|---|---|---|
| **T1（最高）** | **TDLM**，arXiv:2604.03537，2026-04，Wu et al. | 语义树码（嵌入聚类建树）+ 端到端从零训练 + "去掉全词表预测层，训练峰值显存减半、PPL 持平" | 扩散范式（非 AR）；树节点内仍是 512 元 softmax；无归一化全词表密度；无码接口；无蒸馏 |
| **T2（高）** | **BitLM**，arXiv:2605.11577（已编目，本轮补齐实验细节） | "二进制码替换 softmax + 端到端预训练"已到 **8B / 350B token** | 扩散头非单步密度；ID 序码无语义；无归一化密度；**无同算力 softmax 双生子**（质量证据弱：XSum SFT 低于 pointer-generator） |
| **T3（高·形状最贴）** | **撤稿 ECOC 投稿**"Encoding the LLM Vocabulary Bottleneck"，ICLR 2026 Withdrawn，OpenReview NNkDZsBu82 [子代理核实 + 主代理复核 venue] | ECOC 二进制码整体替换 softmax 输出层，⌈log2 V⌉ 最小码，**微调（OPT-1.3B）+ 从零预训练（GPT-2-15M）都做了** | 单伯努利无混合；随机码无语义；**无归一化密度（审稿人明确指出 ECOC 解码不构成合法分布、全文报不出 PPL）**；质量损失大（83-90%）；规模小；**已撤稿** |
| **T4（中）** | **CoBit**，arXiv:2605.07013，2026-05，Cambridge | 极小位码（⌈log2 V⌉）+ 端到端从零 + 去 O(V) 瓶颈 | 连续扩散多步去噪；ID 比特无语义；无自身归一化密度（GenPPL 靠外部 AR 打分） |
| **T5（中·叙事邻居）** | **CALM** arXiv:2510.27688（腾讯微信）、**EBT** arXiv:2507.02092、**LCM** arXiv:2412.08821、**LatentLM** arXiv:2412.08635 | "softmax-free 原生预训练 LM"大叙事正在被连续表示/能量模型路线占领 | 全部 **likelihood-free**（报不出 PPL）；连续向量非离散码；CALM/LCM 牺牲逐 token 粒度；无码接口 |
| **T6（中·公式最近）** | **TarFlowLM**，arXiv:2507.00425，NeurIPS 2025 | 语言 + 混合密度（MoG⇔flow 等价定理）+ 端到端，四项占三 | 连续 latent d=16 非离散码；小规模；无逐 token 归一化质量折扣报告 |
| **T7（Claim C 专杀）** | **CCE** arXiv:2411.09009（ICLR 2025，Apple）、**Liger/FLCE** arXiv:2410.10989（MLSys 2025，LinkedIn） | 训练期 logits 显存瓶颈——kernel 级消灭（Gemma-2-2B loss 显存 24GB→1MB），零架构改动零质量风险，已进生产栈 | 只省**显存物化**，h×V GEMM、V 维 LSE、V×d 权重+优化器状态仍全付；MoB 的幸存主张 = 头的**计算与参数**本身消失 |
| **T8（预警）** | **"Lost in Backpropagation: The LM Head is a Gradient Bottleneck"**，arXiv:2603.10145，2026-03，Godey/Artzi（Cornell） | 纯分析：lm_head 秩-D 线性层压掉 95-99% 梯度范数，受控预训练证明梯度瓶颈使平凡模式不可学，**明确呼吁新 LM head 设计** | 无方案（只是分析文）——但这是学术组正在酝酿新头的信号；同时是 MoB 的理论弹药（softmax 头自己是梯度瓶颈） |

**判定：T1-T8 全部为 PARTIAL，无 FULL。Phase 2 的五要素组合（端到端 × 归一化混合密度 × 极小语义码 × 单步 AR × 码接口）无一篇触及。**

## 3. 逐条威胁详评

### T1 TDLM（arXiv:2604.03537）——沉睡者，最危险

[主代理复核 arXiv 摘要页，v1 2026-04-04，Zihao Wu et al.] 机制：离散扩散 LM 的中间潜状态 = token 在**预构建语义词表树**（嵌入递归 k-means）上的祖先节点；树结构因子化把分类维度指数级压缩，预测头"可忽略"，省下的参数加深注意力块；同参数预算下**峰值显存减半、PPL 持平 SOTA 离散扩散 LM**（131B token 从零训练）。

- **撞到的**：①语义树码构造原则（嵌入聚类）与我们码本同族；②端到端从零训练替代全词表预测层；③**Claim C 的叙事框架**——"全词表预测层主导训练峰值显存，结构化头减半"，与我们训练侧动机同构。
- **幸存的**：①扩散范式——逐位置仍是 K=512 元 softmax，需多步去噪；②**无归一化全词表密度**（树路径概率是截断分布，与 HiDRA 同病）；③无单步 AR、无蒸馏、无码接口（无空余顶点概念）。
- **应对**：必引 + 四点划界（AR vs 扩散 / 极小二进制码 vs 节点 softmax / 归一化密度 vs 树路径截断 / 码接口）；其存在反而**强化领域级动机**（"词表结构正在复兴"——2026 年三个独立组（TDLM/HiDRA/我们）同时回到语义树，方向被同行验证）。

### T2 BitLM（arXiv:2605.11577）——概念正上方邻居，体量超预期

[子代理精读 HTML 全文；主代理复核] 新事实：Qwen-3 架构骨干 + BitDance 扩散头，**0.6B/1.7B/4B/8B 四档全部从零预训练于 FineWeb 350B token**，块因果 m=4 并行去噪；XSum SFT ROUGE-1 25.91，**自认低于 pointer-generator 基线**——"二进制输出头在规模上的诚实折扣证据"。

- **对 Phase 2 的意义**：它占据了"二进制码输出接口可预训练到 8B"这个事实位。幸存的：归一化密度、语义码、单步 AR、**受控双生子**（BitLM 全文只有自 scaling 曲线，无同算力 softmax 对照——我们的 1.893× twin 是它没有的测量）。
- **风险升级路径**：BitLM 结论原文邀请 "learned binary codes" 与 "hybrid softmax-binary architectures"——若其下一版把扩散头换成单步混合头，就是正面撞车。**语义码布局是目前唯一未被别人主张的缓冲带**（TDLM 的树在扩散框架内、ECOC 撤稿文用随机码、BitLM/CoBit 用 ID 码）。

### T3 撤稿 ECOC 投稿（ICLR 2026 Withdrawn，NNkDZsBu82）——最贴形状的先例 + 最好的对照素材

[子代理核实 OpenReview 全文+三位审稿人意见；主代理复核 venue 状态] Agarwal/Huang/Ariyawansa/Pujol 等 9 人。机制：ECOC 多二分类框架整体替换 LM 输出层，L 位码字最小取 ⌈log2 V⌉（Minimal）/ 加冗余位（MinRandom）/ 每比特独立投影（MTL-ECOC）；解码 = 最近码字。**微调（OPT-1.3B）与从零预训练（GPT-2-15M）双设置**；50% 输出层参数时保留 ~90%（预训练）/~83%（微调）基线性能。**已撤稿**。

- **战略价值大于威胁值**：这是 Oda 2017 谱系在 LLM 时代的第一次正式复活尝试，撞的正是"二进制码替换 softmax 端到端"。它的失败模式被审稿人公开记录在案——**"无归一化密度、随机码、质量差"**——恰好逐条对应 MoB 的三个修复（混合分量 / 语义树码 / 教师加权 NLL）。论文写法上，这是比 Oda 更近的"反向印证"：评审意见里已有"structured/semantically informed embeddings might further improve"（审稿人亲口建议语义码）。
- **预警**：撤稿非拒稿，作者群补完意见后极可能转投 ACL/NeurIPS 2026。**若他们加上混合分量或归一化修正，将从 PARTIAL 升级为 FULL 撞车**。行动：投稿前定期扫该作者群（Agarwal/Birdal/Zor）新稿（见 §8）。

### T4 CoBit（arXiv:2605.07013）

[主代理复核摘要页，v1 2026-05-07，v2 改名 CoBit] 文本→每 token ⌈log2 V⌉ 位模拟比特流→VE 高斯扩散→匹配滤波残差参数化逐比特头（O(log V) 输出，LM1B 输出张量缩 2035×）→熵门控 Langevin 采样。130M@LM1B GenPPL 59.76 达 AR 参考线；462M OWT GenPPL 19.5。端到端从零。幸存差异同 BitLM（扩散多步、ID 比特无语义、GenPPL 靠外部 AR 打分=自身无归一化词表密度、128-256 NFE）。**判读：与 BitLM 同属"极小位码+端到端+去 V-softmax"升温赛道，赛道内机制已开始分化（块因果扩散 vs 全扩散）——赛道拥挤度本身在涨。**

### T5 CALM / EBT / LCM / LatentLM——"softmax-free 原生预训练"叙事邻居群

- **CALM**（arXiv:2510.27688，腾讯微信+清华，主代理复核）：K token 压成一个连续向量（AE 重构 99.9%），next-vector 预测，**彻底去掉 V-softmax**，energy score likelihood-free 训练 + BrierLM 评测；371M（K=4）匹配 281M 离散基线，训练 FLOPs -44%。工业级工具链 + 开源 repo。
- **EBT**（arXiv:2507.02092，主代理复核）：Energy-Based Transformer 无监督预训练，预测=能量梯度下降；**预训练 PPL 略差于 Transformer++ 但缩放率快 ~35%**，下游多数反超——"非 softmax 输出头端到端预训练 + 诚实报折扣 + 缩放反超"已被做过（likelihood-free）。
- **LCM**（arXiv:2412.08821，Meta）：句级概念向量 AR，1.3T-2.7T token，MSE/扩散变体头。**LatentLM**（arXiv:2412.08635）：连续模态用扩散头、**文本位仍留 softmax**——反向证明"文本位换头"没被做。
- **共同幸存要素**：全部无逐 token 归一化离散密度（CALM 为此专门发明 BrierLM 替代 PPL）；牺牲逐 token 粒度（CALM/LCM）或需迭代优化（EBT）。**MoB 是唯一"离散 + 归一化 + 单 token 粒度 + 单步"的路线**。叙事风险真实存在：评审可能说"softmax-free 预训练已有 CALM/EBT"——应答：它们是 likelihood-free 范式（放弃概率语义），我们保留完整概率接口（PPL/校准/采样/重排/RLHF 兼容）。

### T6 TarFlowLM（arXiv:2507.00425）

[子代理核实] 语言 + 混合密度（MoG/MoL 耦合变换，1D MoG⇔Mixture-CDF flow 精确等价定理）+ 端到端（Text8/OpenWebText，d=16 latent）——"语言+归一化混合密度+端到端+显式替代 softmax"四项占三项的最强对冲。差异：连续 latent、flow 机器、小规模、无语义码本、无逐 token PPL 折扣报告（报 NELBO 相对扩散 LM 优势）。**必须在 related work 正面处理并逐点区分**（离散有限顶点集 vs 连续 latent；闭式单步密度 vs flow 采样；码接口）。

### T7 CCE / Liger——Claim C 的真正对手

[子代理核实] Apple CCE（ICLR 2025，arXiv:2411.09009）：Flash 式 kernel 不物化 [B,T,V] logits，Gemma-2-2B loss 显存 24GB→1MB、头训练显存 28GB→1GB。LinkedIn Liger/FLCE（MLSys 2025，arXiv:2410.10989）：chunked 融合 linear+CE，e2e 吞吐 +20-43%、显存 -55%，已进 torchtune/HF TRL 生产栈。

- **Claim C 必须换口径**：不能主张"消灭 logits 显存瓶颈"（kernel 已做到且零风险），幸存主张 = **头的计算与参数本身消失**：每步 head FLOPs O(K·(M+1)·宽) vs O(h·V)；V×d 权重矩阵 + 其优化器状态（Adam 下 3×V×d×4B）不存在；kernel 删不掉这些。rung-3 twin 的每步 1.6×（fp32 V100）是实测锚点。论文中 C 的正确表述：**"kernel 路线把 softmax 的显存税降为 0，但 V 线性增长的计算税与参数税仍在；我们删的是税基本身。"**

### T8 梯度瓶颈分析文（arXiv:2603.10145，Cornell）

[主代理复核摘要页 v1 2026-03 / v2 2026-07] 证明 lm_head 秩-D 线性层压掉 95-99% 梯度范数，受控预训练证明梯度瓶颈使平凡模式不可学，明确呼吁新 LM head 设计。**双重意义**：①理论弹药——softmax 头自己是优化瓶颈（softmax bottleneck 从表达力扩展到优化），直接为我们的 Q1 反向假设（"17-bit 瓶颈会不会驱动不了大模型"）提供了对照框架：V 维 softmax 的梯度压缩是已证实缺陷，我们的位瓶颈是否有更优梯度性质是值得抢先回答的问题；②预警——一个 Cornell 组在这条线上，下一步很可能提新头。

## 4. 逐主张判定

| 主张 | 判定 | 最近邻 | 幸存边界 |
|---|---|---|---|
| **A** 端到端归一化混合密度+极小语义码替代 softmax | **成立（空集）** | BitLM（码、端到端、8B，但扩散/ID 码/无密度）；撤稿 ECOC（码、端到端、⌈log2 V⌉，但单伯努利/随机码/无密度/撤稿）；TDLM（语义树、端到端，但扩散/节点 softmax）；TarFlowLM（混合密度、端到端、语言，但连续 latent） | 五要素交集为空；AR 谱系自 Oda 2017 后**七年无人**做码基输出头端到端（引用链核实：Oda 17 引、ECOC-NSP 1 引，2020-2026 无 LM 方向跟随者） |
| **B** 跨分布族蒸馏 / 出生即蒸馏 | **成立（空集）** | SpeLLM 2507.16323（唯一蒸馏进**结构不同**输出头：k 个字符 softmax 头，但仍是类别分布族、硬标签、微调非出生）；ALM 2503.20083 NeurIPS 2025（跨**粒度**蒸馏 subword→byte，但逐位置仍 softmax） | 约 25 篇核验过的跨词表/出生蒸馏工作（ULD/DSKD/MinED/OT/CDM/GOLD/Gemma-2/3/Llama3.2/MiniPLM/Peng/SLS），**学生逐位置输出全部是类别 softmax**；分布族 × 从头训练的 2×2 格子为空 |
| **C** 训练侧收益 | **部分存活，需换口径** | CCE/Liger（kernel 级消灭 logits 显存） | 幸存主张 = 头计算 O(KM·宽) vs O(hV) + V×d 参数/优化器状态消失；实测锚点 twin 1.6×/步 |
| **D** 输出层结构的受控双生子测量 | **成立（空集）** | Tao et al. NeurIPS 2024（arXiv:2407.13623）做词表**大小**的 isoFLOPs 双生子（方法论模板可引用，其 unigram 归一化损失值得采用）；Bochkov 2605.09751 输入侧 3-seed twin（协议先例） | 无人对输出层**结构**做同算力对照；BitLM 有自 scaling 无 twin；我们的 1.893× 是文献中首个此类受控数据点 |

## 5. 理论面评估：~2× 折扣是可约的还是结构的？

[理论子代理核验：Montúfar 谱系、softmax bottleneck 谱系、XMC 谱系全部经原文页核实]

**判决：可约部分主导，且无任何下界封锁自然分布上的收敛。**

1. **最坏情况普适性需要指数级 K**（Montúfar-Ay-Ghazi-Zahedi JMLR 2015 参数计数必要条件；Montúfar-Rauh-Ay NeurIPS 2011 误差预算 (M−1)−log₂(K+1)）——按此 K=32 毫无希望，**但 softmax 双生子在 h=2048 对 rank 131072 同样毫无希望**；两边都是结构化逼近器，正确的问题是自然分布上的拟合，而结构依赖界对我们有利：
2. **Montúfar & Ay 2011（Neural Computation，arXiv:1005.1593）**：覆盖支撑集所需分量数 ∝ 支撑集的**汉明覆盖数**（Hamming-相邻对覆盖），而非 V——**这条定理与我们的语义树码本设计精确咬合**：教师质量集中在 ~10³ token 且语义树把它压在少数汉明簇上。这是论文里"为什么语义码布局重要"的理论锚。
3. **Le Roux & Bengio 2008**：支撑集 S 的分布可用 S+1 隐藏单元任意逼近——逐位置下一 token 分布有效支撑 ~10²-10³（我们自己的 top-1024≈95% 测量），分量需求跟踪支撑大小而非 V。
4. **Chang & McCallum ACL 2022**：softmax **结构性无法**表示多峰下一词分布（与模型/数据规模无关）——我们折扣集中的弥散/多峰位置正是 softmax 自己也失败的位置；Zucchet et al. 2026（arXiv:2608.15448，curse of ambiguity，预印本）：模糊分布对**任何**离散输出网络都更难——折扣的一部分是双方都在付的税，归一化 PPL 比值恰是正确度量。
5. **PLT 理论（Wydmuch et al. NeurIPS 2018，arXiv:1810.11671）**：朴素树/位归约分类分布在一般情况下**不一致**（provably inconsistent）——MoB 靠"混合建模联合码分布而非单一路径"避开此坑；论文可写一句干净的对比。
6. **XMC 教训（SLEEC NeurIPS 2015；Guo et al. NeurIPS 2019）**：全局低秩/嵌入结构在超大标签空间有天花板，局部/混合结构才有效——与"MoB（局部混合）> 线性头（已判死，-27pt）"的内部实验互证。
7. **可识别性注意**：Gordon et al. COLT 2024——K=32 混合的参数识别需要 ≥2K−1=63 个可观测位 > M=17，最坏情况下混合**参数不可识别**；对密度逼近无害，但要防懂行的评审。
8. **我们有意违反经典 ECOC 设计准则**（Dietterich & Bakiri 要求行分离+列去相关；我们的位刻意相关、最小距离 1）——用纠错能力换语义光滑性，**论文要明说这是特性不是疏忽**。

**理论缺口（项目可拥有的贡献）**：①结构化支撑（语义树布局）下混合乘积分布的表达力界——Montúfar-Ay 的覆盖论证从未对码本几何实例化；②共享深度参数化的条件乘积混合（"分布式混合"= MLP(h)→(π, 位 logits) 跨上下文插值）完全无理论；③位瓶颈输出层的梯度/优化理论（我们的 LR 舒适带/日程发现可锚定首个此类分析）；④混合乘积作为类别 softmax 归约的一致性正面定理（PLT 给了反面，正面空白）。

## 6. 蒸馏面评估（Claim B 细节）

- **top-k 监督的合法性有工业级先例**：Gemma-3 技术报告（arXiv:2503.19786）出生即蒸馏用**每 token 256 个采样 logits**（教师概率加权、重归一化 CE，14T token 规模）；Peng et al. ACL 2025（arXiv:2410.16215）系统研究预训练蒸馏设计空间（top-p(0.95)+top-100 截断、forward KL 最优、k=50 足够）；Sparse Logit Sampling（ACL 2025，arXiv:2503.16870）证明**硬截断 top-K 缓存有偏**——但我们的损失是教师概率加权的期望形式（不重归一化），更接近 Peng/Gemma-3 的方案而非被批的硬截断；论文加一句防御即可。
- **散度方向选择有文献背书**：对"覆盖弥散教师的混合头"，forward KL（我们在用）是文献认可的安全选择（MiniLLM 的 reverse KL 是 mode-seeking 会坍缩分量；DistiLLM 的 skew-KL 修梯度爆炸）。
- **投机解码提议器从不缩小输出层**（DistillSpec/AdaSPEC/Medusa/EAGLE 的 draft 全是完整 h×V softmax，已核验）——与 Phase 1 定位一致，不赘述。
- **SpeLLM 必须显式划界**：唯一"蒸馏进结构不同输出头"的已核实工作（字符 softmax 头、top-5 教师、best-match 硬标签、微调含末 5 层 FFN）——划界四点：因式是类别分布族非伯努利混合、硬标签非教师加权分布匹配、微调非出生、无归一化全词表密度（需 AutoCorrect 最近 token 兜底）。

## 7. 学术生态位评估

### 7.1 生态地图（2026-09 现状，六大玩家位）

| 生态位 | 代表 | 输出对象 | 归一化密度 | 端到端 | 逐 token |
|---|---|---|---|---|---|
| 检索/筛选 wrapper | CFHead/FlashHead/CSV | token 候选集 | 无 | 免训练 | ✔ |
| 层级路由器 | HiDRA | 子树 | 截断分布 | 微调 | ✔ |
| 连续表示路线 | CALM/LCM/TarFlowLM/LatentLM | 连续向量/latent | 无（或 flow） | ✔ | ✘（块/句级） |
| 二进制扩散路线 | BitLM/CoBit/BitDance | 位码 | 无 | ✔ | ✘（块/迭代） |
| 能量模型路线 | EBT/Residual-EBM/EDLM | 能量分 | 无 | ✔ | ✘（迭代优化） |
| 语义树扩散 | TDLM/HDLM | 树节点 | 截断 | ✔ | ✘ |
| **MoB（我们）** | — | **超立方体顶点（语义极小码）** | **✔ 全词表归一化** | **✔** | **✔** |

**生态位判定：Phase 2 占据的格子——"离散码 + 归一化单步密度 + 端到端 + 逐 token + 码接口"——是空位，且五家邻居各占了一条护城河但没有任何一家能同时跨两条。** 与 Phase 1 时代（docs/18 的"组合式增量"判读）相比，Phase 2 的位置反而更干净：加速臂的竞品（CFHead 等）在 Phase 2 语境全部退场（它们定义上依赖原头在场），剩下的邻居都是"范式级不同"而非"同赛道更快"。

### 7.2 学术价值（两轴合看后的 Phase 2 特有增量）

1. **首个受控输出层结构双生子**（D）：文献无任何同算力 output-layer-structure twin；Tao et al. 做了词表大小的，Bochkov 做了输入侧的，输出侧空白。1.893×@0.49B token 且单调收拢的轨迹本身就是数据贡献。
2. **首个跨分布族蒸馏**（B）：空格子；且 top-k 监督、forward KL 方向、from-birth 管线全部有工业先例护航（Gemma-3/Llama3.2/Peng），风险低、可信度高。
3. **理论锚升级**：Montúfar-Ay 汉明覆盖定理 + Chang-McCallum 多峰失败 + 梯度瓶颈文（T8）三件弹药，把"为什么是这个结构"从工程叙事升级为理论叙事；理论缺口清单（§5 末）本身就是可发表的副产品。
4. **诚实折扣有同伴框架**：EBT（PPL 略差但缩放快 35%）、BitLM（XSum 低于基线）、CALM（K=1 落后）——"新输出表示先付折扣再追平"已有三个同伴案例，我们的 2.81→1.89 收拢轨迹放进了有同伴的框架而非孤立弱点。

### 7.3 时间窗与抢发风险（排序）

1. **ECOC 撤稿组转投**（最高）：审稿意见公开建议语义码；若补混合分量=FULL 撞车。缓冲：他们目前无密度概念、无码本几何、质量差——但修订成本低。
2. **BitLM/CoBit 家族向单步 AR 收拢**：dLLM 赛道资金热度足（Google/腾讯/字节在场）；BitLM 原文邀请 learned codes + hybrid architectures。
3. **连续表示路线抢占"softmax-free 预训练"叙事**（CALM 有 repo 有工业背书）。
4. **Cornell 梯度瓶颈组提新头**（T8 的下一步）。
5. **窗口估计：6-12 个月。** 行动含义：Phase 2 的 twin 数据 + 出生即蒸馏实验是当前最高优先级（既是论文核心实验也是抢跑护城河）；投稿前再扫一次 §8 监控清单。

## 8. 行动建议

**写作（投稿必做）**：
1. related work 按 §7.1 六流派组织，显式引 T1-T8 全部；撤稿 ECOC 文按 withdrawn preprint 引用并注明状态（它是最好的"反向印证"素材——审稿意见公开记录了我们三个修复各自对应的失败模式）。
2. Claim C 换口径为"删税基"（§3 T7）；与 CCE/Liger 的对比写进 related work 而非被打。
3. 理论段引 Montúfar-Ay（语义码为何有效）+ Chang-McCallum（折扣住在哪里）+ PLT 不一致性（混合为何必要）+ Godey 梯度瓶颈（softmax 头自身的优化缺陷）。
4. BitLM/CoBit/TDLM 引为 concurrent works，一句话划界模板："they replace the softmax with iterative denoising over unordered codes; we replace it with a single-step normalized density over semantic codes."

**实验（按优先级，供用户决策）**：
- P0：出生即蒸馏（docs/18 §9.3 条件 2，Claim B 的实证）——文献空格 + 抢发护城河双重价值。
- P0：twin 扩规模阶梯（docs/16 Q1，Claim D 的 scaling 斜率）——采用 Tao et al. 的 isoFLOPs 协议与 unigram 归一化损失口径。
- P1：K 扫描（32→64→128）回应理论面"分量数=每位置可表示联合模式数"的预测——Montúfar-Ay 框架下的可检验预言。
- P1：语义树 vs 随机码 vs ID 码消融（原队列）——现在升级为对撤稿 ECOC 文（随机码）与 BitLM（ID 码）的差异化实证，一石三鸟。

**监控（投稿前复扫）**：Agarwal/Birdal/Zor 作者群新稿（ECOC 撤稿组）；BitLM/CoBit 更新与引用；CALM repo 更新；Godey/Artzi 后续；"binary code output language model" arXiv 月度扫。

## 附录 A：Phase 2 新增必引清单（本轮核实，不含 docs/18/23/25 已编目条目）

| 文献 | 出处 | 用途 |
|---|---|---|
| TDLM, Wu et al. | arXiv:2604.03537, 2026-04 | T1：语义树+端到端+训练显存，四点划界 |
| CoBit, Batzolis/Girolami/Ambrogioni | arXiv:2605.07013, 2026-05 | T4：极小位码扩散，赛道拥挤度证据 |
| CALM, Shao/Li/Meng/Zhou | arXiv:2510.27688, 2025-10 | T5：连续 next-vector 范式，likelihood-free 划界 |
| EBT, Gladstone et al. | arXiv:2507.02092, 2025-07 | T5：非 softmax 头预训练+缩放率叙事 |
| LCM, Barrault et al. (Meta) | arXiv:2412.08821, 2024-12 | T5：句级概念模型 |
| LatentLM, Sun et al. | arXiv:2412.08635, 2024-12 | "文本位仍留 softmax"引证 |
| MAR/Diffusion Loss, Li et al. | NeurIPS 2024, arXiv:2406.11838 | 扩散头谱系 |
| CCE, Wijmans et al. (Apple) | ICLR 2025, arXiv:2411.09009 | T7：Claim C 口径 |
| Liger/FLCE, Hsu et al. (LinkedIn) | MLSys 2025, arXiv:2410.10989 | T7：同上 |
| Godey et al.（梯度瓶颈） | arXiv:2603.10145, 2026-03 | T8：理论弹药+预警 |
| Headless LMs, Algayres et al. | ICLR 2024, arXiv:2309.08351 | 删头≠换头一句引用 |
| ECOC 撤稿文, Agarwal et al. | ICLR 2026 Withdrawn, OpenReview NNkDZsBu82 | T3：反向印证+监控对象（注明 withdrawn 状态） |
| SpeLLM, Ben-Arzy & Schwartz | arXiv:2507.16323, 2025 | Claim B 最近邻划界 |
| ALM, Minixhofer et al. | NeurIPS 2025, arXiv:2503.20083 | 跨粒度蒸馏最近邻 |
| ULD, Boizard et al. | TMLR 2025, arXiv:2402.12030 | 跨词表蒸馏线 |
| DSKD, Zhang et al. | EMNLP 2024, arXiv:2406.17328 | 同上 |
| FuseLLM/MinED, Wan et al. | ICLR 2024, arXiv:2401.10491 | 同上 |
| Gemma-3 TR | arXiv:2503.19786, 2025 | top-256 出生蒸馏先例 |
| Peng et al.（预训练蒸馏设计空间） | ACL 2025, arXiv:2410.16215 | top-k/forward-KL 合法性 |
| Sparse Logit Sampling, Anshumann et al. | ACL 2025, arXiv:2503.16870 | top-K 偏差理论（防御引） |
| MiniPLM, Gu et al. | ICLR 2025, arXiv:2410.17215 | 出生蒸馏（数据侧） |
| Tao et al.（词表 scaling law） | NeurIPS 2024, arXiv:2407.13623 | Claim D 方法论模板 |
| Montúfar & Ay | Neural Computation 2011, arXiv:1005.1593 | 汉明覆盖表达力定理 |
| Montúfar, Rauh & Ay | NeurIPS 2011, arXiv:1406.3140 | K-vs-误差预算 |
| Montúfar & Morton | SIAM JDM 2015, arXiv:1206.0387 | 混合 vs 混合之乘积分离 |
| Chang & McCallum | ACL 2022 | 多峰位置 softmax 结构性失败 |
| Wydmuch et al.（PLT） | NeurIPS 2018, arXiv:1810.11671 | 位归约不一致性 |
| Mnih & Hinton | NeurIPS 2008 | 嵌入二叉树码最老祖先（必引划界） |
| Le Roux & Bengio | Neural Computation 2008 | 支撑集大小表达力 |
| Allman, Matias & Rhodes | Annals of Statistics 2009, arXiv:0809.5032 | 混合可识别性（若讨论） |
| Zucchet et al.（curse of ambiguity） | arXiv:2608.15448, 2026（预印本，注明状态） | 折扣的头无关成分 |

## 附录 B：检索审计摘要

- **5 路子代理**：①端到端非 softmax 输出层（Semantic Scholar 引用链：Oda 17 引/ECOC-NSP 1 引/TCMBN 8 引/BitLM 1 引/GIVT 84 引，全部逐条过筛）；②密度头与连续潜变量（GIVT 引用列表、vMF 引用列表全扫）；③跨输出空间蒸馏（~25 篇核验）；④理论（Montúfar 谱系/softmax bottleneck 谱系/XMC 谱系原文核验）；⑤2025-2026 实时扫雷（arXiv API 三组精确检索 0 命中、OpenReview 全文检索三组、WebSearch 13+ 查询变体、NeurIPS 2025/ACL-EMNLP 2025  venue 扫查、ICLR 2026 投稿抽查）。
- **主代理复核**：TDLM/CoBit/CALM/EBT/梯度瓶颈文 arXiv 摘要页；ECOC 撤稿文 OpenReview API（venue=ICLR 2026 Withdrawn 核实）。
- **空结果如实记录**："mixture of Bernoullis language model head"、"binary code output layer LM"、"Bernoulli mixture next token" 三组 arXiv 全文检索 0 命中；OpenReview "Bernoulli language model"、"binary codes vocabulary language model head" 无相关命中。
- **已知局限**：Semantic Scholar 匿名限流（429）致部分引用链经 OpenAlex 交叉完成（两源互证一致）；ICLR 2026 投稿无法穷举；撤稿文摘要细节来自子代理的 OpenReview 抓取（主代理复核了 venue/标题/作者，摘要内容以子代理抓取为准）。
