# 27. CFHead vs MoB 对打报告（Qwen3.5-2B，2026-09-19）

> 计划与决策：docs/26；撞车分析：docs/25（附录 C/D 的口径疑点由本报告逐一实测收口）。复现侧（Llama-3.2-1B-Instruct，论文数字核对）：`E:\CFHead\CFHead\repro\REPRO_REPORT.md`。
>
> **一句话**：论文的数字没有造假，但都测在自条件化的轨迹区；换到 held-out 教师强制后真序显形——**同预算（32 候选）MoB 在最难的两个源（EN/ZH）领先 CFHead 最好配置 +2.5~+3.1pt，CFHead 的 100% 命中口径不迁移（held-out 98.8-99.9%@k=256），其 benchmark 头条配置（T8/C4/k8192）在 Qwen 上候选饿死、下游崩塌 -6~-26pt**；密度轴与无原头可用性 MoB 结构性独占。

## 1. 设置与配置谱系

试验台：Qwen3.5-2B（instruct，V=248,320，h=2048，tied）。MoB 头：swiglu K=128 m=18（蒸馏 224k/244k 步 = 92%，余弦尾未跑完，劣势已知）。CFHead 免训练缓存三档，全部按仓库语义本地构建（seed 99/iters 10；构建器与仓库同机对照 2.3e-4/99.9999%，运行时逐位一致）：

| 档位 | 配置 | 论文出处 | 每步流量（V=248k） |
|---|---|---|---|
| CFHead-T4 | T=4, C=64, k=256 | **Table II-V 命中表配置**（"tile 4, cluster 64, top-k 256"） | 索引图 95MiB(INT6 理论)/127MiB(uint8 运行) + 1MiB gather |
| CFHead-Q | T=8, C=64, k=256 | **Table VIII-X 运行点**（"tile 8 ... cluster 64 ... top-k 256"） | 45/61MiB + 1MiB |
| CFHead-M | T=8, C=4, k=8192 | **Table VI/VII benchmark+存储头条**（7.84/39.8MiB，逐值命中） | 8MiB(INT2)/15MiB(uint8) + 32MiB gather |
| MoB | swiglu K=128；round-32 / mixture-32 / roundflip(~85) | 本项目 | 头 95MiB + 级联 gather ~0.2MiB |

三口径：B=教师强制 held-out（四源 dev 各 16,384 位配对测量）；A1=benchmark 贪心（their Table VI protocol）；A2=长文自由生成 + 原生 shadow 打分（their Table II protocol，6 prompt×8192 步×4 臂=49,152 步/臂）。

## 2. 口径 B：教师强制 held-out（论文缺失的表）

### 2.1 候选包含率（原生 argmax ∈ 粗筛/提议集）

| 源 | CFHead-T4@32 | CFHead-Q@32 | **MoB mix@32** | MoB round@32 | T4@256 | Q@256 | M@8192 |
|---|---|---|---|---|---|---|---|
| EN | 0.9415 | 0.7856 | **0.9667** | 0.9482 | 0.9916 | 0.9571 | 0.9709 |
| ZH | 0.9114 | 0.7126 | **0.9420** | 0.9159 | 0.9879 | 0.8901 | 0.9322 |
| Code | **0.9623** | 0.8766 | 0.9503 | 0.9202 | 0.9921 | 0.9686 | 0.9241 |
| Instruct | **0.9907** | 0.9311 | 0.9845 | 0.9712 | 0.9989 | 0.9921 | 0.9790 |

**预算对齐（k=32）结论**：MoB 混合在 EN +2.5pt、ZH +3.1pt 领先 CFHead 最好配置（T4）；CFHead-T4 在 Code +1.2pt、Instruct +0.6pt 反超。零 GEMM 的 round-32 在 EN/ZH 也赢 T4 之外的档位。此前 T8 口径的 +5~+23pt 头条被 T4 修正——诚实记录：**tile 粒度对 CFHead 在 held-out 上极其关键（zh@32：T4 91.1% vs T8 71.3%，+20pt），而他们自己的轨迹口径几乎看不到（Table XI：100 vs 99.6）**。

### 2.2 弥散位分桶（原生 top-1 概率四分位，containment@32）

| 源 | T4 q1（最弥散）→q4 | MoB mix q1→q4 |
|---|---|---|
| EN | 0.865 → 0.993 | 0.923 → 0.992 |
| ZH | 0.826 → 0.991 | 0.877 → 0.984 |

VQ 粗排名的软肋正是弥散/多峰位（q1），MoB 的混合密度为那种位置而造——q1 桶 MoB 全源领先（EN +5.8pt / ZH +5.1pt）。

### 2.3 细阶段后一致率与密度

- 细阶段一致率（=containment − 并列翻转 1~2pt，bf16 归约序）：T4@256 = EN 0.970 / ZH 0.971 / Code 0.985 / Instruct 0.988。MoB 级联（32 候选重排）：0.952 / 0.930 / 0.945 / 0.975——**以 1/8 预算在 EN/ZH 贴着 T4@256 打**。
- **全词表归一化密度**：MoB dev NLL 对应 PPL 比 EN 2.17× / ZH 3.66× / Code 2.30× / Instruct 1.47×（ZH 最弱=训练覆盖问题，已知）；CFHead 候选集外无分数，此行结构性 N/A。M 档（C=4）k=8192 时 held-out containment 92.4-97.9%——论文"benchmark 配置在 100% 区"的隐含假设不成立（附录 D 悖论的实测解法）。

## 3. 口径 A1：benchmarks（贪心，Table VI 口径）

| 任务 | original | T4/Q 档（k=256） | M 档（k=8192） | MoB 级联（~86 候选） |
|---|---|---|---|---|
| MGSM-en (250) | 67.2 | 66.4 | 59.2 (-8.0) | 62.0 (-5.2) |
| MGSM-zh (250) | 48.0 | 52.4 | 39.6 (-8.4) | 43.2 (-4.8) |
| MMLU (100) | 60.0 | 60.0 | 44.0 (**-16.0**) | 57.0 (-3.0) |
| GSM8K (128) | 60.9 | 63.3 | 54.7 (-6.3) | 59.4 (-1.6) |
| ARC-C (128) | 79.7 | 79.7 | 53.9 (**-25.8**) | 78.1 (-1.6) |

（Q 档 = T8/C64 实测；T4 只会更好，未重跑——见 §6 遗留。）

- **CFHead-Q 高于原生的两行（zh +4.4 / gsm8k +2.3）是轨迹彩票**：配对翻转 zh 19 赢 8 输、gsm8k 5 赢 2 输、en 7 赢 9 输——双向大翻动 + 小额净差 = 噪声，非系统性强。n=128-250 下下游 accuracy 不是保真度指标。
- **M 档全线崩塌**，且其在原生轨迹上的 containment 只有 81-85%（MMLU/ARC）——候选饿死，正是他们批评 VocabTrim 的死法。论文的 "-0.21%p" 是 Llama-1B 上的幸运抽签，不迁移。
- MoB 级联以 ~1/3 预算付 -1.6~-5.2pt，与教师强制保真度一致（偏差集中在 ZH/CoT 弥散位，同训练覆盖问题）。

## 4. 口径 A2：长文自由生成（their Table II protocol，对称化）

49,152 步/臂，每步原生全头 shadow：

| 臂 | hit | agree | 候选 | 复读桶 hit | 新词桶 hit |
|---|---|---|---|---|---|
| native 轨迹（对 T8/Q 候选） | 1.0000 | — | 256 | 1.0000 | 0.9998 |
| CFHead-Q | 0.9996 | 0.9992 | 256 | 0.9999 | 0.9985 |
| CFHead-M | 0.9998 | 0.9998 | 8192 | 0.9999 | 0.9995 |
| MoB 级联 | 0.9987 | 0.9986 | **85** | 0.9993 | 0.9965 |

**机制解剖（附录 C 的实锤）**：原生贪心长文轨迹 **90.3%** 的步数是复读循环（44,400/49,152），该桶内包含率对一切方法平凡成立；轨迹口径的"100%"由此灌出。轨迹上 ~100% 与 held-out 89-99% 同时为真——**自条件化区（自产上下文=高置信尖锐分布）喂饱了粗排名，人类文本不会**。论文之错不在数据，在于把轨迹口径当成保真度呈递。

## 5. 综合判定（Table 1 素材）

1. **轨迹/benchmark 口径**：人人 ~100% / ±噪声（CFHead-Q 确实稳，该承认）；M 档在 Qwen 不可用。
2. **教师强制 held-out 口径（预算对齐）**：EN/ZH（难且弥散的源）MoB 混合领先 +2.5~+3.1pt；Code/Instruct（尖锐源）CFHead-T4 小胜 ~1pt。**MoB 的赢面集中在弥散/多峰位（q1 桶 +5-6pt）——结构决定**。
3. **预算轴**：MoB 32（round 零 GEMM）~85（roundflip）vs CFHead 256-8192；同预算 MoB 胜率见 §2.1。
4. **密度轴**：MoB 全词表归一化密度（PPL 1.47-3.66×，92% 训练进度）vs CFHead 候选外无密度——独立生成/校准/截断采样的前提。
5. **无原头可用性**：CFHead 恒需原表在场（1017MiB 常驻 + 每步 9-127MiB）；MoB 级联只需 32 行 gather（~131KB），纯替换模式零原表——docs/25 §4 的三支柱叙事不变。
6. **训练成本记账**：CFHead 免训练（离线 k-means 分钟级）；MoB 需蒸馏（~35 GPU·时/2B）。此为其结构性优势，如实入表。

## 6. 诚实边界与遗留

- MoB 头 92% 训练进度（余弦尾未完），所有 MoB 数字是它的下界。
- benchmark 臂用的 Q 档（T8/C64）；T4 档的 benchmark 臂未跑（教师强制已含 T4；预计只会缩小 lottery 噪声）。如需补：单臂 ~2.5h。
- benchmark 样本 128-250，下游 accuracy 差 ±4pt 内均视为噪声（配对翻转法为准）。
- MGSM 数据事件：复现侧 parquet 的 answer 文本列 100% NaN；gold 改用 answer_number，示范用 train_en 完整解答（EN-CoT 条件）——旧跑（'nan' 示范污染）已作废存档 `runs/cfhead_benchmarks_void_nan_demos/`。教训：smoke 必须显式核对 gold 抽取。
- 采样口径（论文 Table III/IV 的 nucleus 91-97%）未测——双方采样保真都未上表，列为 P2。

## 7. 资产与复现

- 数据：`runs/cfhead_teacher_forced/cfhead_eval_{en,zh,code,instruct,all}.json`（三档 CFHead + MoB 全指标 + 分桶）；`runs/cfhead_benchmarks/summary.csv` + 逐样本 JSONL；`runs/cfhead_longform/longform_summary_{arm,all}.json` + 逐步 JSONL。
- 代码：`tools/build_cfhead_cache.py`（忠实移植）、`eval/cfhead_rt.py`（共享运行时）、`eval/cfhead_eval.py`、`eval/cfhead_benchmarks.py`、`eval/cfhead_longform.py`。
- 复现命令（本地 4060 8GB 实测可行）：缓存 ~22s/档；教师强制四源 ~30min；benchmarks ~9h；长文 ~7h。
