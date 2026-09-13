# Qwen3.5 小模型系列 + 多语言/代码数据集选材（调研报告）

> 工单：`docs/21-qwen35-recon-workorder.md`
> 调研日期：**2026-09-14** ｜ 执行方式：5 个并行调研代理（型号规格 / 预训练配比 / 中文语料 / 代码语料 / 英文语料），逐条核验原始 `config.json`、HF/ModelScope API 与节目录。
> 说明：每条事实后附来源 URL。**未核实的字段明确写"未查到"或"未核实"**；⚠️ 标注为"估计"的数字是推断值，**不是** Qwen 官方披露。

---

## 摘要（TL;DR）

1. **"Qwen3.5" 真实存在。** Qwen3.5 系列 2026-02-16 首发，**Qwen3.5-0.8B / 2B 于 2026-03-02 发布**，且 ≤2B 恰好就是这两个规格（还有各自的 `-Base`）。→ [QwenLM/Qwen3.8 News](https://github.com/QwenLM/Qwen3.8)、[官方 collection](https://huggingface.co/collections/Qwen/qwen35)
2. **规格：两者 V=248,320（比 Llama-3.2-1B 的 128,256 大 1.94×，是本方法最优场景），hidden 1024/2048，24 层，tied embeddings=true，Apache-2.0，262K 上下文，ModelScope 同日可得。** 码长 **M = 18**（2^18 = 262,144，仅 13,824 个未用顶点）。
   ⚠️ 但有架构代价：**多模态包装（`Qwen3_5ForConditionalGeneration`，含 vision encoder）+ 混合 Gated DeltaNet/Attention + MTP**，`config.transformers_version` ≥ 4.57.0.dev0 —— 上机前必须验证环境。
3. **选型建议：先 2B 级 `Qwen3.5-2B`，后 0.8B 级 `Qwen3.5-0.8B`（两者共用同一 tokenizer/V ⇒ 一份码本同时服务两个规模）；失稳则回退 `Qwen3-1.7B` / `Qwen3-0.6B`**（成熟 dense 架构，但 V 只有 151,936 且已被证实是蒸馏训练）。
4. **Qwen 预训练数据：无任何语言/代码百分比公开。** 官方硬事实：Qwen3 = **36T tokens / 119 语言**（S1 >30T + S2 ~5T 推理强化 + S3 数千亿长文）；Qwen3.5 = **201 语言 + "trillions of multimodal tokens"**（文本总量未公布）；代码锚点只有代码专模型（Qwen3-Coder **7.5T tokens / 70% code**）。→ Q2 给出**带标签的估计区间**。
5. **2B-token 数据配比建议：英文 55% / 中文 30% / 代码 15%**（估计值，理由见 Q3.4）。首选源：**EN = FineWeb-Edu `sample-100BT`**；**ZH = CCI3-HQ（主）+ FineWeb-2 `cmn_Hani`（辅）**；**Code = The Stack v3 `stack-v3-train` 的 permissive 子集**（2026 新发布，**不需要 Software Heritage**，有 ModelScope 镜像）。

---

## Q1 目标模型规格核实

### Q1.1 "Qwen3.5" 是否存在？

**存在。** 时间线（官方仓库 News）：2026-02-16 首发 `Qwen3.5-397B-A17B`；02-24 发 122B-A10B / 35B-A3B / 27B；**03-02 发 9B / 4B / 2B / 0.8B**。此后还有 **Qwen3.6（2026-04）与 Qwen3.8（2026-08）**，但这两代**没有 ≤2B 开放模型**（最小 27B）。
→ [github.com/QwenLM/Qwen3.8](https://github.com/QwenLM/Qwen3.8)、[huggingface.co/Qwen](https://huggingface.co/Qwen)、[Qwen3.5 collection](https://huggingface.co/collections/Qwen/qwen35)

因此用户说的"Qwen3.5 的 0.8B 和 2B"**完全正确**，不是记错。

### Q1.2 型号事实表（≤2B，截至 2026-09-14）

| 型号 | HF / ModelScope id | 发布 | 参数 | **V (vocab_size)** | hidden | 层数 | tied emb. | 上下文 | 许可 | 是否蒸馏训练 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-0.8B** | `Qwen/Qwen3.5-0.8B`（同 id） | 2026-03-02 | 0.8B LM（含视觉 0.9B） | **248,320** | 1024 | 24 | **true** | 262,144 | Apache-2.0 | 未查到 | 多模态；`-Base` 版同名后缀 |
| **Qwen3.5-2B** | `Qwen/Qwen3.5-2B`（同 id） | 2026-03-02 | 2B | **248,320** | 2048 | 24 | **true** | 262,144（Base 可扩至 1,010,000） | Apache-2.0 | 未查到 | 多模态；`-Base` 版同名后缀 |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B`（同 id） | 2025-04-29 | 1.7B（非嵌入 1.4B） | **151,936** | 2048 | 28 | **true** | 32,768 | Apache-2.0 | **是**（strong-to-weak 蒸馏，后训练阶段） | 纯文本 dense |
| Qwen3-0.6B | `Qwen/Qwen3-0.6B`（同 id） | 2025-04-29 | 0.6B（非嵌入 0.44B） | **151,936** | 1024 | 28 | **true** | 32,768 | Apache-2.0 | **是**（同上） | 纯文本 dense |
| Qwen2.5-1.5B (base) | `Qwen/Qwen2.5-1.5B`（同 id） | 2024-09-19 | 1.54B（非嵌入 1.31B） | **151,936** | 1536 | 28 | **true** | 32,768 | Apache-2.0 | 无官方蒸馏记载 | 纯文本 dense |
| Qwen2.5-0.5B (base) | `Qwen/Qwen2.5-0.5B`（同 id） | 2024-09-19 | 0.49B（非嵌入 0.36B） | **151,936** | 896 | 24 | **true** | 32,768 | Apache-2.0 | 无官方蒸馏记载 | 纯文本 dense |

**原始 config 摘录（逐字核验）**
- Qwen3.5-0.8B：`"hidden_size": 1024, "num_hidden_layers": 24, "intermediate_size": 3584, "tie_word_embeddings": true, "vocab_size": 248320, "max_position_embeddings": 262144` → https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json（Base 同：https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/raw/main/config.json）
- Qwen3.5-2B：`"hidden_size": 2048, "num_hidden_layers": 24, "intermediate_size": 6144, "tie_word_embeddings": true, "vocab_size": 248320` → https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json（Base：https://huggingface.co/Qwen/Qwen3.5-2B-Base/raw/main/config.json）
- Qwen3-0.6B：`"vocab_size": 151936` → https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
- Qwen3-1.7B：`"vocab_size": 151936` → https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json
- Qwen2.5-0.5B：`"vocab_size": 151936, "max_position_embeddings": 32768` → https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json
- Qwen2.5-1.5B：`"vocab_size": 151936, "max_position_embeddings": 131072` → https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/config.json

**ModelScope 可得性（全部 Apache-2.0，组织名 `Qwen`）**
- Qwen3.5-0.8B → https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-0.8B（`"License":"apache-2.0"`）
- Qwen3.5-2B → https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-2B（`"BaseModel":["Qwen/Qwen3.5-2B-Base"]`）
- Qwen3-0.6B → https://modelscope.cn/api/v1/models/Qwen/Qwen3-0.6B ｜ Qwen3-1.7B → https://modelscope.cn/api/v1/models/Qwen/Qwen3-1.7B ｜ Qwen2.5-1.5B → https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-1.5B
- 人读页面：`https://modelscope.cn/models/Qwen/<name>`

**Qwen3.5 架构细节（对项目关键）**
- `model_type: qwen3_5`，`architectures: Qwen3_5ForConditionalGeneration`（**含 vision encoder**，多模态）
- 混合注意力：`Hidden Layout: 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`（每 4 层一个全注意力层）
- `mtp_num_hidden_layers: 1`（多 token 预测）
- 模型卡字段：`Token Embedding: 248320 (Padded)`、`LM Output: 248320 (Tied to token embedding)`
- **需要 `transformers ≥ 4.57.0.dev0`**（`config.transformers_version`）
→ https://huggingface.co/Qwen/Qwen3.5-0.8B 、https://huggingface.co/Qwen/Qwen3.5-2B 、https://huggingface.co/Qwen/Qwen3.5-2B-Base

### Q1.3 码长 M 与未用顶点（对本项目直接影响）

| V | M = ceil(log₂V) | 2^M | 未用顶点 | 相对 Llama-3.2-1B 的词表倍数 |
|---|---|---|---|---|
| **248,320**（Qwen3.5-0.8B/2B） | **18** | 262,144 | 13,824 | **1.94×** |
| **151,936**（Qwen3/2.5 小模型） | **18** | 262,144 | 110,208 | 1.18× |
| 128,256（Llama-3.2-1B，现状） | 17 | 131,072 | 2,816 | 1.00× |

> 笔记：Qwen3.5 的 248,320 是模型卡明示的 **"Padded"** 张量尺寸（softmax/lm_head 跨度的尺寸），**非填充的真实 token 数未查到**。对码本而言 248,320 即 v 维度，M=18；13,824 个未用顶点反而不多（比 151,936 的 110,208 少得多）。

### Q1.4 蒸馏证据（工单要求字段）

- **Qwen3-0.6B/1.7B：是。** 技术报告 §4.5 "Strong-to-Weak Distillation"：*"For smaller models, we use strong-to-weak distillation, leveraging both off-policy and on-policy knowledge transfer from larger models…"* → https://arxiv.org/html/2505.09388v1 （`#S4.SS5`）
- **Qwen2.5-0.5B/1.5B：无官方蒸馏记载**（报告覆盖 18T 预训练 + SFT/DPO/GRPO，无 strong-to-weak 章节）→ https://arxiv.org/html/2412.15115v2
- **Qwen3.5-0.8B/2B：未查到。** 模型卡/GitHub 只描述 "early fusion training on trillions of multimodal tokens" + RL，无蒸馏表述；且 **arXiv 上不存在 Qwen3.5 文本基座技术报告**（标题检索只有 Qwen3.5-Omni 报告）→ https://huggingface.co/Qwen/Qwen3.5-0.8B 、https://arxiv.org/abs/2604.15804

### Q1.5 选型建议

- **2B 级首选 `Qwen3.5-2B`。** 理由：同类最大词表 V=248,320（logits 比 151,936 多 63%），tied embeddings（级联解码可直接复用 embedding 行），Apache-2.0，有 Base 权重，ModelScope 同日可得——正是"大词表小模型"的终极场景。
  **风险**：新型混合线性/全注意力 + MTP + 多模态包装，需要 transformers ≥4.57.0.dev0——**先在集群/本地确认该版本前向能跑通并验证 `hidden_states[-1]` 仍等价于 lm_head 输入**。
- **0.8B 级首选 `Qwen3.5-0.8B`。** 与 2B **共用同一 tokenizer / V** ⇒ **一份码本同时服务两个规模**（省一次建码本）。
- **回退**：`Qwen3-1.7B`（2B 级）/ `Qwen3-0.6B`（0.8B 级）——经典 dense decoder、工具链成熟，但 V 小 39% 且为蒸馏骨干（会干扰"原生 vs 蒸馏"研究线索的解读）。
- 方法收益排序（词表驱动）：**Qwen3.5-2B > Qwen3.5-0.8B > Qwen3-1.7B/0.6B ≈ Qwen2.5-1.5B/0.5B**；全部 Apache-2.0、全部 ModelScope 有（国内友好）。

### Q1.6 Q1 未查到 / 存疑

- Qwen3.5 真实（非 padded）token 数 —— **未查到**（模型卡仅 "248320 (Padded)"）。
- Qwen3.5-0.8B/2B 蒸馏状态、Qwen3.5 文本基座技术报告 —— **未查到**。
- Base 变体精确发布日期 —— **未查到**（collection 显示 "Updated Apr 23"，GitHub News 未列）。
- 上下文长度 **model card（32,768）与 raw config（Qwen2.5-1.5B = 131072；Qwen3 = 40960）不一致** —— 已标注，下载后以 config 为准。

---

## Q2 Qwen 预训练数据公开信息（附不确定性标注）

### Q2.1 官方硬事实（可直接引用）

| 代际 | 预训练总量 | 语言数 | 关键官方表述 | 来源 |
|---|---|---|---|---|
| Qwen2 | **7T** tokens | ~30 | "7 trillion tokens" | https://arxiv.org/abs/2407.10671 |
| Qwen2.5 | **18T** tokens | **29** | "focus on knowledge, coding, and mathematics" | https://arxiv.org/abs/2412.15115 |
| **Qwen3** | **~36T** tokens | **119** | "nearly twice … approximately 36 trillion tokens covering 119 languages and dialects" | https://qwenlm.github.io/blog/qwen3/ ｜ https://arxiv.org/html/2505.09388v1 |
| Qwen3.5 | **未公布** | **201** | "Early fusion training on **trillions of multimodal tokens**" | https://github.com/QwenLM/Qwen3.5 ｜ https://huggingface.co/Qwen/Qwen3.5-397B-A17B |

**Qwen3 三阶段 token 分配（官方）**
- S1 通用：**>30T**，序列长 4K，覆盖 119 语言
- S2 推理：**约 5T** 更高质量 token，"increasing the proportion of STEM, coding, reasoning, and synthetic data"
- S3 长上下文：**数千亿**，32K（语料 75% 为 16K–32K、25% 为 4K–16K）
→ https://arxiv.org/html/2505.09388v1（§3.2）

**Qwen3 数据构成（定性，无比例）**
- 网页 + PDF 文档：用 Qwen2.5-VL 抽取文字、Qwen2.5 精修，"**trillions in total**"
- 合成数据：Qwen2.5 / Qwen2.5-Math / Qwen2.5-Coder 生成，"**trillions of text tokens**"（教科书、问答、指令、代码片段，覆盖数十领域）
- 标注系统对 **>30T tokens** 按教育价值/领域/学科/安全打标；**instance-level 数据配比优化**（在小型代理模型上做消融，而非仅按数据源/域调比）
→ https://arxiv.org/html/2505.09388v1

**Qwen2.5 过滤细节（最接近"dedup/质量"的公开信息）**
- 用 Qwen2-Instruct 作质量过滤器，多维打分；域重平衡——**下采样**电商/社交/娱乐，**上采样**技术/科学/学术
- 合成数据经 reward model 过滤；数学/代码数据取自 Qwen2.5-Math / Qwen2.5-Coder
- 去污染：**LCS ≥ 13 tokens 且 ≥ 0.6×min(len) 的训练序列被剔除**
→ https://arxiv.org/html/2412.15115v2（§3.1、§5）

**代码占比的官方锚点（仅代码专模型披露）**
- **Qwen3-Coder：7.5T tokens（70% code ratio）** → https://qwenlm.github.io/blog/qwen3-coder/
- Qwen2.5-Coder：继续预训练 **>5.5T tokens** → https://arxiv.org/abs/2409.12186

**旁证（仅多模态 Omni，勿外推文本基座）**
- Qwen3.5-Omni S2 通用阶段约 **4T tokens**：text 0.92T / audio 1.99T / image 0.95T / video 0.14T / video-audio 0.29T；音频编码器 AuT 数据语言比 "Chinese, English and multilingual … **3.5 : 3.5 : 3**"（≈35:35:30，**仅音频对**）
→ https://arxiv.org/html/2604.15804v1（§3）

### Q2.2 ⚠️ 有界估计（**非官方数字**，仅作配比起点）

官方从未公布 English/Chinese/其余 的语言百分比，也无基座代码占比。下面是**基于公开事实的推断**：

| 成分 | 估计区间 | 推理依据 |
|---|---|---|
| English | **~40–60%** | 119 语言覆盖，但网页语料天然英文主导；Qwen 定位中英双语 |
| Chinese | **~15–30%** | 中文能力是 Qwen 核心卖点、评测单列 "English & Chinese"，但无任何官方比例 |
| Code | **~5–12%** | S2 仅 ~5T/36T ≈ 14%，且是 STEM+code+reasoning+synthetic 合计；代码专模型才 70%（Qwen3-Coder）⇒ 基座远低于此 |
| 其他（多语 + 数学/STEM + 合成非代码） | **~15–30%** | 119 语言承诺 + "trillions" 合成/PDF 数据 |

**据此，下游新数据集的起点建议：English 50–60% / Chinese 20–30% / Code 10–15%（估计值）。**

### Q2.3 Q2 未查到清单（不要编）

- ❌ 任何 Qwen 世代的 **English/Chinese/其他 语言百分比**（全部代际均未公布）。
- ❌ 基座 Qwen3/Qwen3.5 的 **代码占比 / 合成数据占比 / PDF 抽取 token 占比**（只有 "trillions" 定性）。
- ❌ **Qwen3.5 文本预训练总 token 数**（博客/模型卡/GitHub 均无；无文本技术报告）。
- ❌ Qwen3 的具体 **去重（dedup）方法**（报告未写）。
- ❌ 中文媒体（机器之心/量子位等）对具体配比的报道：本次未能获取（qbitai 403；Bing 中文检索被污染；Exa 搜索额度耗尽）。**建议用户人工复核一次。**

---

## Q3 数据集选材（总预算 ~2B tokens，中国大陆网络）

> 核验环境说明：中国大陆用户走 **hf-mirror.com**（`HF_ENDPOINT=https://hf-mirror.com`）与 **ModelScope 直连**。本次调研从非大陆 IP 测试，hf-mirror 会 geo-redirect 到 huggingface.co（预期行为），但 **hf-mirror 的 `/api/...` 与 `resolve` 代理可用**，且 ModelScope 的 `/api/v1/datasets/<id>` 全部 200。所有 id 均已在 2026-09-14 逐条核验。

### Q3.1 中文语料候选

| 候选 | HF repo id | ModelScope id | 规模（已公布） | 许可 | 格式 | 门禁/注册 |
|---|---|---|---|---|---|---|
| **CCI 3.0-HQ** ⭐ | `BAAI/CCI3-HQ` | `BAAI/CCI3-HQ`（116k DL） | **~500 GB**；679 个 jsonl（`data/part_000000.jsonl`…） | Apache-2.0 | jsonl | HF gated=auto（登录+接受）；MS 需登录 + BAAI 使用协议 |
| **FineWeb-2 `cmn_Hani`** ⭐ | `HuggingFaceFW/fineweb-2` | `HuggingFaceFW/fineweb-2`（81k DL） | **543.5B words / 636,058,984 docs / 2.42 TB 文本 / 1.48 TB parquet**；train 370 shards | ODC-BY | parquet | **无**（gated=false） |
| CCI 3.0（全量） | `BAAI/CCI3-Data` | `BAAI/CCI3-Data`（22k DL） | ~1000 GB；1,193 jsonl | Apache-2.0 | jsonl | HF gated=auto；MS 登录 |
| OpenCSG Fineweb-Edu-Chinese V2.1 | `opencsg/Fineweb-Edu-Chinese-V2.1` | 同名（510k DL） | 2.42 TB；4-5: 46B tok / 3-4: 530B tok / 2-3: 930B tok | Apache-2.0 + OpenCSG CL（商用需邮件） | parquet | 无 |
| CCI 4.0-M2 Base（2025/2026） | `BAAI/CCI4.0-M2-Base-v1` | 同名（178k DL） | Web-ZH 5,161 GB；全套 5.72 TB | Apache-2.0 | json | HF gated=auto；MS 登录 |
| Ultra-FineWeb（2025） | `openbmb/Ultra-FineWeb` | `OpenBMB/Ultra-FineWeb`（466k DL） | ~1T EN + **~120B ZH**（zh 492.7 GB） | Apache-2.0 | parquet | 无 |
| SkyPile-150B | `Skywork/SkyPile-150B` | 无官方 id（社区镜像 `swift/SkyPile-150B` 等） | **~150B tokens / ~233M 页 / 620 GB 文本**；441 文件 | Skywork Community License | jsonl | 无（gated=false） |
| TeleChat-PTD | `Tele-AI/TeleChat-PTD` | `TeleAI/TeleChat-PTD` | ~1 TB 原始 / 480 GB 压缩；**189 个 `data/N.jsonl.gz`** | Apache-2.0 | jsonl.gz | 无 |
| ChineseWebText 2.0 | `CASIA-LM/ChineseWebText2.0` | 未查到 | 3.8 TB（仓库 1.83 TB），带 quality_score/domain/toxicity | Apache-2.0 | json | 无 |
| ChineseWebText 1.0 | `CASIA-LM/ChineseWebText` | 未查到 | 1.42 TB | **未声明许可** | jsonl | 无 |
| MAP-CC | `m-a-p/MAP-CC` | 未查 | 800B tokens（宣称） | **CC-BY-NC-ND-4.0（禁商用+禁演绎）** | json | 无 |
| WanJuan-CC | `opendatalab/WanJuan-CC` → **401 不可访问** | `Shanghai_AI_Laboratory/WanJuanCC` | 100B tokens；**注意：这是英文** | CC-BY-4.0 | jsonl | OpenDataLab |
| WanJuan 1.0（中文） | — | `Shanghai_AI_Laboratory/WanJuan1_dot_0` | 2 TB 多模态；文本 >1 TB | GitHub 说 CC-BY-4.0，**ModelScope 显示 CC-BY-NC-4.0（冲突）** | jsonl | OpenDataLab |
| HPLT zh | `TiWu-Lab/HPLT-zh` | — | zh 子集 1,198 GiB | **CC0-1.0（最干净）** | jsonl.zst / parquet | 无 |
| WuDao | 仅社区镜像（`p208p2002/wudao` 等） | — | 220 GB 版 ≈125 GB parquet | 混合 | parquet | 原始需 SciDB 账号（未核实） |
| CLUECorpus2020 | — | — | 100 GB | **仅申请制，无许可声明** | txt | 邮件 CLUEbenchmark@163.com |

**中文要点**
- **官方没有叫 "fineweb-zh" 的数据集**；只有社区衍生（`TiWu-Lab/fineweb-2-zh` ODC-BY；`lianghsun/fineweb-2-zhtw` gated=manual）。
- **per-shard 估计**：FineWeb-2 `cmn_Hani` 370 shards ⇒ **每 parquet ≈ 1.3–1.5B tokens（估计）**；CCI3-HQ 679 文件 ⇒ ~0.76 GB/文件 ⇒ **~150–200M tokens/文件（估计）**；TeleChat-PTD 189 文件 ⇒ ~2.5 GB/文件。
- **许可风险清单**：MAP-CC（NC+ND）、WanJuan 1.0（MS 显示 NC）风险高；OpenCSG / SkyPile 商用需邮件/协议；CCI（Apache-2.0）、FineWeb-2（ODC-BY）、Ultra-FineWeb（Apache-2.0）、TeleChat-PTD（Apache-2.0）、HPLT（CC0）对研究友好。

### Q3.2 代码语料候选

| 候选 | repo id | 规模 | 许可/过滤 | 格式 | 门禁 | 模型站 | 中国大陆下载 |
|---|---|---|---|---|---|---|---|
| **The Stack v3 train** ⭐ | `HuggingFaceCode/stack-v3-train` | 3.55 TB parquet / 8,192 shards；11.5 TB 文本；**permissive 子集 507 GB / ~159.3B tokens** | ODC-BY；逐文件 `license_type`（permissive/no_license），**非 permissive 已剔除** | parquet，**正文内联**（`content`），文件级行 | **无** | `HuggingFaceCode/stack-v3-train`（镜像在传） | hf-mirror（免 token）或 `modelscope download` |
| Stack v2（full） | `bigcode/the-stack-v2` | 427 GB 仓库 = **SWH 元数据** | gated=auto；**批量正文需 SWH/INRIA 协议** | parquet 元数据（blob id） | auto | `bigcode/the-stack-v2` | 元数据可，正文走 SWH |
| Stack v2 dedup | `bigcode/the-stack-v2-dedup` | 185 GB 仓库 | 同上 | parquet 元数据 | auto | 有 | 同上 |
| Stack v2 train-full-ids | `bigcode/the-stack-v2-train-full-ids` | 45.4 GB / 64 parquet | gated=auto；schema 含 `license_type` | parquet 元数据 | auto | — | 同上 |
| Stack-Edu | `HuggingFaceTB/stack-edu` | 34.9 GB 元数据（125B tokens，15 语言） | ungated；Stack v2 条款 | parquet **仅元数据**（blob_id） | **无** | — | 元数据易；正文走 SWH |
| StarCoderData | `bigcode/starcoderdata` | 783 GB / 86 语言 / ~250B tok | gated=auto（Stack 条款）；由 starcoder-dedup 构建（permissive） | 按语言 parquet | auto | **`AI-ModelScope/starcoderdata`** | ModelScope 直连（最佳） |
| The Stack v1 | `bigcode/the-stack` | ~1.43 TB；6,828 parquet；358 语言 | gated=auto；**v1.1 = 仅 permissive** | 按语言 parquet，正文内联 | auto | 未找到 | hf-mirror + token |
| The Stack v1 dedup | `bigcode/the-stack-dedup` | ~2.64 TB；5,144 parquet | gated=auto | parquet | auto | `bigcode/the-stack-dedup`（182.9 GB，疑似部分） | hf-mirror + token |
| GitHub Code（clean） | `codeparrot/github-code-clean` | ~1 TB 文本；880 parquet | **ungated**，Apache-2.0 标签；**有 `license` 列** | parquet；schema `[code, repo_name, path, language, license, size]` | **无** | — | hf-mirror 直连 |
| GitHub Code | `codeparrot/github-code` | 669.9 GB；1,126 parquet；115M 文件 | ungated；含 GPL 等 15 种许可（需自行过滤） | parquet | 无 | — | hf-mirror 直连 |
| CodeParrot clean | `codeparrot/codeparrot-clean` | 34.3 GB；55 `json.gz` | ungated；仅 Python | JSON gz | 无 | 未找到 | hf-mirror 直连 |
| python-edu | `HuggingFaceTB/smollm-corpus`(`python-edu`) | 989 MB 元数据；7.68M Python 文件 | ungated，ODC-BY | parquet **仅元数据** | 无 | — | 正文走 SWH |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus`(`cosmopedia-v2`) | 212.5 GB；39.1M 样本 | ungated | parquet | 无 | — | ⚠️ 合成网页，**非代码语料** |
| Nemotron code v1 | `nvidia/Nemotron-Pretraining-Code-v1` | 246 GB | **gated=manual（NVIDIA 审批）** | parquet | manual | — | 不实用 |

**代码要点**
- **重大新发现：The Stack v3（2026-07-23）** `HuggingFaceCode/stack-v3-train`：**不需要 Software Heritage**、正文内联、逐文件 `license_type`、含 ModelScope 镜像 ⇒ 推荐首选。JSON 统计（`stats/train/totals.json`）：11.5 TB 文本、2.27B 文件、~3.61T tokens（Qwen3-Coder-Next-Base tokenizer 估）；**permissive 子集 ~159.3B tokens**。
- **Stack v2 的真实路径是 SWH**（HF 上的 v2 仓库只有 427 GB 元数据，装不下 67.5 TB 正文）：需 `s3://softwareheritage/content/{blob_id}` 或 `archive.softwareheritage.org` 逐 blob 拉取，批量需 SWH/INRIA 协议；对 200–400M tokens 意味着 10 万+ 次抓取，**不实用**。
- `bigcode/*` 正文数据集全是 **gated=auto**（须 HF 账号接受条款 + token）。**hf-mirror 能否代下载 gated 资源未核实** ⇒ 推荐只用 ungated 源。
- "Google GitHub Code Clean" **在 HF 上没有公开 `google/*` 仓库**（`google/github-code-clean` 401）；用户所指很可能就是 `codeparrot/github-code-clean`（源自 Google BigQuery GitHub 快照）。

### Q3.3 英文语料候选

| 候选 | HF id | 规模 | 许可 | 中国大陆摩擦 |
|---|---|---|---|---|
| **FineWeb-Edu** ⭐ | `HuggingFaceFW/fineweb-edu` | **1.3T tokens**（v1.4.0，2025-07-11；2026 Smol-Data 卡写 "~1.6T"，两说并存） | ODC-By 1.0 | hf-mirror 可；ModelScope 官方镜像 `HuggingFaceFW/fineweb-edu` 存在（122k DL） |
| FineWeb-Edu `sample-100BT` | 同上，tier | **140 文件 / 266.7 GB**，每文件 ≈ **0.71B gpt2 tokens** | ODC-By | 同上 |
| fineweb_edu_100BT（现成切片） | `HuggingFaceFW/fineweb_edu_100BT` | ~100B tokens，seed 42，299 GB | ODC-By | 同上（2026-03 发布） |
| fineweb-edu-score-2 | `HuggingFaceFW/fineweb-edu-score-2` | 5.4T tokens（阈值更松，质量更低） | ODC-By | — |
| fineweb-2（多语） | `HuggingFaceFW/fineweb-2` | ~20 TB / 5B docs / >3T words / 1,870 语言 | ODC-By | **非英文 edu，不是 FineWeb-Edu 继任** |
| DCLM-baseline-1.0 | `mlfoundations/dclm-baseline-1.0` | 4T tokens / 3B docs | CC-BY-4.0 | 有 ModelScope 镜像 `AI-ModelScope/dclm-baseline-1.0` |
| Dolma v1.7 | `allenai/dolma` | 2,308.5B OLMo tokens / 4.5 TB gzip | ODC-BY | 仓库仅 URL 列表，数据走 Ai2 直链；2026 继任 **Dolma 3**（`allenai/dolma3_pool` / `dolma3_mix-6T`，ungated） |
| Zyda-2 | `Zyphra/Zyda-2` | 5.07T tokens；`sample-100BT` = 252 GB / ~100B gpt-neox | ODC-BY | 现成英文混合 |
| Nemotron-CC-v2 | `nvidia/Nemotron-CC-v2` | 6.586T tokens / 10.3 TB | "other"（NVIDIA 协议，**仅训练用**） | **gated=manual ⇒ 跳过** |
| RedPajama-Data-1T | `togethercomputer/RedPajama-Data-1T` | 1.2T tokens | 无统一许可 | 仓库仅 URL 列表，数据走 `data.together.xyz` |
| SlimPajama-627B | `cerebras/SlimPajama-627B` | 627B tokens | Apache-2.0（历史） | **原仓库 2026-09-14 返 401（gated/移除，未核实）**；用重传版 |
| FineWeb-HQ | `epfml/FineWeb-HQ` | FineWeb top-10%（规模未核实） | ODC-BY | 2025-09 发布，1B 模型消融称优于 DCLM 与 FineWeb-Edu |
| Ultra-FineWeb | `openbmb/Ultra-FineWeb` | ~1T EN + 120B ZH | Apache-2.0 | 中国团队，混合 EN+ZH 可选 |

**英文结论**：截至 2026-09-14 **不存在 FineWeb-Edu-v2**；`fineweb-edu` 自 v1.4.0（2025-07-11）未再更新。继续用 **`sample-100BT`**（与已完成的 1B 英文 run 同源，保证连续性）。

### Q3.4 ~2B-token 配比建议

> 目标：复刻 Qwen 分布（多语言 + 代码），补上现有英文 FineWeb-Edu 蒸馏导致"**中文顶点拿不到梯度**"的缺口。

| 成分 | 占比 | tokens | 首选源 | 切片量 |
|---|---|---|---|---|
| **English** | **~55%** | ~1.05B | `HuggingFaceFW/fineweb-edu` `sample-100BT` | **2 个 parquet ≈ 1.43B**，按 `token_count` 截到 ~1.05B |
| **中文** | **~30%** | ~0.6B | 主：`BAAI/CCI3-HQ`；辅：`HuggingFaceFW/fineweb-2`(`cmn_Hani`) | CCI3-HQ **3–4 个 jsonl（估计 ~0.5–0.8B）** + FineWeb-2 cmn_Hani **1 个 shard 的 ~40%（~0.5B）**，合计按需截 |
| **代码** | **~15%** | ~0.3B | `HuggingFaceCode/stack-v3-train`（permissive 过滤） | **~15–18 个 shard ≈ 200–400M**（估计每 shard ~18–20M permissive tokens） |

**配比理由（可辩护但为估计）**：Qwen3 官方唯一硬事实是 "36T / 119 语言 / S2 仅 ~14% 用于 STEM+code+reasoning+synthetic 强化 / 代码专模型 70% code"；英文仍应占多数，中文取 Qwen 双语定位的 ~30%，代码取 10–15%（远低于代码专模型的 70%）。**这份比例是估计，不是 Qwen 披露值。**

### Q3.5 采购步骤清单（按顺序执行）

> 通用：`pip install -U "huggingface_hub[cli]" datasets pyarrow duckdb modelscope`；
> Windows PowerShell 设环境变量：`$env:HF_ENDPOINT="https://hf-mirror.com"`。
> ⚠️ **Xet 警告（2026 新情况）**：HF Hub 已全面 Xet 化，hf-mirror 的 Xet 桥有报告不稳定。若出现 `cas-bridge.xethub.hf.co` 超时/损坏，加 `$env:HF_HUB_DISABLE_XET="1"` 重试，并**校验 SHA-256**。

**① 英文（~1.05B tokens）**
```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
hf download HuggingFaceFW/fineweb-edu --repo-type dataset `
  --include "sample/100BT/000_00000.parquet" "sample/100BT/000_00001.parquet" `
  --local-dir data/fineweb_edu_100bt
# 每文件 ≈0.71B tokens；按 token_count 列截到 ~1.05B（或流式 token 预算）
```
- 卡片 https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu ｜ 已知 SHA-256：`000_00000` = `955ff462…f32b695`，`000_00001` = `77665438…d76293680`（`hf` 会自动校验）。

**② 中文（~0.6B tokens）**
```powershell
# 主：CCI3-HQ（需登录；ModelScope 优先）
modelscope download --dataset BAAI/CCI3-HQ `
  --include 'data/part_0000*.jsonl' --local_dir data/cci3hq
# 备（HF 镜像，需 HF 登录并接受 gate 一次）：
# hf download BAAI/CCI3-HQ --repo-type dataset --include "data/part_000000.jsonl" --local-dir data/cci3hq

# 辅：FineWeb-2 cmn_Hani，无门禁（取 1 个 shard 的 ~40%）
hf download HuggingFaceFW/fineweb-2 --repo-type dataset `
  --include "data/cmn_Hani/train/000_00000.parquet" --local-dir data/fineweb2_zh
```
- CCI3-HQ https://modelscope.cn/datasets/BAAI/CCI3-HQ ｜ FineWeb-2 https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- 更省事的中文单源替代：`Tele-AI/TeleChat-PTD`（1–2 个 `data/0.jsonl.gz`，Apache-2.0，无门禁）https://huggingface.co/datasets/Tele-AI/TeleChat-PTD

**③ 代码（~0.3B tokens，permissive）**
```powershell
# 先取 1 个 shard 测 permissive 产率（已验证 shard）
hf download HuggingFaceCode/stack-v3-train --repo-type dataset `
  --include "data/part-00000-7d88aa48*" --local-dir data/stack_v3
# 再扩到 ~15 个同 UUID 批次
hf download HuggingFaceCode/stack-v3-train --repo-type dataset `
  --include "data/part-0000*-7d88aa48*" "data/part-0001[0-4]-7d88aa48*" --local-dir data/stack_v3
```
```sql
-- DuckDB 过滤取出正文（language 名为 go-enry/Linguist 命名，先 GROUP BY 确认）
COPY (SELECT content FROM read_parquet('data/stack_v3/*.parquet')
      WHERE license_type='permissive'
        AND language IN ('Python','JavaScript','TypeScript','Java','C','C++','C#','Go','Rust','Shell')) 
TO 'data/code_slice.parquet' (FORMAT PARQUET);
```
- ModelScope 替代（避开镜像依赖）：`modelscope download --dataset HuggingFaceCode/stack-v3-train --include 'data/part-0000*-4beed122*' --local_dir data/stack_v3_ms`
- **注意**：ModelScope 镜像分多批上传，shard 名含 UUID（如 `4beed122…`/`50e95205…`/`990b4288…`），**只用同一批次**避免重复/漏取。
- 回退（按顺序）：① `AI-ModelScope/starcoderdata`（ModelScope 直连，permissive 构造，Python 59 shards）② `codeparrot/github-code-clean`（hf-mirror，按 `license IN ('mit','apache-2.0','bsd-3-clause','bsd-2-clause','cc0-1.0','unlicense','isc','artistic-2.0')` 过滤）③ 社区 Stack-Edu 重打包（未核实，慎用）。

**④ 入库**：三路按 Qwen3.5 tokenizer 编码为 `.npy` 缓存；**建议按比例交错**（见 Q3.4），并在缓存里打上来源标签，便于后续审计与 ablation。

### Q3.6 数据可下载性/许可风险

- **最稳组合**（无门禁 + 许可干净）：FineWeb-Edu（ODC-By）+ FineWeb-2 cmn_Hani（ODC-BY）+ The Stack v3 permissive（ODC-BY）。CCI3-HQ 需登录/协议但 Apache-2.0。
- **需注意**：CCI3 系（HF gated=auto / MS 登录+协议）；OpenCSG / SkyPile（商用需邮件）；MAP-CC（NC+ND）与 WanJuan 1.0（NC）**不建议**用于任何可能商用的场景。
- **未核实**：hf-mirror 对 gated 资源的代下载行为；SWH S3 桶的公开性/凭据；Stack v3 逐语言 permissive 比例；ModelScope Stack v3 镜像完整性。

---

## 附 A：Q3 未查到清单

- 官方 HF 数据集 `fineweb-zh`：**不存在**（仅社区衍生）。
- `Skywork/SkyPile-150B` 官方 ModelScope id：**未查到**（仅社区镜像）。
- `m-a-p/MAP-CC` 的 ModelScope 页面：**未查**。
- `opendatalab/WanJuan-CC` 在 HF：**401 不可访问**；且它是**英文**语料，非中文。
- "MiChao" 作为独立公开数据集：**未查到**（仅作为 Ultra-FineWeb 内部来源被提及）。
- Google "GitHub Code Clean" 的 `google/*` 官方 HF 仓库：**不存在**（401）。
- SlimPajama-627B 原仓库状态/许可（401）：**未核实**。
- FineWeb-HQ 精确 token 量：**未查到**。

## 附 B：来源索引（主要）

- 型号：https://github.com/QwenLM/Qwen3.8 ｜ https://huggingface.co/collections/Qwen/qwen35 ｜ https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/config.json ｜ https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json ｜ https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-0.8B
- 预训练数据：https://arxiv.org/abs/2505.09388 ｜ https://arxiv.org/html/2412.15115v2 ｜ https://arxiv.org/abs/2407.10671 ｜ https://arxiv.org/abs/2409.12186 ｜ https://qwenlm.github.io/blog/qwen3-coder/ ｜ https://arxiv.org/html/2604.15804v1
- 中文语料：https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 ｜ https://modelscope.cn/datasets/BAAI/CCI3-HQ ｜ https://modelscope.cn/datasets/BAAI/CCI4.0-M2-Base-v1 ｜ https://huggingface.co/datasets/openbmb/Ultra-FineWeb ｜ https://huggingface.co/datasets/Skywork/SkyPile-150B ｜ https://huggingface.co/datasets/Tele-AI/TeleChat-PTD
- 代码语料：https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train ｜ https://modelscope.cn/datasets/HuggingFaceCode/stack-v3-train ｜ https://huggingface.co/datasets/bigcode/the-stack-v2 ｜ https://modelscope.cn/datasets/AI-ModelScope/starcoderdata ｜ https://huggingface.co/datasets/codeparrot/github-code-clean
- 英文语料：https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu ｜ https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT ｜ https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0 ｜ https://huggingface.co/datasets/epfml/FineWeb-HQ
- hf-mirror/Xet：https://github.com/padeoe/hf-mirror-site/issues/50 、/60 、/64 、/66 ｜ https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables

*报告完 · 2026-09-14*
