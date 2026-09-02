# P0 报告：环境、模型冒烟、top-k 覆盖率、码本构建与验证

日期：2026-08-31 · 状态：P0 完成，全部决策已落定，进入 P1

## 1. 环境与模型

- cose2 环境：torch 2.11.0+cu130，CUDA 可用（RTX 4060 Laptop 8GB）；新装 transformers 5.16.1 / datasets 5.0.1 / accelerate 1.14.0 / sentencepiece。
- 模型：魔搭镜像 `LLM-Research/Llama-3.2-1B`（HF 申请未通过，ModelScope 下载），safetensors 2.47GB，参数 1,235,814,400 —— 与官方 API 数字完全一致。
- 加载冒烟通过（`scripts/smoke_test.py`）：
  - 嵌入 tied 确认（`lm_head.weight is embed_tokens.weight`）。
  - **`hidden_states[-1]` 与 lm_head 输入逐项相等（max diff = 0）**——MDN 头的输入点确定为骨干最终 RMSNorm 输出。
  - 加载后显存 2.32 GiB，本地 8GB 可跑小批量。

## 2. top-k 覆盖率实测（`scripts/measure_coverage.py`，BabyLM-10M，~24.5k 位置）

| k | mean | median | p10 | p90 |
|---|---|---|---|---|
| 40 | 0.777 | 0.837 | 0.461 | 0.990 |
| 128 | 0.860 | 0.918 | 0.620 | 0.995 |
| 256 | 0.897 | 0.948 | 0.707 | 0.997 |
| 512 | 0.927 | 0.967 | 0.785 | 0.998 |
| 1024 | 0.951 | 0.980 | 0.856 | 0.999 |

**结论：BabyLM 域上教师分布明显弥散，top-40 只覆盖 ~78% 质量。** 这与"top-32 足够"的直觉（针对分布内文本）不符。据此用户拍板 **k=1024**。注：损失计算中 k 的成本线性且极小（目标在每步前向中顺带提取，无需存储）。用户另备注：BabyLM 是图数据小的一时之选，非必须；若结果不佳可换数据集。

## 3. 码本构建（`scripts/build_codebook.py`）

### 3.1 构建方法的重要修正
- **原 spec**：每层对当前簇做 2-means（cosine）直接二分。
- **实测失败**：LLM 嵌入集中在单位球面的窄锥内，2-means 每层只剥下离群小簇，主簇几乎不收缩 → 85% token 在第 17 层仍未成单点，仅 27254 个合法顶点，码本无效。
- **修正**：2-means 仅用于找分裂方向（取两质心差 w），实际切分改为**沿 w 投影的中位数平衡切分**（按秩，平手安全）。每层严格对半 → 树深恰好 ≤17，所有 128256 个码字唯一。此即文献中的平衡谱树/谱哈希式构造（spectral hashing, Weiss et al. 2008 的逐维中位数阈值思想），与原 spec 的意图（cosine 语义二叉树）一致。
- 结果：合法顶点 128256 / 131072（空余 2816 符合预期），无回退分裂，构建耗时 ~13 分钟（一次性）。

### 3.2 语义性验证
全局指标（弱，符合预期）：corr(Hamming, cosine 距离) = 0.041（随机码对照 0.003）；cosine 最近邻平均 Hamming 7.51/17（对照 8.42）。全局弱是树结构只保留 17 个全局分裂方向的固有性质。

**操作性指标（强，这才是关键的）**——`scripts/measure_code_compactness.py`，教师 top-40 目标集合内的加权平均两两 Hamming：

| 集合 | 值 |
|---|---|
| **树码本 · 教师 top-40** | **4.16** |
| 随机码本 · 同样 top-40 | 6.43 |
| 树码本 · 随机 40 token | 8.29 |
| 深度-8 前缀去重数/位置 | ~9.3（40 个目标收进 ~9 个子树） |

即：在教师实际给质量的目标集合上，树码本比随机码本紧致 35%；40 个目标收敛于 ~9 个深度-8 子树，32 分量 MoB 覆盖余裕充足。**码本通过验证。**

## 4. 产物清单

- `data/babylm_data/`（10M/100M/dev/test，已解压校验）
- `data/codebook.npz`（codes[128256,17] uint8、vertex_id、valid_vertices[131072]、depth_reached）+ `data/codebook_meta.json`
- `scripts/`：smoke_test.py、measure_coverage.py、build_codebook.py、measure_code_compactness.py
- `LLM-Research--Llama-3.2-1B/`（魔搭镜像权重，config 与官方一致）

## 5. 决策结果（2026-08-31 用户拍板）

- **蒸馏目标 top-k = 1024**（覆盖率 95.1%；若效果不佳可考虑换数据集——BabyLM 非必须）。
- **ECC：主线不加**（17 位纯树码），留 P4 消融。

## 6. 下一步（P1 冒烟，本地）

MoB 头（3×2458 MLP + 576 输出）按 docs/03 §4.2–4.4 实现；千级位置过拟合：NLL 下降、分量告警面板、解码复现教师 top-1 的比例、显存画像。
