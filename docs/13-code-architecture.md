# 代码架构与运行机制

日期：2026-09-02 · 读者：接手开发的新 agent（含上下文清空/换 harness 的场景）· 性质：施工级细节文档，写到一个新人能直接改代码的程度

## 0. 仓库布局

**2026-09-11 重组（用户要求）**：`scripts/` 平铺结构废除，按职责分四目录 + slurm。**全部入口以模块形式从仓库根运行**（`python -m train.p2_train ...`，不要 `python train/p2_train.py`——包导入依赖根目录在 sys.path 上）。旧文档/日志里的 `scripts/X.py` 路径按 `X.py` 的归属目录心算映射即可。

```
llm_codec/
├── mob/         # 模型源码：mob_common.py（头族 MoB/Linear/ResMLP/SwiGLU、损失、全部解码路径、ECC 基建）
├── train/       # 训练入口：p2_train.py（单头蒸馏）、p2_train_dual.py（多臂共享基座）、p4_joint_train.py、
│                #   p4_backbone_probe.py、p5_twin_pretrain.py、p7_lr_scan.py、train_encoder.py（已终止）、p1_smoke.py
├── eval/        # 评测与分析：p3_eval.py（验收）、eval_topk_recall.py、eval_rerank_gen_ppl.py、p8_eval_adaptive.py、
│                #   p4_leakage_check.py、analyze_*.py、plot_*.py、benchmark_decode.py、gen_samples.py、measure_*.py
├── tools/       # 一次性工具：build_codebook.py、build_fineweb_cache.py、smoke_test.py、dump_lr_scan.py 等
├── slurm/       # 全部集群作业模板（.sbatch）
├── LLM-Research--Llama-3.2-1B/snapshots/master/   # 冻结骨干（魔搭镜像，bf16 safetensors 2.47GB + tokenizer）
├── data/
│   ├── codebook.npz / codebook_meta.json / codebook_smollm2.npz   # 语义二叉树码本 + 元数据
│   ├── babylm_data/                               # BabyLM 2023 原始文本（10M/100M/dev/test）
│   ├── *_tokens.npy                               # int32 token 缓存（babylm_10m / babylm_dev / fineweb_*）
├── runs/                                          # 训练/评测产物（checkpoint、train_log.jsonl、metrics）
├── docs/                                          # 全部文档（编号递进）
└── AGENTS.md                                      # 项目级指令源
```

包机制：四个代码目录各带空 `__init__.py`；跨目录引用一律绝对导入（`from mob.mob_common import ...`、`from train.p2_train import kmeans_vertex_init`）。本地环境：conda env `cose2`（`E:\MiniConda\envs\cose2\python.exe`，conda 不在 PATH 上，直接调 env 内 python）。集群：burgundy.hpc.cityu.edu.hk:22，远端根目录 `/gpfs1/home/yihe47/llm_codec`，python 在 `/home/yihe47/miniconda3/envs/cose2/bin/python`——集群同样 `cd` 到仓库根后 `$PY -m train.xxx`。

## 1. 核心库：`mob/mob_common.py`

共享的 MoB 头定义与全部数值核心。常量 `H=2048, WIDTH=2458, K=32, M=17`（M 永远 17；K/WIDTH 是旧默认——新代码一律走构造参数）。

- `MoBHead(k=K, width=WIDTH)`：`trunk = 3×(Linear→SiLU)`（H→width→width→width），`out = Linear(width, k*M + k)`，out 权重 `N(0, 1e-3)` 初始化（配合 bias 注入做顶点 init）。forward 返回 `(bit_logits [n,k,M], pi_logits [n,k])`。**改容量就改这两个参数，checkpoint 不跨配置兼容**（评测脚本要带匹配的 `--k/--width`）。
- `mob_mix_logprob(bit_logits, log_pi, codes) -> [n,k]`：对数混合概率。核心恒等式 `log Bern(c;z) = c·z + logsigmoid(−z)`（z 是 bit-logit）——免 exp、免溢出；`einsum("nkm,njm->nkj", codes, bit_logits)` + `logsigmoid(−z).sum(M)` 得分量对数概率，再 `logsumexp(log_pi + comp, dim=K)`。**全部 fp32，别放进 autocast**。
- `decode_greedy(bit_logits, log_pi, vid_to_tok, codes_all) -> token ids`：argmax π → 逐位取整 → 打包顶点 id → `vid_to_tok` 查表；落空（2816 个空余顶点之一）时对该位置做全表 Hamming 最近邻（`codes_all != bits` 求和取 argmin）。

## 2. 生产训练：`train/p2_train.py`

### 数据流
```
文本 → tokenizer → int32 token 缓存（.npy，一次性构建）
→ np.lib.stride_tricks.as_strided 切出 [n_chunks, T+1] 视图（stride=T，相邻块重叠 1 token 供 shift）
→ 每步取 batch 块（默认 8×512）→ 冻结骨干前向（fp16/bf16）
→ h = hidden_states[-1]（即 lm_head 原输入，已验证逐项相等）+ logits topk(1024) → 权重 = softmax(top-k logits)（top-k 内重归一）
→ MoB 头 fp32 前向 + 加权 NLL（按 --loss-chunk 分块防显存爆）→ 反传只进头
```

### 关键机制（都是踩过坑后定型的）
- **顶部初始化**：`kmeans_vertex_init`——按目标分布多项式采样 2 万个目标顶点 → torch k-means（kmeans++ 式起始 + 30 轮 Lloyd）→ 中心取整为顶点 → bit-logits ±`--init-c 4` 写进 `out.bias`（out 权重近零 → 初始输出≈bias，与 h 无关）。
- **熵正则**：`+λ·Σπ log π`（λ=`--lambda-ent 0.01`），拉平 π 防权重萎缩。
- **DAEM**：`log_pi = log_softmax(pi_logits / T)`，T 从 `--daem-t0 2.0` 线性降到 1.0，占前 `--daem-frac 0.10` 总步数。
- **日程** `--schedule`：`cosine`（200 warmup + 余弦到 0.1×peak）或 `wsd`（warmup → 平台 → 末 `--decay-frac 0.35` 线性降到 0.1×peak）。峰值 `--lr`，warmup 步数 `--warmup`。
- **日志**：`runs/<outdir>/train_log.jsonl` 每 `--log-every 200` 步一条（step/lr/nll/ent/T/elapsed_s），每 `--panel-every 1000` 步一条 dev 面板（n_eff/dead/dev_top1_agree/dev_nll）。
- **checkpoint**（2026-09-02 强化，实测验证）：每 epoch 末存 `head_epochN.pt`（永久）；`--ckpt-every N` 周期存**全状态归档** `head_stepN.pt`（head + 优化器 + 全局步数 + 当前 epoch 数据顺序），按 `--ckpt-keep 3` 轮转保留最近 3 份，另存一份权重版 `head_latest.pt` 供评测脚本直接用。
- **断点续跑（真·resume）**：`--resume <head_stepN.pt>` 恢复 head+优化器+步数+数据顺序，从断点精确继续（LR 日程按全局步数无缝衔接；实测：step-49 恢复后 epoch 终点指标与原跑逐项相同）。旧格式（纯权重 state_dict）仍可 `--resume`，退化为热启动（日程重来，需手动对齐 `--lr/--epochs`）。注意：真续跑须用相同的 `--epochs/--seq-len/--batch`（总步数决定日程）。K=128 头单份全状态 checkpoint 约 390MB（权重 129MB + 优化器 258MB），3 份归档约 1.2GB。
- **显存公式**（K=128 为例）：损失分块 [loss_chunk, topk, K] fp32 每块 ~loss_chunk×topk×K×4B ×~2（term+comp），骨干 logits batch×512×V×2B。8GB 卡用 `--loss-chunk 1024`；V100 32GB 用 2048-4096。

### 完整 CLI（默认值即 BabyLM 时代基线）
`--data-root --model-dir --outdir --seq-len 512 --batch 8 --epochs 3 --lr 2e-3 --warmup 200 --schedule cosine --decay-frac 0.35 --topk 1024 --k 32 --width 2458 --lambda-ent 0.01 --daem-t0 2.0 --daem-frac 0.10 --init-c 4.0 --loss-chunk 4096 --log-every 200 --panel-every 1000 --ckpt-every 0 --seed 42 --max-chunks 0（调试截断） --backbone-dtype bfloat16 --resume --data-format babylm|hf --hf-dataset HuggingFaceFW/fineweb-edu --hf-config sample-10BT --hf-split train --hf-field text --hf-max-docs 0 --dev-format babylm|hf --hf-dev-docs 2000 --hf-skip-docs 0 --token-cache`

HF 数据路径：`load_or_pack_tokens_hf` 流式拉取 + 批量分词 + EOS 拼接 + int32 缓存；`--hf-skip-docs N` 跳过流前 N 篇（训练集）；`--dev-format hf` 时 dev = 流的前 `--hf-dev-docs` 篇（缓存 `fineweb_dev_tokens.npy`）——**二者错开无泄漏**。计算节点无外网时：登录节点预下载 HF 数据集，作业内 `HF_HUB_OFFLINE=1`。

## 3. 验收评测：`eval/p3_eval.py`

- **归一化 PPL**：p(t) = exp(mix_logprob(c_t) − logZ)，logZ = 对全部 128256 个合法顶点求 mix_logprob 的 logsumexp（按 `--pos-chunk 64` 分块，[64, V, K] 张量）。native 侧 = 标准 CE（同位置同骨干）。
- 还输出：top-1 一致率、生成样本（贪心 + τ=0.7/1.0 + 教师贪心对照）→ `generations.txt`。
- 关键参数：`--head <ckpt> --k --width --token-cache <dev 缓存路径> --n-chunks 64 --batch 4 --backbone-dtype`。

## 4. 分析与工具脚本

| 脚本 | 作用 |
|---|---|
| `smoke_test.py` | 加载验证：tied 确认、hidden_states[-1]==lm_head 输入、显存画像 |
| `measure_coverage.py` | 教师 top-k 质量覆盖率曲线（k=40..1024）——定 top-k 的依据 |
| `build_codebook.py` | 语义二叉树构建（2-means 方向 + 中位数平衡切分）+ 验证 |
| `measure_code_compactness.py` | 码本操作性验证（目标集紧致度 vs 随机对照） |
| `p1_smoke.py` | 早期过拟合冒烟（自包含，已被 p2_train 取代为新代码源） |
| `analyze_gap.py` | PPL 差距分解（按教师熵分位/真值排名分桶）→ `runs/gap_analysis.json` |
| `benchmark_decode.py` | 推理基准：头部延迟 GPU/CPU、端到端解码 tok/s、带宽/FLOPs 解析表 |
| `gen_samples.py` | 批量生成展示样本（MoB vs 教师并排） |
| `build_fineweb_cache.py` | 集群用 CPU 版 FineWeb-Edu 缓存构建（无 GPU 依赖） |
| `train_encoder.py` | 已终止的编码器实验（负结果，docs/10） |

## 5. 集群作业脚本与运维

- `slurm_setup.sbatch`：一次性搭建（pip 装包 + ModelScope 权重 + FineWeb-Edu 缓存）。**分区用 tiny**（batch 分区有最小资源门槛，小 CPU 作业会被拒）。
- `slurm_train_fwe.sbatch`：1024 步验证跑（已过）；`slurm_train_fwe_long.sbatch`：余弦 2e-3 长训；`slurm_train_fwe_wsd.sbatch`：WSD 7e-4 变体。三者都在 gpu_v100s（V100S-32GB，5 天时限）。
- **V100 必须 `--backbone-dtype float16`**（Volta 无 bf16 tensor core）；本地 4060 用默认 bf16。
- 日常运维（登录节点秒级只读）：`squeue -u yihe47`；`tail runs/<run>/train_log.jsonl`；`scp -P 22 yihe47@burgundy...:<remote> <local>` 拉回产物。

## 6. 已知坑（全是用事故换来的）

1. **PowerShell 写中文文件**：`Set-Content` 会把 UTF-8 中文写坏——用编辑器/write 工具，别用 PS 重定向。
2. **GBK 控制台**：打印含非 ASCII 的文本会 `UnicodeEncodeError`——脚本里 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`，且先写文件后打印。
3. **transformers 5.x 的 `device_map="cuda"` 需要 accelerate**——集群没装；全部改用手动 `.cuda()`。
4. **cross_entropy 的 target 必须 int64**：token chunk 从 numpy int32 来，统一 `.long()`。
5. **torch 生成器分设备**：CUDA 张量的 multinomial 不能用 CPU generator（`kmeans_vertex_init` 的教训）。
6. **后台启动的工具假象**：本地 `Start-Process` 会报 "ChildProcess.kill" 但进程其实起来了——用日志+`Get-Process`+`nvidia-smi` 验证，别重发。
7. **缓冲不可见**：后台进程 stdout 到文件是块缓冲——训练日志以 `train_log.jsonl`（逐条 flush）为准；一次性脚本加 `PYTHONUNBUFFERED=1`。
8. **CPU 时间≠墙钟**：GPU 自旋线程让进程 CPU 时间 ≈1.24× 墙钟，日志记录看起来"迟到"实为正常。
9. **2-means 直接二分在窄锥嵌入上退化**：必须用"方向+中位数平衡切分"（docs/04 §3.1）。
10. **--max-chunks 小于 4×batch 时 warm-up 初始化会切出空批次**：已修（`warm_n` 守卫），改这块代码时注意保持。

## 7. 2026-09-07 大归档增补（解码体系/多模型/新脚本/新坑）

### 8.1 mob_common.py 现状（头族 + 解码族 + 基建）
- **常量**：`H=2048, WIDTH=2458, K=32, M=17`（M 现为**默认**值，全部类/函数支持 `m=` 参数；各脚本从码本读 `CB_M = codes.shape[1]`）。
- **头族**（统一 forward 签名：返回 `(bit_logits [n,k,m], pi_logits [n,k])`）：
  - `MoBHead(k, width, h=2048, depth=3, m=17)`——主线头，3 层 SiLU MLP（`depth` 可变）。
  - `LinearHead(k, h, m)`——纯线性探针头（已判死：-27pt）。
  - `RMSNorm(d)`——RMS 归一。
  - `ResMLPHead(k, width, h, depth=5, m)`——pre-norm RMSNorm + 残差 + 终 norm（三头赛马臂 2）。
  - `SwiGLUHead(k, h, hidden=7300, m, blocks=1)`——h→hidden→h 的 SwiGLU FFN（臂 3；`blocks>1` 残差堆叠）。
- **解码族**：
  - `mob_mix_logprob`——对数混合概率（全 fp32，免 exp 恒等式不变）。
  - `mixture_vertex_logprob(bit_logits, log_pi, codes_all) -> [n,V]`——全合法顶点对数混合概率；n==1 走 [V,m+1]@[m+1,K] GEMM 快路。
  - `mixture_topk(bit_logits, log_pi, codes_all, topn)`——**界剪枝精确 top-k**（L=max_k A ≤ log p ≤ L+log K，幸存者才做精确 logsumexp；与全排名逐位相同）。
  - `decode_greedy(..., m=)`——选分量+逐位取整（旧主线，带 Hamming 回退）。
  - `decode_mixture_argmax(...)`——全顶点混合概率 argmax（主线档）。
  - `decode_rerank_native(..., E, h, topn=32, proposal={"mixture","round","roundflip","adaptive","scoreflip"}, vid_to_tok, adaptive_kwargs)`——级联重排：提议+嵌入表行 gather 精确重打分。**roundflip（v3 默认 aug_topn=32/per_comp_cap=8/α=0.15）**：众数优先+全分量小额翻转对冲，召回 96.5%/重排 96.5%/hit8 100%，零码本 GEMM。
  - **2026-09-19 预算 256 档（docs/27 §2.4-2.5）**：新增 `propose_varflip`（全 K 模式去重+分量方差序——**已证伪**，256 打不过 roundflip-85）与 `propose_scoreflip`（π 质量域发射+方差序翻转格点+**精确混合概率重打分按分截断**——**全胜便宜策略，尖锐位自截断 90-105 候选，新工作推荐默认提议**；函数签名默认未动以保历史可比，decode_rerank_native 用 `proposal="scoreflip"` 显式取用）。bit 级方差序（翻转优先级=翻转位方差和降序=分量内精确似然序）入两新函数。`decode_greedy` 之外还有 `tools/bench_scoreflip.py`（等带宽保真+延迟对打）。
- **ECC 基建（实验冻结）**：`ecc_parity_matrix(m,p,seed)` + `ecc_extend(codes,pmat)`；p2_train `--ecc-bits` 在线扩位。
- **多模型基建**：`--codebook/--dev-cache` 开关（p2_train/p3_eval/p2_train_dual）；`build_codebook.py` 参数化（`--model-dir/--out-npz/--out-meta/--seed/--device`，M=ceil(log2 V) 自动，分片安全 glob 加载）。

### 8.2 脚本清单（新增）
| 脚本 | 作用 |
|---|---|
| `p2_train_dual.py` | 双臂蒸馏共享冻结基座（配对 batch 流；`--arm-set {scale,shallow}`，易扩到 3+ 头赛马） |
| `p4_backbone_probe.py` | 解冻可行性探针（纯 MoB-NLL 训骨干；--mode unfreeze/lora；--freeze-head/--norm-trainable） |
| `p4_joint_train.py` | 集群联合微调（LoRA/全参末 N 层+norm；`--resume-adapter/--schedule/--compute-dtype fp16+GradScaler`） |
| `p4_leakage_check.py` | 未用顶点泄漏测量（logZ=合法顶点质量） |
| `p5_twin_pretrain.py` | 从零双生子预训练（212M Llama-mini ×2；fp32；NaN 回滚规则；`--resume/--skip-steps/--final-frac`） |
| `p7_lr_scan.py` | 本地 LR 舒适度扫描（断点续跑：每单元一结果 JSON，重启跳过；runs/lr_scan/） |
| `p8_eval_adaptive.py` | 自适应提议评估（网格+round32 基线+翻转补丁臂，env 变量调补丁参数） |
| `analyze_p5_twins.py` | 双生子曲线对比（CPU）+出图 |
| `analyze_headscale.py` | 双联臂 vs 原版 WSD 同步位对照 |
| `plot_wsd_curve.py` | WSD vs 余弦训练曲线图 |
| `eval_rerank_gen_ppl.py` | 各解码路径生成文本的原生头 PPL |
| `eval_topk_recall.py` | 语义评估：recall/覆盖率/命中率/例外分析/round 提议/mass32 |

### 8.3 sbatch 清单
`slurm_train_fwe{,_long,_wsd,_cont}.sbatch`（蒸馏系）、`slurm_p2_headscale_dual.sbatch`、`slurm_p2_shallow_dual.sbatch`、`slurm_p4_joint.sbatch`、`slurm_p4_polish.sbatch`、`slurm_p4_fullft.sbatch`、`slurm_p5_twin{,_cont}.sbatch`（$1=native|mob）、`slurm_smollm2_prep.sbatch`（tiny 预备）、`slurm_smollm2_probe.sbatch`。**全部模板必须带 `#SBATCH --exclude=gpu-v100s-06`**。

### 8.4 新坑增补（编号续 §6）
11. **V100 的 bf16 全参训练 NaN**：Volta 无 bf16 硬件走模拟路径，全参反向即炸（LoRA 因适配器≈0 幸存）；集群训练一律 **fp16 主干 + fp32 可训参数 + autocast fp16 + GradScaler**（溢出步自动跳过），还比 bf16 模拟快 ~4×。
12. **bf16 全顶点 logits 并列**：原生 top-2 精确并列率 4.83%，"召回-一致率缺口"全是并列假象（docs/09 §6.1）。
13. **edit 工具吞换行**：oldString 带行尾换行、newString 不带 → 下一行被粘到上行造成语法错误；编辑后必须编译检查。
14. **PowerShell `-c` 内联 Python 引号被吞**：含双引号/f-string 的内联脚本必炸——一律写临时 .py 或用 edit 工具。
15. **`if ($?)` 链在 python 写 stderr 时断**：stderr 进度条会让 `$?` 变 False，后续命令不执行——长链用 `;` 分隔。
16. **集群 .git 丢失修复**：重克隆到 /tmp → 移植 .git → `git checkout -- .`（缓存/权重是 gitignored 不受影响）。
17. **QOS 5-job 上限**：提交报 `QOSMaxSubmitJobPerUserLimit` 就是满了，等槽位。
18. **peft 包装后骨干访问路径**：`model.get_base_model().model`（直接 `model.model` 拿到的是整个 CausalLM）。
19. **`torch.load(weights_only)` 警告**：集群 transformers/torch 新版默认会翻转——老脚本注意兼容。
20. **k-means init 只覆盖 bias 的 bit 部分**：`head.out.bias[:k*CB_M]`（ECC 时扩到 `k*HEAD_M`，扩位码本参与 init）。

### 8.5 2026-09-15 Qwen 线增补（数据管线/大词表/ scratch 纪律）

- **p2_train.py 新增**：`--head-type {mob,resmlp,swiglu}` + `--hidden/--blocks`（单头脚本也能跑赛马架构）；`kmeans_vertex_init` 改为**先按 token 聚合权重到 V 维再 multinomial**（边缘分布精确等价，已数值+经验验证：`tools/test_kmeans_boundary.py`）——原实现对 [位置×topk] 摊平采样，撞 torch multinomial 的 2²⁴ 类别硬上限（batch-16 warmup 32768×1024=33.5M 超限）。
- **Qwen3.5 管线脚本**：`tools/qwen35_download.py`（ModelScope-only、幂等、list-then-download）、`tools/build_multimix_cache.py`（四源 50/30/15/5 文档级交错 + 每源 2M dev + 审计侧车 segments.npy；每源非空断言）、`tools/verify_qwen35_tap.py`（新架构 tap 点验证，蒸馏前置闸）、`tools/chinese_qual.py`（中文涌现定性探针）、`eval/benchmark_race3.py`（三头全量效率基准）、`eval/analyze_race3.py` / `eval/summarize_race3.py`。
- **Qwen3.5 sbatch**：`slurm_qwen35_prep.sbatch`（tiny：下载+混料缓存）、`slurm_qwen35_verify.sbatch`（GPU：tap+18 位码本）、`slurm_qwen35_distill.sbatch`（swiglu K=128 m=18，batch 16 余弦 1e-3 warmup 4000，~244k 步，5 路链式验收）。**产物全落 `/gpfs1/scratch/yihe47/qwen35/`**（home 配额纪律）。

### 8.6 新坑增补（编号续 §7.4）
21. **torch.multinomial 类别数 ≤ 2²⁴**：超过即 RuntimeError。摊平维度设计时算一遍总量。
22. **pyarrow `pf.schema.names` 含嵌套 struct 子字段**——顶层列名要用 `pf.schema_arrow.names`（stack-v3/smoltalk 两连踩）。
23. **stack-v3 是 repo 级行**：`files = list<struct<content, license_type, language...>>`，逐文件过滤须炸开嵌套（调研报告的"文件级行"说法有误，以实测 schema 为准）。
24. **hf-mirror 的 LFS 桥对集群不通**（元数据 API 通、文件下载全挂）——集群下载一律 ModelScope；下前用 `GET /api/v1/datasets/<ns>/<name>/repo/tree?Revision=master&Root=...` 侦察真实文件名，别猜。
25. **batch 尺寸是吞吐杠杆也是配方变量**：batch-8 配方出身本地 8GB 卡，在 V100 上只用 ~28% 显存/~26% 算力（launch-bound）；提 batch 按 √2 缩 LR 且别出实测舒适带。一致性 vs 吞吐的取舍显式记录。
26. **大宗产物一律 scratch**，home 只放代码/文档/小日志；`git pull` 卡 merge 先查未跟踪阻挡文件（删之再拉）。

### 7.5 2026-09-09~18 增补（赛马线 + Qwen 线）

**头族修订**：`SwiGLUHead` 09-09 修复——原版无 pre-norm 且 block0 无残差（高 LR 必炸的设计事故，第一版 LR 扫描的 swiglu 数据因此作废）；现为 transformer FFN 标准形（每块 pre-norm RMSNorm + 残差，终 norm 后进 out），hidden=7300×1 块 = 49.6M 与 resmlp5 51.2M 对齐。

**p2_train_dual.py**：`--schedule {wsd,cosine}` + 每臂独立峰值（ARM_SET 里的 `"peak"` 字段 + `--peaks tag=val,...` CLI 覆盖）+ `race3` 臂集（control/resmlp/swiglu）+ 每臂独立 lr 日志（`lr_<tag>`）+ `--backbone-dtype`。日志 record 不再有全局 `lr` 键。

**p2_train.py**：`--head-type {mob,resmlp,swiglu}` + `--hidden/--blocks`；`kmeans_vertex_init` 改为**权重先按 token 聚合到 V 维再 multinomial**（分布逐点等价，数值+经验双验证 `tools/test_kmeans_boundary.py`）——原实现摊平 [位置×topk] 采样，撞 multinomial 的 2²⁴ 类别硬上限（batch-16 warmup 时 33.5M 超限）。

**多模型 plumbing 普及**：`p3_eval.py` / `eval_topk_recall.py` / `p8_eval_adaptive.py` 全部支持 `--model-dir/--codebook/--dev-cache/--head-type/--depth/--hidden/--blocks`；M 一律从码本读（`CB_M = codes.shape[1]`），头部构建发生在码本加载之后（先建头会拿默认 M=17 炸 shape mismatch）。

**新脚本**：`eval/benchmark_race3.py`（三头效率全套：头延迟 GPU/CPU、解码路径成本、e2e tok/s b1/32/128、解析表）；`eval/analyze_race3.py` + `eval/summarize_race3.py`（曲线形态+汇总表）；`tools/chinese_qual.py`（中文涌现定性探针，三路径并排）；`tools/verify_qwen35_tap.py`（新架构 tap 点验证闸）；`tools/qwen35_download.py`（ModelScope-only、幂等、list-then-download）；`tools/build_multimix_cache.py`（四源 50/30/15/5 文档级交错缓存 + 每源 dev + 非空断言 + segments 审计侧车）；`tools/test_kmeans_boundary.py`。

**新 sbatch**：`slurm_race3{,_smoke}.sbatch`、`slurm_diag_{load,env}.sbatch`（已废，配额案后归档）、`slurm_qwen35_{prep,verify,distill}.sbatch`。**蒸馏 sbatch 的产物路径全部指向 `/gpfs1/scratch/yihe47/qwen35/`**（runs/cache/data/models 全在 scratch，home 只放代码）。

### 7.6 新坑增补（编号续 §7.4）

21. **torch.multinomial 类别数 ≤ 2²⁴**：摊平维数设计时先算总量（batch-16 warmup 的 32768×1024=33.5M 直接炸）。
22. **pyarrow `ParquetFile.schema.names` 含嵌套 struct 子字段名**——拿顶层列名必须 `schema_arrow.names`（stack-v3 的 repo 级嵌套行、smoltalk 的 messages struct 各踩一次）。
23. **hf-mirror 的 LFS/Xet 桥对集群不通**（元数据 API 通、文件下载全报 connection 错）——集群下载一律 ModelScope；且**先 API 侦察文件名再下**（`GET /api/v1/datasets/<ns>/<name>/repo/tree?...`），猜文件名猜炸过两轮。
24. **作业无声死亡 + exit 120 + 无 traceback** = 先查磁盘配额（CPython stdio flush 失败签名）；配额清理优先级：跑完跑的 step 归档 > ~/.cache/pip > conda pkgs 缓存；大宗产物一律 scratch。
25. **集群同步 = git pull**；pull 卡 "untracked files would be overwritten" 是此前中断的 pull 留下的残片——删掉列出的文件再拉（别 git clean，data/ 下有未忽略的贵资产）。
26. **batch 是吞吐杠杆也是配方变量**：batch-8 出身本地 8GB 卡，V100 上只用 ~28% 显存；提 batch 时 LR 按 √2 缩且不出实测舒适带；改配方撤单重挂要在汇报里写明（556232 被计划内 scancel 被误读成崩溃）。
27. **重组后入口一律 `python -m <pkg>.<mod>`**：按路径跑脚本会让包导入炸（slurm_qwen35_distill 初版和本地测试各忘一次）。

### 7.7 2026-09-18~20 增补（CFHead 对打 + 优化周）

**CFHead 侧**：`tools/build_cfhead_cache.py`（忠实移植其 k-means 构建，seed99/iters10，缓存格式互通可互载；嵌入键嵌套兜底 `model.language_model.embed_tokens.weight`）；`eval/cfhead_rt.py`（共享运行时：LUT+gather 粗分（fp32）+ 精确细阶段 GEMV，与仓库 select 逐位一致，并列翻转按仓库 masked_fill/min 语义）；`eval/cfhead_kernels.py`（vendored 其 Triton 粗/细核，`cfhead_rt.coarse` 在有 triton 且 CUDA 时自动走快路）。

**对打 harness 三件套**：`eval/cfhead_eval.py`（教师强制配对：CFHead containment k 扫描 + MoB 密度/提议/级联 + p1 四分桶；bf16 logits 基座 + 逐块 fp32 cast 的 VRAM 节食；进度行 pos/s+ETA+显存；roundflip 默认关）；`eval/cfhead_benchmarks.py`（MGSM/MMLU/GSM8K/ARC-C × original/CFHead 三档/MoB 两臂；containment 记账可关（`--mob-containment off`）；**断点续跑**：original pass 落 `_orig_{task}.pkl`、臂级 JSONL 完整即跳）；`eval/cfhead_longform.py`（6×8192 自轨迹 + 原生 fp32 shadow + 复读分桶）。

**解码策略实验**：`eval/proposal_256_eval.py`（256 预算五臂对打 + 精确集合覆盖率）；`eval/sampling_shape_eval.py`（采样形状：mass(C)/覆盖曲线/worst-casualty/分桶漏失/nucleus 包含率/T=1.0&0.7）；`mob/cpu_proposals.py`（**numba CPU 版 scoreflip/roundflip**，packed uint32 码本 1.3MB，与 torch 参考集合等价 100/100）；`tools/bench_offload.py`（CPU-offload 拆分延迟：GPU 粗筛→KB 传输→CPU 提议+RAM 取行 GEMV）。

### 7.8 新坑增补（编号续 §7.6）

28. **triton-windows 是 Windows 上免 MSVC 的 CUDA 扩展替代**（3.8.0 在 4060/cu130 直接可用，7.8× 于 torch 兜底）——但属非官方 fork，生产慎用，集群走官方 triton。
29. **numba 的 np.argpartition 会段错误**（exit -1073741819 无任何输出）——全排序替代，别在这上面省时间。
30. **benchmark 数据要先验 gold 再开跑**（MGSM parquet answer 列 100% NaN 事故）：smoke 必须打印一条 gold+pred 对，全 False/全 0 先查数据再怪模型。
31. **PowerShell `Add-Content` 按 GBK 写 UTF-8 文件**（docs 直接花掉）——文件写入用 `[IO.File]::AppendAllText($p,$t,(New-Object Text.UTF8Encoding($false)))` 或编辑工具。
32. **edit 工具在长表格行上锚定行首会吃掉前缀**（docs/12 三连犯）——编辑后必须重读该行验证。
33. **combo 位序参考实现是 u-1-j**（combo 高位 ↔ 前序不确定位）——移植枚举算法时对源逐字核对，集合相同≠选择相同。

## 8. 扩展指引

- **换容量**：`--k/--width`（+ 评测脚本同步）；K=128 时 n_eff 只有 ~74-89，提示分量数可能偏多，升 K 前先看 n_eff。
- **换数据**：`--data-format hf --hf-dataset ...`（或写好新的文本目录 loader）；dev 集用 `--dev-format` 对齐分布。
- **换日程**：`--schedule/--lr/--warmup/--decay-frac`；新形状先在本地 `--max-chunks 256 --log-every 1` 干跑看 lr 序列再上集群。
- **换骨干**：改 `mob_common.py` 的 `H`、码本需重建（嵌入表变了）、`--backbone-dtype` 按硬件选。
- **恢复中断**：`--resume <ckpt>`（仅权重）+ 手动对齐 `--epochs/--lr`；集群作业务必带 `--ckpt-every`。
