# P2/B+A 集群提交 Runbook（V100 slurm）

日期：2026-08-31 · 修订 2026-09-01（B+A 方向：FineWeb-Edu 10B + 大解码头）· 状态：待用户提供集群信息后执行

## 1. 我需要你提供的四项信息

| 项 | 用途 |
|---|---|
| slurm 分区名（+account/QoS 如有） | 填 `scripts/p2_train.sbatch` 的 `FILL_ME` |
| 集群 Python 环境怎么搞 | 现成 conda/module 还是新建（依赖清单见 `requirements-cluster.txt`） |
| 上传路径 | 权重 + 数据 + 代码放集群哪里（下文默认 `~/llm_codec`） |
| 谁提交 | 你自己 `sbatch`，或给我 ssh 途径 |

另请确认：**计算节点是否有外网**（决定 FineWeb-Edu 的两种数据获取方式，见 §3）。

## 2. 上传清单（约 2.6 GB）

```
~/llm_codec/
├── LLM-Research--Llama-3.2-1B/snapshots/master/   # 模型权重 2.5GB（model.safetensors + config + tokenizer）
├── data/codebook.npz                              # 码本（含 valid_vertices 掩码）
├── scripts/                                       # mob_common.py, p2_train.py, p3_eval.py, p2_train.sbatch
├── requirements-cluster.txt
└── （token 缓存视 §3 方案而定，见下）
```

## 3. 集群环境与数据集（B+A：FineWeb-Edu 10B 子集）

- `pip install -r requirements-cluster.txt`（含 datasets；**torch 必须匹配集群 CUDA**，别照搬本地 cu130 build）。
- V100（Volta sm_70）无 bf16 tensor core：两个脚本已带 `--backbone-dtype float16`（sbatch 模板已默认）；本地 4060 用 bf16 即可。
- 数据集 = **FineWeb-Edu 10B 子集**（`HuggingFaceFW/fineweb-edu`，config `sample-10BT`；2026-09-01 用户拍板）。两种获取方式：
  - **计算节点有外网**：直接流式，`--data-format hf`，首次训练顺带建 token 缓存；
  - **无外网**：登录节点先 `huggingface-cli download HuggingFaceFW/fineweb-edu --repo-type dataset` 到共享 FS 的 `HF_HOME`，作业里 `export HF_HUB_OFFLINE=1`。
- BabyLM dev 仍作为评测集（跨域 held-out，和之前所有读数可比）。

## 4. 提交

```
mkdir -p logs runs
sbatch scripts/p2_train.sbatch    # 填好 FILL_ME 后
```

作业内容（单卡 V100 32G）：
1. `p2_train.py --data-format hf --k 128 --width 3072`：FineWeb-Edu 10B 子集蒸馏（epoch 数/数据量为提交时决策点），dev 面板每 1000 步，按 epoch 存 checkpoint，日志 `train_log.jsonl`；
2. 接着自动跑 `p3_eval.py`（`--k 128 --width 3072`）：babylm_dev 归一化 PPL（MoB vs 原头）+ 生成样本。

## 5. 拿回结果

把 `runs/` 下对应目录（train_log.jsonl、checkpoint、p3_metrics.json、generations.txt）拉回本地即可，我来做分析报告（落档 docs/08）和验收判定。

## 6. 预期参照（本地实测）

| 配置 | 训练量 | dev top-1 一致率 | MoB/native PPL 比 |
|---|---|---|---|
| K=32, 1M tokens / 256 步 | 0.366 | 3.19× |
| K=32, 4M tokens / 1024 步 | 0.502 | 2.35× |
| K=32, 45M tokens / 3 epochs（P2 最终） | 0.540 | 2.13× |
| K=128, 4.1M tokens / 1024 步 | 0.687 | 未测 |
| B+A 集群跑（K=128 + FineWeb-Edu 10B） | 待测 | 待测 |

健康标准：训练中 NLL 单调下行、死分量 = 0、有效分量数 n_eff 不低于 K 的 ~40%；若曲线平台化或告警爆表，拉回日志我来诊断。

## 8. 集群执行记录（2026-09-01，CityU burgundy）

- 集群：burgundy.hpc.cityu.edu.hk:22（本机公钥已认证）；家目录 `/gpfs1/home/yihe47`，conda cose2 在 `/home/yihe47/miniconda3/envs/cose2`（Python 3.11.15，**torch 2.5.1+cu121 原生支持 V100**）。
- 分区：`gpu_v100s`（6×V100S-32GB，5 天）、`gpu_a100`（6×A100，5 天）、`tiny`（小作业）、`batch`（CPU 大作业，有最小资源门槛——小 CPU 作业会被拒，用 tiny）。
- 规则：**登录节点只跑秒级只读命令**；一切其他操作走 sbatch。
- 搭建作业（tiny，job 519605）已完成：pip 装 transformers 5.16.1/datasets/safetensors/modelscope（**注意：transformers 5.x 的 `device_map` 需要 accelerate——已从所有脚本移除，改 CPU 加载后 `.cuda()`**）；ModelScope 下拉 Llama-3.2-1B；FineWeb-Edu 缓存：train 1M 篇 ≈ **10.0 亿 tokens**（4GB）、dev 2000 篇。
- 验证作业（gpu_v100s，job 519710）已通过：1024 步训练 + 评测全链路 OK；**实测 V100 步速 0.36s/step**（batch 8×512，fp16）→ 1B token 缓存 1 epoch ≈ 245k 步 ≈ **24.5 小时**；评测链路与 `--k 128 --width 3072 --token-cache` 参数组合验证通过。
- 验证跑读数（1024 步欠训练基线）：native NLL 2.54 / PPL 12.70（FineWeb-Edu dev）；MoB NLL 4.60 / PPL 99.8（比值 7.86×）；域内一致率 0.516；n_eff 83.5/128，死分量 0。

## 7. 历史：本地 BabyLM 跑（已完成，2026-08-31/09-01）

全量蒸馏（10M×3 epochs）已在本地完成（含一次自动休眠中断后的热启动恢复，详见 `docs/07-p2-report.md` §2）。
