# Distillation of Compact Mixture Heads: Data, Architectures, and Efficiency

## Abstract

We evaluate semantic-code mixture heads on three frozen language models. Exact
top-1024 distillation learns useful token densities without sampled supervision.
On SmolLM2-1.7B, a pre-normalized SwiGLU head reaches 1.78 times native perplexity,
compared with 2.41 times for a three-layer MLP. Candidate reranking attains 96.5%
agreement on 16,384 held-out positions. CPU and bandwidth measurements show the
clearest efficiency gains; GPU throughput depends on vocabulary size and batch.

## 1. Models and corpora

| Backbone | Hidden size | Vocabulary | Code bits | Distillation data |
|---|---:|---:|---:|---|
| Llama-3.2-1B | 2,048 | 128,256 | 17 | BabyLM preliminary study; FineWeb-Edu main study |
| SmolLM2-1.7B | 2,048 | 49,152 | 16 | FineWeb-Edu |
| Qwen3.5-2B | 2,048 | 248,320 | 18 | English, Chinese, code, instruction mixture |

[BabyLM 2023](https://babylm.github.io/) supplies the preliminary 10M-word training
subset and development text. [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
supplies English educational web text. Text is tokenized with the corresponding
backbone tokenizer, separated by EOS, and packed into 512-token contexts.

The Qwen run uses a nominal 2B-token document-interleaved mixture: 50% FineWeb-Edu,
30% [CCI3-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ),
15% [The Stack v3](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train),
and 5% [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).
Code processing expands repository-level `files` records and filters languages
and permissive license metadata. Each source's development documents are held
out before mixing. The implemented builder reserves 2M tokens per source;
reported comparisons evaluate fixed 16,384-position source slices.

Dataset text and token caches are not redistributed. Source terms remain
applicable; the repository includes builders and numerical results.

## 2. Optimization

Distillation freezes the full backbone and trains only the head. AdamW, clipped
gradients, fp32 log-space loss, entropy regularization $\lambda=0.01$, and
vertex-based initialization are shared controls. V100 forwards use fp16;
the loss remains outside low-precision autocast. Context length is 512 and the
teacher support is 1,024 tokens per position.

| Study | Head settings | Budget | Schedule |
|---|---|---|---|
| Llama main | MLP, width 3072, depth 3, K=128 | approximately 1B tokens, then 50k-update polish | WSD peak 7e-4; polish peak 2e-4 |
| SmolLM2 paired architecture study | MLP / ResMLP / SwiGLU, K=128 | 228k updates, batch 8, shared paired batches | cosine; peaks 1.5e-3 / 1e-3 / 7e-4, warmup 4000 |
| Qwen multilingual | SwiGLU hidden 7300, one block, K=128 | 224k completed updates, batch 16 | cosine peak 1e-3, warmup 4000 |

The SmolLM2 heads share each frozen-backbone forward and the same data order.
Residual and gated heads are approximately parameter-matched; the MLP is smaller.
Per-architecture learning rates differ, so this is an optimized architecture
comparison rather than a depth-only ablation. The Qwen checkpoint corresponds to
224k of 244k scheduled updates (about 92%); all reported Qwen results use that
fixed checkpoint. Completion of its unexecuted schedule is not assumed.

## 3. Density quality

All perplexity ratios compare normalized MoB likelihood with the native head on
the same backbone and evaluation positions.

| Study | Head | Parameters | MoB / native PPL |
|---|---|---:|---:|
| Llama, initial BabyLM experiment | MLP K=32 | 18.5M | 2.13× |
| Llama, FineWeb-Edu polished | MLP K=128 | 32.3M | 2.00× |
| SmolLM2, architecture study | MLP | 31.9M | 2.41× |
| SmolLM2, architecture study | ResMLP | 50.8M | 1.90× |
| SmolLM2, architecture study | SwiGLU | 49.3M | **1.78×** |

SmolLM2's acceptance evaluation uses 32,768 positions. A separate 16,384-position
fidelity slice measures SwiGLU/native perplexity 16.34/9.20, also approximately
1.78×. Exact values differ across slices and should not be combined as one test.
Llama and SmolLM2 ratios describe different prediction tasks and are not direct
cross-model rankings. Qwen source-specific densities are reported in the
[CFHead comparison](03-cfhead-comparison.md).

## 4. Candidate quality

On SmolLM2 FineWeb-Edu development data, 16,384 positions:

| Head | Mixture recall@1 | Recall@32 | Cascade agreement@32 | Native mass in 32 candidates |
|---|---:|---:|---:|---:|
| MLP | 0.779 | 0.945 | 0.932 | 0.821 |
| ResMLP | 0.825 | 0.969 | 0.956 | 0.847 |
| SwiGLU | **0.829** | **0.978** | **0.965** | **0.856** |

SwiGLU's 32 proposals contain at least one native top-8 token on 99.97% of
positions. On a separate 2,048-position proposal study, roundflip reaches 0.973
recall and 0.969 rerank agreement using a mean of about 36 candidates. These
figures measure different proposal budgets and slices.

## 5. Efficiency

| Configuration | Native head | MoB head | Interpretation |
|---|---:|---:|---|
| Llama K=32, bandwidth per step | 501 MiB | 70.7 MiB | 86% less head-weight traffic |
| Llama K=32, CPU fp32 batch 1 | 41.2 ms | 3.2 ms | 12.9× head-only speedup |
| Llama K=128, CPU fp32 batch 1 | 41.2 ms | 5.6 ms | 7.3× head-only speedup |
| SmolLM2 SwiGLU, fp16 head storage | 201 MiB | 98.6 MiB | about 2× reduction |
| SmolLM2 SwiGLU, CPU head batch 1 | 18.6 ms | 8.5 ms | 2.2× head-only speedup |
| SmolLM2 SwiGLU, GPU head batch 1 | 0.80 ms | 0.85 ms | approximate parity |

Llama CPU end-to-end greedy throughput improves by 29–31% in the measured setup.
At high GPU concurrency, the backbone and kernel-launch behavior reduce the
head's contribution. SmolLM2 batch-128 native/MoB-round generation reaches
3467/3336 tokens/s. Head-only speedup is not an end-to-end speedup guarantee.

## 6. Discussion

Distillation produces accurate proposal sets before it matches native density.
Residual normalization and gating improve the optimized head family, while the
remaining density penalty is concentrated in diffuse or multimodal positions.
Mixture calibration, sampling, and free-running generation remain imperfect in
the evaluated artifacts. These are measured limitations of the completed study.
