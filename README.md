# MoB_Head: Distilling LLM unembedding into a Mixture-of-Bernoullis head over semantic binary codes

> **Race3 milestone (2026-09-13): best head = SwiGLU-MoB — normalized PPL 1.78× vs native on SmolLM2-1.7B, cascade top-1 fidelity 96.5%, candidate-set hit@top8 100%.** Three-architecture race + full fidelity/efficiency re-benchmark complete (see [Status](#status) and `docs/20`).

## What is this?

Small LLMs pay a heavy per-token cost at the output layer: every decode step multiplies the hidden state by the full V×h unembedding matrix and softmaxes over the entire vocabulary (V = 128,256 for Llama-3.2-1B — a 501 MiB weight read per generated token). On bandwidth-constrained devices (phones, NPUs, CPUs) this is the bottleneck.

This project replaces the unembedding layer with a compact **Mixture of factorized Bernoullis (MoB)** head over **17-bit binary codes**:

1. **Semantic binary tree codebook** — each vocab token gets a 17-bit code (a vertex of the 17-dim unit hypercube), assigned by a balanced spectral-style binary tree over the model's own embedding table, so nearby codes are semantically close.
2. **MoB head** — a small network maps the backbone's hidden state to K mixture components of per-bit Bernoulli logits + mixture weights. **Architecture matters**: a three-way race (2026-09-13, SmolLM2-1.7B, per-arm cosine LR schedules, 228k steps) put a pre-norm **SwiGLU head** (1.78× normalized PPL) and a pre-norm **5-layer residual head** (1.90×) far ahead of the original 3-layer MLP (2.41×). Trained by **exact weighted-expectation distillation** from the frozen teacher's top-1024 next-token distribution. No sampling, no noise, no V-dim anything.
3. **Decoding** — rank all valid vertices by full mixture probability and take the argmax (mixture-argmax; no h×V matmul, no V-length softmax; exact bound-pruned ranking via `mixture_topk`). A cascade mode (mainline) rescores MoB proposals with a few rows of the tied embedding table. Proposal flavors: `round` (top-32 component modes, zero GEMM, fastest), `roundflip` (modes + uncertain-bit flip hedge, ~36 candidates, recall 97.3%), `adaptive` (dynamic component scope, recall ceiling 98.1%) — see `docs/09` §6.1-6.2 and `docs/20` §3.

### Decode fidelity (SwiGLU-MoB head, SmolLM2-1.7B; FineWeb-Edu dev, 16,384 positions)

| Decode path | top-1 agreement vs native | notes |
|---|---|---|
| component pick + round (early) | 0.805 | cheapest |
| mixture-argmax (recall@1) | 0.829 | full-vertex ranking GEMM |
| cascade rerank, round proposal | 0.951 | zero-GEMM proposal |
| cascade rerank, roundflip proposal | **0.969** | ~36 candidates, no GEMM |
| cascade rerank, mixture proposal | 0.965 | exact ranking |
| native lm_head + softmax (reference) | 1.000 | — |

Candidate-set quality: MoB top-32 recalls the native top-1 on **97.8%** of positions; catches ≥1 of the native top-8 on **99.97%**; the 32 candidates carry **85.6%** of the native distribution's probability mass. Misses degrade gracefully to the teacher's 2nd–8th choice. Full-distribution density: normalized PPL **1.78×** vs native (was 2.05× on the Llama line's best head — different vocab size, not directly comparable).

## Why it's interesting

The head-level win scales with vocabulary size. On **Llama-3.2-1B (V=128,256)** with the 32M-param head:

| | native lm_head + softmax | MoB head (K=32) | MoB head (K=128) |
|---|---|---|---|
| Head bandwidth / decode step | 501 MiB | **70.7 MiB (−86%)** | 123 MiB (−75%) |
| Head FLOPs / decode step | 525M | **37M (14× less)** | 65M (8× less) |
| Head latency, CPU fp32 batch 1 | 41.2 ms | **3.2 ms (12.9×)** | 5.6 ms (7.3×) |
| Head latency, GPU batch 256 | 8.5 ms | **2.2 ms (3.8×)** | 3.6 ms (2.4×) |
| + cascade decode step, CPU batch 1 | 41.2 ms | — | **27.6 ms (1.5×)** |
| + cascade decode step, GPU batch 1 | 2.43 ms | — | 2.56 ms (parity) |

On **SmolLM2-1.7B (V=49,152)** with the 49.3M SwiGLU head (race3, RTX 4060): head bandwidth 201→98.6 MiB fp16 (2.0×↓), CPU head step 18.6→8.5 ms (**2.2×**), GPU batch-1 head latency 0.80→0.85 ms (parity — kernel-launch-bound), e2e greedy generation parity at batch 1/32/128 (native 3467 vs MoB-round 3336 tok/s at b128). At high concurrency or on GPU the backbone dominates and the head share is small; the bandwidth/latency win concentrates where it hurts: CPU, NPUs, bandwidth-starved edge devices, and large-vocab models.

End-to-end greedy decode on Llama (KV cache): parity to +15% tok/s on an RTX 4060 laptop GPU (run-to-run variance), **+29–31% on CPU fp32** (see `docs/08-inference-benchmark.md`; race3 numbers in `docs/20`).

The architecture also makes the vocab interface fully symmetric and code-driven: `binary code → (encoder side) → backbone → MoB head → binary code`. (An MLP replacement for the *embedding table* was evaluated and shown infeasible — see `docs/10-encoder-mlp.md`; embedding offloading is the practical answer on the encoder side.)

## Status

- ⚖️ **CFHead head-to-head complete, inference-acceleration arm mothballed (2026-09-20)**: three-protocol + budget/bandwidth-matched comparison against CFHead (IEEE TC 2026, training-free coarse-fine head) on Qwen3.5-2B — CFHead-T4 wins top-1 fidelity (1-2pt at k=256) and candidate-set distribution quality (mass +2-8pt), MoB wins at tight budget (k=32, EN/ZH +2.5-3.1pt) and uniquely provides a normalized full-vocab density + zero-original-table operation; offload-split latency tied (~4.4 ms/step). Formal closing report: **`docs/28`**; measurements: `docs/27`, `docs/25`. Next phase: end-to-end distillation.
- ✅ **Three-head race complete (2026-09-13)**: pre-norm SwiGLU (49.3M) and pre-norm ResMLP-5 (50.8M) heads decisively beat the original 3L-MLP (32M) under per-arm cosine schedules — normalized PPL **1.78×** / 1.90× / 2.41× on SmolLM2-1.7B (FineWeb-Edu dev). Per-architecture LR comfort zones measured locally first (`docs/16` §5).
- ✅ **Decode fidelity re-benchmarked on the race heads**: cascade top-1 agreement up to **96.9%** (roundflip proposal, ~36 candidates, zero ranking GEMM); candidate set catches a native top-8 token on **99.97%** of positions; roundflip Pareto-dominates at low budget, adaptive-scope proposals win on recall ceiling (98.1%) at 6× budget (`docs/20` §3).
- ✅ **Efficiency**: head bandwidth −50%~86% and CPU head latency 2.2-12.9× faster depending on vocab size; e2e parity on GPU at high concurrency. Round-proposal cascade decode step 0.25 ms GPU.
- ✅ **From-scratch viability**: MoB head pretrains from scratch (212M twin, matched 0.49B-token budget): final PPL ratio 1.89× vs native twin, monotone-closing trajectory, perfect training health (`docs/15`).
- ❌ Known gaps: full-distribution fidelity not yet at parity (1.78×); sampling-quality decoding untested; heads above 50M / longer training untested; the multi-layer heads lose to the native matvec on GPU at batch ≥256 (launch-bound; the win is bandwidth/CPU/edge, not datacenter GEMM).

## Repository layout

```
mob/                # model source: heads (MoB/Linear/ResMLP/SwiGLU), losses, all decode paths
train/              # training entry points (distill single/dual, joint-FT, twin pretrain, LR scan)
eval/               # acceptance eval, recall/rerank evals, benchmarks, analysis scripts
tools/              # codebook builders, token-cache builders, smoke tests
slurm/              # all cluster job templates (.sbatch)
data/               # codebook.npz, token caches, BabyLM 2023
docs/               # full documentation (design, logs, reports) — see below
runs/               # checkpoints, train logs, eval metrics
AGENTS.md           # project conventions and current state (for AI agents)
```

## Documentation

| Doc | Content |
|---|---|
| `docs/11-design-methodology.md` | **Design principles & methodology (start here)** |
| `docs/13-code-architecture.md` | **Code architecture & operational details (for contributors/agents)** |
| `docs/12-experiment-log.md` | Chronological experiment log with key numbers |
| `docs/01-…02-…` | Literature surveys (GMM stability; output-layer prior art) |
| `docs/03-…07-…` | Experiment plan, P0/P1/P2 phase reports |
| `docs/08-…` | Inference benchmark |
| `docs/10-…` | Encoder-side experiment (negative result) |

## Quickstart (local, RTX-4060-class GPU)

```bash
pip install torch transformers numpy safetensors datasets
# backbone: Llama-3.2-1B at ./LLM-Research--Llama-3.2-1B/snapshots/master
# (HF gated; any mirror works, e.g. ModelScope LLM-Research/Llama-3.2-1B)
# all entry points run as modules from the repo root:

python -m tools.smoke_test                     # sanity: loads, tied embeddings, lm_head input point
python -m tools.build_codebook                 # build the semantic binary-tree codebook -> data/codebook.npz
python -m train.p2_train --epochs 3 --batch 8  # distill the MoB head (BabyLM-10M default)
python -m eval.p3_eval --head runs/p2_10m/head_final.pt   # normalized PPL vs native + generations
python -m eval.benchmark_decode                # inference speed/bandwidth benchmark
```

## License & citation

TBD (pre-release research code). Llama 3.2 weights remain governed by the Llama 3.2 Community License.
