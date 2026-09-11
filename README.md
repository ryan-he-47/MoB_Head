# MoB_Head: Distilling LLM unembedding into a Mixture-of-Bernoullis head over semantic binary codes

> **Research prototype — minimal implementation closed the loop end-to-end (2026-09-02).** Codebook → distillation → three decode paths → CPU/GPU benchmarks, all measured on real checkpoints. Final large-scale acceptance numbers are landing (see [Status](#status)).

## What is this?

Small LLMs pay a heavy per-token cost at the output layer: every decode step multiplies the hidden state by the full V×h unembedding matrix and softmaxes over the entire vocabulary (V = 128,256 for Llama-3.2-1B — a 501 MiB weight read per generated token). On bandwidth-constrained devices (phones, NPUs, CPUs) this is the bottleneck.

This project replaces the unembedding layer with a compact **Mixture of factorized Bernoullis (MoB)** head over **17-bit binary codes**:

1. **Semantic binary tree codebook** — each vocab token gets a 17-bit code (a vertex of the 17-dim unit hypercube), assigned by a balanced spectral-style binary tree over the model's own embedding table, so nearby codes are semantically close.
2. **MoB head** — a small MLP (3 layers, SiLU) maps the backbone's hidden state to K mixture components of per-bit Bernoulli logits + mixture weights. Trained by **exact weighted-expectation distillation** from the frozen teacher's top-1024 next-token distribution. No sampling, no noise, no V-dim anything.
3. **Decoding** — rank all valid vertices by full mixture probability and take the argmax (mixture-argmax; no h×V matmul, no V-length softmax; exact bound-pruned ranking via `mixture_topk`). A cascade mode (mainline) rescores the MoB top-32 proposals with 32 rows of the tied embedding table (top-1 fidelity 92.2% vs native, ~131 KB/step) — see `docs/09` §6.1.

### Decode fidelity (WSD K=128 final head, 1B-token run; FineWeb-Edu dev)

| Decode path | top-1 agreement vs native | step latency (GPU / CPU) |
|---|---|---|
| component pick + round (early) | 0.770 | 1.76 ms / 5.45 ms |
| mixture-argmax (mainline) | 0.798 | 2.72 ms / 20.3 ms |
| cascade rerank, round proposal (mainline, efficiency mode) | **0.922** | **1.14 ms / 6.42 ms** |
| cascade rerank, mixture proposal (mainline, fidelity mode) | **0.938** | 2.76 ms / 22.4 ms |
| native lm_head + softmax (reference) | 1.000 | 2.43 ms / 41.2 ms |

Candidate-set quality (MoB top-32, 16,384 dev positions): recall of the native top-1 **95.7%**; catches ≥1 of the native top-8 on **99.9%** of positions, ≥1 of the native top-4 on **99.7%**. Of the 4.3% top-1 misses, **98.2%** still land on a native top-8 token. The 32 candidates carry **80.3%** of the native distribution's probability mass. Cascade generations score **PPL 2.27 under the native head** vs native greedy's own 2.35 — statistically indistinguishable (`docs/09` §6).

Honest boundary: the full-distribution gap is narrowed but not closed — normalized PPL of the MoB mixture vs native is **2.05×** on the final WSD head (cosine run: 3.05×), and the training curve was still rising at termination (`docs/09` §6). The cascade closes *decode* fidelity; the density gap is the training-side frontier.

## Why it's interesting

| | native lm_head + softmax | MoB head (K=32) | MoB head (K=128) |
|---|---|---|---|
| Head bandwidth / decode step | 501 MiB | **70.7 MiB (−86%)** | 123 MiB (−75%) |
| Head FLOPs / decode step | 525M | **37M (14× less)** | 65M (8× less) |
| Head latency, CPU fp32 batch 1 | 41.2 ms | **3.2 ms (12.9×)** | 5.6 ms (7.3×) |
| Head latency, GPU batch 256 | 8.5 ms | **2.2 ms (3.8×)** | 3.6 ms (2.4×) |
| + cascade decode step, CPU batch 1 | 41.2 ms | — | **27.6 ms (1.5×)** |
| + cascade decode step, GPU batch 1 | 2.43 ms | — | 2.56 ms (parity) |

The cascade's exact all-vertex ranking is a bound-pruned `[V,18]@[18,K]` GEMM (≈0.59 GFLOP — FLOP-comparable to the native matvec, but it reads only the codebook: 8.7 MB fp32, 0.3 MB bit-packable — vs the 501 MiB table) plus a 131 KB row gather; verified bit-identical to full ranking. End-to-end greedy decode (KV cache): parity to +15% tok/s on an RTX 4060 laptop GPU (run-to-run variance), **+29–31% on CPU fp32** — the win scales with how bandwidth-starved the device is (see `docs/08-inference-benchmark.md`).

The architecture also makes the vocab interface fully symmetric and code-driven: `binary code → (encoder side) → backbone → MoB head → binary code`. (An MLP replacement for the *embedding table* was evaluated and shown infeasible — see `docs/10-encoder-mlp.md`; embedding offloading is the practical answer on the encoder side.)

## Status

- ✅ **Minimal implementation closed the loop end-to-end (2026-09-02)**: semantic-tree codebook → MoB distillation on a frozen backbone (exact weighted-NLL on teacher top-1024) → three decode paths (round / mixture-argmax / cascade rerank, exact bound-pruned) → CPU/GPU benchmarks — every stage verified on real checkpoints.
- ✅ **Large-scale run complete (2026-09-03)**: FineWeb-Edu 1B tokens, K=128 head, WSD schedule — normalized PPL **2.05×**, dev top-1 agreement **0.770** (round decode); cosine-2e-3 baseline: 3.05×/0.679 (note: two-variable comparison — peak LR changed together with schedule shape, so this is not a controlled schedule verdict). Curve still rising at termination — more training is headroom, not a fix (`docs/09` §6).
- ✅ **Decode fidelity at usable level** (final head): cascade top-1 agreement **92.2–93.8%** vs native; candidate set catches a native top-8 token on **99.9%** of positions; misses degrade to the teacher's 2nd–8th choice; cascade generations indistinguishable from native under the native head (PPL 2.27 vs 2.35).
- ✅ **Efficiency verified**: head bandwidth −75…−86%, head FLOPs 8–14× less, CPU head step up to 12.9× faster; efficiency-mode cascade step 1.14 ms GPU / 6.42 ms CPU (6.4× faster than native on CPU); e2e +29–31% tok/s on CPU.
- ❌ Known gaps: full-distribution fidelity (normalized PPL 2.05× — "comparable" not yet met); sampling-quality decoding untested; training not topped out (second epoch / extended decay untried).

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
