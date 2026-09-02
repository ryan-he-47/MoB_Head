# MoB_Head: Distilling LLM unembedding into a Mixture-of-Bernoullis head over semantic binary codes

> **Research prototype — minimal implementation closed the loop end-to-end (2026-09-02).** Codebook → distillation → three decode paths → CPU/GPU benchmarks, all measured on real checkpoints. Final large-scale acceptance numbers are landing (see [Status](#status)).

## What is this?

Small LLMs pay a heavy per-token cost at the output layer: every decode step multiplies the hidden state by the full V×h unembedding matrix and softmaxes over the entire vocabulary (V = 128,256 for Llama-3.2-1B — a 501 MiB weight read per generated token). On bandwidth-constrained devices (phones, NPUs, CPUs) this is the bottleneck.

This project replaces the unembedding layer with a compact **Mixture of factorized Bernoullis (MoB)** head over **17-bit binary codes**:

1. **Semantic binary tree codebook** — each vocab token gets a 17-bit code (a vertex of the 17-dim unit hypercube), assigned by a balanced spectral-style binary tree over the model's own embedding table, so nearby codes are semantically close.
2. **MoB head** — a small MLP (3 layers, SiLU) maps the backbone's hidden state to K mixture components of per-bit Bernoulli logits + mixture weights. Trained by **exact weighted-expectation distillation** from the frozen teacher's top-1024 next-token distribution. No sampling, no noise, no V-dim anything.
3. **Decoding** — rank all valid vertices by full mixture probability and take the argmax (mixture-argmax; no h×V matmul, no V-length softmax; exact bound-pruned ranking via `mixture_topk`). A cascade mode (mainline) rescores the MoB top-32 proposals with 32 rows of the tied embedding table (top-1 fidelity 92.2% vs native, ~131 KB/step) — see `docs/09` §6.1.

### Decode fidelity (WSD K=128 head @ step 100k — pre-decay checkpoint; FineWeb-Edu dev, 16,384 positions)

| Decode path | top-1 agreement vs native | step latency (GPU / CPU) |
|---|---|---|
| component pick + round (early) | 0.717 | 1.76 ms / 5.45 ms |
| mixture-argmax (mainline) | 0.745 | 2.72 ms / 20.3 ms |
| cascade rerank, mixture proposal (mainline, fidelity mode) | **0.922** | 2.76 ms / 22.4 ms |
| cascade rerank, round proposal (mainline, efficiency mode) | **0.899** | **1.14 ms / 6.42 ms** |
| native lm_head + softmax (reference) | 1.000 | 2.43 ms / 41.2 ms |

Candidate-set quality (MoB top-32): catches ≥1 of the native top-8 on **99.7%** of positions, ≥1 of the native top-4 on **99.2%**. On the 6.1% of positions where the native top-1 escapes the set, **95.1%** still contain a native top-8 token — the reranker lands on the teacher's 2nd–8th choice, not on garbage. The 32 candidates carry **77.9%** of the native distribution's probability mass on average. Generations decoded with the cascade score **PPL 2.02 under the native head** vs native greedy's own 2.35 — statistically indistinguishable fluency (`docs/09` §6.1).

Honest boundary: the full-distribution gap remains — normalized PPL of the MoB mixture vs native is 2.585× on this pre-decay checkpoint (the cosine run finished at 3.05×). The cascade closes *decode* fidelity; the density gap is the training-side frontier and narrows through the in-flight LR-decay phase.

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
- ✅ **Decode fidelity at usable level** (K=128 WSD head, pre-decay checkpoint): cascade top-1 agreement **92.2%** vs native; candidate set catches a native top-8 token on **99.7%** of positions; misses degrade to the teacher's 2nd–8th choice.
- ✅ **Efficiency verified**: head bandwidth −75…−86%, head FLOPs 8–14× less, CPU head step up to 12.9× faster; cascade decode step 1.5× faster than native on CPU, parity on GPU; e2e +29–31% tok/s on CPU.
- ✅ BabyLM-10M full distillation (K=32): normalized PPL 2.13×; cosine 1B-token run (K=128): PPL 3.05×, agreement 0.679 (`docs/09`).
- 🔄 **In flight**: FineWeb-Edu 1B-token WSD-schedule run (ahead of the cosine baseline at matched steps) — final acceptance on completion; the fidelity numbers above are from a pre-decay checkpoint and should improve.
- ❌ Known gaps: full-distribution fidelity (normalized PPL 2.6–3.0×); sampling-quality decoding untested.

## Repository layout

```
scripts/            # all code (train/eval/analysis/cluster jobs) — see docs/13
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

python scripts/smoke_test.py                     # sanity: loads, tied embeddings, lm_head input point
python scripts/build_codebook.py                 # build the semantic binary-tree codebook -> data/codebook.npz
python scripts/p2_train.py --epochs 3 --batch 8  # distill the MoB head (BabyLM-10M default)
python scripts/p3_eval.py --head runs/p2_10m/head_final.pt   # normalized PPL vs native + generations
python scripts/benchmark_decode.py               # inference speed/bandwidth benchmark
```

## License & citation

TBD (pre-release research code). Llama 3.2 weights remain governed by the Llama 3.2 Community License.
