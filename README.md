# MoB_Head: Distilling LLM unembedding into a Mixture-of-Bernoullis head over semantic binary codes

> **Work in progress — research prototype.** Numbers below are verified intermediate results; the large-scale run is in flight (see [Status](#status)).

## What is this?

Small LLMs pay a heavy per-token cost at the output layer: every decode step multiplies the hidden state by the full V×h unembedding matrix and softmaxes over the entire vocabulary (V = 128,256 for Llama-3.2-1B — a 501 MiB weight read per generated token). On bandwidth-constrained devices (phones, NPUs, CPUs) this is the bottleneck.

This project replaces the unembedding layer with a compact **Mixture of factorized Bernoullis (MoB)** head over **17-bit binary codes**:

1. **Semantic binary tree codebook** — each vocab token gets a 17-bit code (a vertex of the 17-dim unit hypercube), assigned by a balanced spectral-style binary tree over the model's own embedding table, so nearby codes are semantically close.
2. **MoB head** — a small MLP (3 layers, SiLU) maps the backbone's hidden state to K mixture components of per-bit Bernoulli logits + mixture weights. Trained by **exact weighted-expectation distillation** from the frozen teacher's top-1024 next-token distribution. No sampling, no noise, no V-dim anything.
3. **Decoding** — pick a component by its weight (temperature), round its per-bit probabilities to a vertex, map to a token id. The h×V matmul and V-softmax never happen.

## Why it's interesting

| | native lm_head + softmax | MoB head (K=32) | MoB head (K=128) |
|---|---|---|---|
| Head bandwidth / decode step | 501 MiB | **70.7 MiB (−86%)** | 123 MiB (−75%) |
| Head FLOPs / decode step | 525M | **37M (14× less)** | 65M (8× less) |
| Head latency on CPU (batch 1) | 42.0 ms | **3.7 ms (11.3× faster)** | — |
| Head latency on GPU (batch 256) | 7.4 ms | **1.9 ms (3.8× faster)** | 3.1 ms (2.4×) |

End-to-end decode on an RTX 4060 laptop: **+11% tok/s** — the win scales with how bandwidth-starved the device is. (Measured on a distilled BabyLM-scale prototype head; see `docs/08-inference-benchmark.md`.)

The architecture also makes the vocab interface fully symmetric and code-driven: `binary code → (encoder side) → backbone → MoB head → binary code`. (An MLP replacement for the *embedding table* was evaluated and shown infeasible — see `docs/10-encoder-mlp.md`; embedding offloading is the practical answer on the encoder side.)

## Status

- ✅ Codebook built & validated; full BabyLM-10M distillation (K=32 head): normalized PPL 2.13× vs native, top-1 agreement 0.540; grammatical greedy generations.
- ✅ Gap analysis: the residual gap concentrates in diffuse multi-modal positions (median per-position gap ≈ 0) → capacity is the binding constraint; K=128 head verified stable and much stronger at equal steps.
- 🔄 **In flight**: large-scale distillation on FineWeb-Edu (1B tokens, K=128 head) on an HPC V100, including a schedule-variant (WSD, lower peak LR) head-to-head — see `docs/12-experiment-log.md`.
- ❌ Not yet: final acceptance metrics on the large-scale run; sampling-quality decoding.

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
