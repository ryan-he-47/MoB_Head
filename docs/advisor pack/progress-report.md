# Progress Report: MoB Head — A Coded Output Representation for LLM Decoding

**Date:** 18 September 2026
**Companion document:** Literature Survey Report (same folder)

---

## 1. Status Summary

The project investigates whether an LLM's unembedding layer (the h×V projection plus V-way softmax) can be replaced by a compact head that predicts a short binary code for the next token. Prototype validation and the initial hyperparameter/architecture search are now complete. Work has shifted to two tracks: (i) **scaling validation** — from-scratch pretraining tests and longer/larger distillation runs to establish how the approach behaves with scale; and (ii) **evaluation alignment** — porting the evaluation pipeline to the protocols and harnesses used by related work, so that upcoming comparisons are like-for-like. A reproduction of a concurrent method (CFHead) is underway as part of track (ii) and is reported in §5.

Two concurrent works published in 2026 (HiDRA and CFHead) partially overlap elements of the original framing. As a consequence, the definition of the project's main contribution, and to some extent its direction, are being adjusted — away from the now-crowded "faster LM head" framing and toward the axes these works do not cover: a replacement head carrying a normalized output distribution usable without the original head, native-training viability, and the semantic code layout as a structural interface. Section 4 gives the details.

## 2. Architecture in Brief

Vocabulary tokens are assigned binary codes of length M = ⌈log2 V⌉ by a balanced binary tree built over the embedding table, so that tokens with similar embeddings share long code prefixes (vertex addresses of a hypercube). The replacement head is a small MLP that outputs a mixture of K factorized Bernoulli distributions over the M bits — i.e., an explicit, normalized probability for every vocabulary token — trained on a frozen backbone by teacher-probability-weighted distillation over the teacher's top-1024 distribution. Decoding picks a mixture component and rounds its bit probabilities to a vertex, optionally followed by exact rescoring of a small candidate set against the original head when that head is available.

## 3. Representative Results (measured; Llama-3.2-1B backbone unless noted)

**Distillation fidelity.** After 1B-token distillation (FineWeb-Edu), the head reaches a normalized perplexity of 2.00× the native head (26.2 vs 13.1) with 77.4% next-token agreement. The training curve had not saturated at completion.

**Decode-path fidelity.** When the original head is available for rescoring, 32-candidate reranking reaches 93.8% agreement with the native argmax (95.7% candidate recall; 99.9% of positions contain a native top-8 token), and text generated through this path scores 2.02 perplexity under the native head versus 2.35 for native greedy generation.

**Efficiency.** Head memory traffic per decoding step drops from 501 MiB to 71–123 MiB (−75–86% depending on configuration); the head-only step is ~11× faster on CPU, and end-to-end CPU decoding improves by 29–31%.

**Native-training viability (scaling track).** In a controlled twin pretraining test (212M-parameter models, identical data and steps, one with the coded head and one with the standard head), the coded-head model closed its perplexity ratio monotonically from 2.81× to **1.89×** over 0.49B tokens, meeting the predefined ≤2× viability criterion, with stable training throughout. Floor-probing continuation runs are in progress.

**Architecture search outcome.** A three-head controlled race (same backbone, data, and budget) showed head structure is a significant variable: a gated (SwiGLU) head of ~49M parameters reached 1.78× normalized perplexity versus 2.41× for the plain-MLP control of ~32M parameters (SmolLM2-1.7B backbone; cross-model numbers are not directly comparable to the Llama figures above).

## 4. Impact of Concurrent Works on Positioning

Two works appeared during this project that overlap with elements of the original design space, and this has materially changed how the contribution is defined:

- **HiDRA (ICLR 2026 TTU workshop)** constructs a hierarchical binary decision tree over the vocabulary with an embedding-space splitting procedure close in spirit to the project's codebook tree, and routes hidden states through it at decode time. It has no public code.
- **CFHead (IEEE Trans. Computers 2026)** implements coarse candidate screening over the full vocabulary followed by exact rescoring with the original head — the same system shape as the project's cascade decode path — in a training-free form, with strong reported accuracy preservation and speedups. (FlashHead, known from the project's early survey phase, remains the standing baseline for the training-free retrieval approach and did not prompt this adjustment.)

Both works rely on the original head being present at inference. Neither provides a normalized output distribution that can serve as a standalone head, and neither addresses training a model natively with the new head. The cascade decode path is accordingly being repositioned from a headline contribution to a compatibility mode, and the main line of work is being consolidated around the replacement representation and its training-side properties.

## 5. Reproduction of CFHead (Ongoing)

To establish a fair baseline and to import the community's evaluation practices, I am reproducing CFHead from the authors' public repository (github.com/HigherIdeal/CFHead) on a local RTX 4060 Laptop with Llama-3.2-1B-Instruct. Status as of this report:

| Paper component | Reproduction status |
|---|---|
| Offline construction, memory & compression figures | **Reproduced exactly** (e.g., 7.84 MiB auxiliary memory; 63.9× storage reduction; all 16 compression-ratio configurations match the paper value-for-value) |
| Candidate-coverage (hit-ratio) curves over top-k and tile/cluster sweeps | **Closely matched** (differences within ~1–4 decoding steps of 256; trends identical) |
| Downstream accuracy on ARC / HellaSwag / GSM8K with baselines | **In progress** |
| Speedup tables (LM-head-only and end-to-end) | **Not reproducible on this machine** — the released kernels require a build toolchain (ninja/MSVC) and Triton that are unavailable in the local Windows environment; the GPU platform also differs from the paper's. Local reproduction is therefore restricted to correctness, coverage, and memory metrics. |

The reproduction workspace is at `E:\CFHead\CFHead`, with a running technical report at `repro\REPRO_REPORT.md`. The benchmark datasets used by the paper (TruthfulQA, PIQA, WinoGrande, MGSM, HumanEval+, ShareGPT/LMSYS-Chat-1M samples, LongBench) have been fetched and verified (23/23 files) in preparation for the accuracy phase. HiDRA has no public code; comparison with it is analytical, based on the published description.

## 6. Current Focus

Two activities dominate the immediate effort: completing the CFHead accuracy comparison and integrating its protocols (datasets, candidate-coverage metrics) into the evaluation pipeline; and continuing the scaling-validation runs (twin-pretraining floor probing and larger-vocabulary distillation) to establish where the coded head's trade-off curve settles. Literature monitoring continues on a monthly cadence, as this subfield is currently moving quickly.
