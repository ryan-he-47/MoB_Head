# Literature Survey Report: Efficient Output Layers and Vocabulary-Coded Language Model Heads

**Project:** MoB Head — distilling the LLM unembedding layer into a compact mixture-of-factorized-Bernoullis head over semantic binary codes
**Date:** 18 September 2026
**Companion document:** Progress Report (same folder)

---

## 1. Scope and Method

This report summarizes a three-round literature survey on the two research questions central to this project: (i) efficient decoding/output-layer design for large language models (eliminating or reducing the h×V projection and the V-way softmax at decode time), and (ii) alternative output representations for the vocabulary, in particular binary-code-based heads. All arXiv identifiers and venues below were verified against arXiv / IEEE Xplore / ACL Anthology / OpenReview. The survey is organized around two recent survey papers and nine selected papers, grouped by the reason for their selection.

## 2. Field Overview: Five Years of Development

The two survey papers frame the efficiency landscape within which this project sits.

**Efficient Large Language Models: A Survey** (Wan et al., arXiv:2312.03863, TMLR 2024) provides the model-level taxonomy — quantization, pruning, distillation, low-rank factorization, sparse/MoE architectures — and documents that architecture-level efficiency research has concentrated almost exclusively on the transformer trunk; the output projection has been treated as a fixed component.

**A Survey on Efficient Inference for Large Language Models** (Zhou et al., arXiv:2404.14294, 2024) organizes inference optimization into data-level, model-level, and system-level tiers, with comparative experiments. Its model-level treatment of the output layer still reflects the pre-2025 state (hierarchical/adaptive softmax and sampling approximations), which usefully marks how recently the output layer has re-emerged as an active design surface.

Reading the selected papers along this frame, five years of development resolve into three threads:

- **2021–2022 (context).** Efficient-LLM research is dominated by post-hoc compression of trained models. Concurrently, the community establishes that autoregressive decoding is memory-bandwidth-bound, and speculative decoding (Leviathan et al., ICML 2023; Chen et al., ICLR 2023) opens the parallel "reduce the number of sequential steps" line — orthogonal to, and composable with, per-step output-layer cost.
- **Roots (2017–2019, re-appreciated in retrospect).** Two largely dormant lines already contained the core ingredients of the current wave: Yang et al.'s *Breaking the Softmax Bottleneck* (ICLR 2018) shows a *mixture* of output distributions is needed for expressiveness; Oda et al. (ACL 2017) and O'Neill & Bollegala (2019) predict vocabulary items via *binary codes* with per-bit Bernoulli heads. These works demonstrated feasibility but reported fragility (redundant codes or error-correcting codes were needed to approach softmax quality), and the ideas were not pursued for modern LLM decoding.
- **2023 (statistical form).** Shou et al.'s TCMBN (AAAI 2023) adopts a *mixture of factorized Bernoullis* as a predictive head — the same statistical family used in this project — in the multi-label classification domain, without vocabulary coding, distillation, or decoding considerations.
- **2025–2026 (convergence on the output layer).** Three threads converge on the LM head: (a) candidate proposal + rescoring as a general architecture pattern, now crossing model boundaries (S2T, 2026); (b) binary/hypercube vocabulary representations for generation (BitLM, 2026); (c) direct LM-head replacement — FlashHead (2026) as the training-free retrieval approach known to this project from its early survey phase and used since as the reference baseline for the acceleration axis, followed during the project's advancement by CFHead (training-free coarse-fine screening with exact rescoring, IEEE TC 2026) and HiDRA (learned hierarchical routing, ICLR 2026 TTU workshop). The pace of this last thread is itself evidence that the output layer is now regarded as a designable component rather than a fixed softmax.

The remainder of this report groups and briefly introduces the nine selected papers (the two surveys above are the framing references throughout).

## 3. Selected Papers

### Group A — Conceptual lineage: mixture heads and binary-code vocabularies

These four works supply the individual ingredients of the current design; each is cited as prior art for one specific element.

**[A1] Oda et al., "Neural Machine Translation via Binary Code Prediction" (ACL 2017, arXiv:1704.06918).**
Maps each target word to a B-bit binary code and predicts each bit with an independent sigmoid, optionally mixed with a standard softmax; decoding uses convolutional ECC with Viterbi. This is the closest historical predecessor of a binary-coded output layer, and its central finding — that raw per-bit prediction is fragile and needs error-correcting redundancy to approach softmax quality — is the direct motivation for this project's use of a *mixture* over codes rather than a single factorized Bernoulli. Differences are substantial: identity-ordered (non-semantic) codes, no frozen-backbone distillation, no normalized density objective, and NMT-scale models.

**[A2] O'Neill & Bollegala, "Error-Correcting Output Coding with Neural Signal Processing for Efficient Neural Network Inference" (2019, arXiv:1901.07002).**
Applies error-correcting output codes to language modeling: per-bit Bernoulli predictors over redundant codewords (tens to thousands of bits), with lightweight signal processing at decode; reports roughly 1/20 of the output parameters at close-to-full-softmax perplexity. Relevance: demonstrates that bit-wise prediction can carry LM output at large compression, and quantifies the redundancy required — informing the project-side choice of a minimal-length (⌈log2 V⌉) code combined with a probabilistic mixture instead of algebraic redundancy.

**[A3] Yang et al., "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model" (ICLR 2018, arXiv:1711.03953).**
Introduces Mixture of Softmaxes (MoS) and proves that the standard softmax-with-dot-product logits has bounded rank, limiting expressiveness; a convex combination of K softmax components yields better language-modeling quality at K× the cost. Relevance: MoS is the conceptual ancestor of the mixture head in this project; the design replaces each component's V-dimensional softmax with a factorized Bernoulli distribution over a short semantic binary code, turning the mixture from a quality-enhancing but cost-increasing device into a compact output representation.

**[A4] Shou et al., "Transformer-Based Conditional Mixture Bernoulli Network for Multi-label Classification" (AAAI 2023, arXiv:2301.05868).**
Proposes exactly the statistical form used here — gating weights from a softmax over an MLP, per-bit probabilities from sigmoid outputs, and p(y) = Σ_k π_k ∏_l Ber(y_l; μ_kl) trained by NLL — for multi-label classification over event streams. The architecture is mathematically isomorphic to the head in this project; the domains do not overlap (no vocabulary coding, no teacher-weighted distillation, no decoding strategy), so the work is cited as the statistical-family precedent.

### Group B — LM-head replacement works: known baseline and concurrent arrivals

FlashHead has been known to this project since its early survey phase and serves as the standing strong baseline for the acceleration axis. CFHead and HiDRA appeared during the project's advancement and overlap two specific elements of the design space; they define the current competitive landscape and motivated the repositioning described in the Progress Report.

**[B1] FlashHead, "Efficient Drop-in Replacement for the Classification Head in Language Model Inference" (arXiv:2603.14591, 2026).**
Training-free drop-in replacement: balanced clustering of LM-head rows with multiprobe retrieval, inference-time sampling, and selective quantization; reports ~1.75× end-to-end speedup on consumer GPUs with no reported benchmark loss. Role in this project: the reference training-free head accelerator, used as the primary comparison baseline for the acceleration axis from the early survey stage onward.

**[B2] CFHead, "Fast LM Head via Coarse-Fine Decomposition and Tile-Wise Weight Clustering" (IEEE Transactions on Computers 2026, DOI 10.1109/TC.2026.3725429).**
Reformulates the LM head as candidate screening: offline tile-wise k-means vector quantization of the head weights yields a small codebook plus an index map; at decode time, approximate coarse scores are computed for every vocabulary token by table lookup and accumulation, top-k candidates are selected at token level, and exact logits are computed for those candidates with the original weights. Training-free; reports −0.21%p average over nine downstream benchmarks and 5.82× average LM-head speedup across 23 models. Relevance: its coarse-propose / exact-rerank system shape overlaps the cascade decode path of this project; a reproduction of this work is currently underway (see the Progress Report) to establish a like-for-like baseline.

**[B3] HiDRA, "A Blazing-Fast LM-Head Replacement" (ICLR 2026 Test-Time Updates Workshop, Kumar & Ndubuaku, Cactus Compute).**
Replaces the dense head with a binary decision tree over the vocabulary (depth ⌈log2 V⌉): per-node hyperplane decisions initialized by Fisher LDA and refined by binary cross-entropy, with beam routing at inference; the original LM head then scores only the tokens of the reached subtrees. Reports 8.04× output-stage CPU speedup at a −4.2pt top-1 change on Gemma 3 270M. Relevance: its tree-over-embedding construction is methodologically close to the semantic binary-tree codebook used in this project, and its routing is deterministic (a hard path through the tree) rather than probabilistic; the work currently has no public code. Together with CFHead, its appearance motivated the adjustment of the project's main contribution (see Progress Report §4).

### Group C — Adjacent paradigms informing positioning

**[C1] "Select to Think" (S2T) (arXiv:2604.26940, 2026).**
Formalizes *local sufficiency*: at divergence points, a large model's preferred token is usually already inside a small model's top-K candidates; reframes LLM assistance from open-ended generation to *selection over the small model's proposals*, and distills the selection logic back into the small model (S2T-Local). Relevance: independent, concurrent evidence that "propose a small candidate set, then rescoring" is becoming a general cross-model architecture pattern; the cascade decode path of this project is the same pattern applied within a single frozen model with a distilled density-based proposer.

**[C2] BitLM (arXiv:2605.11577, 2026).**
Maps tokens to 18-bit binary codes (hypercube vertices) and generates them with a discrete diffusion head under block-causal attention, decoding multiple tokens per block in parallel. Relevance: shares the thesis that a large-vocabulary softmax is a historical choice rather than a necessity, and uses the same hypercube geometry; the mechanism (iterative denoising, from-scratch training, no normalized per-step density, integer codes without semantic layout) differs from this project's. Its appearance indicates that binary-coded vocabularies are gaining traction as a research direction.

## 4. Implications for This Project

Three conclusions follow from the survey. First, every individual ingredient of the current design has identifiable prior art: binary-code vocabularies (A1, A2), mixture output heads (A3), and the mixture-of-factorized-Bernoullis form (A4). Second, the two concurrent works in Group B now also cover two system-level elements — training-free candidate screening with exact rescoring (CFHead), and hierarchical tree codes over the vocabulary (HiDRA) — and the project's main contribution and direction are being adjusted accordingly, toward the aspects these works do not cover: a *replacement* output representation carrying a normalized full-vocabulary density (usable without the original head), the viability of training models natively with such a head, and the semantic code layout as a programmable interface. Third, the evaluation pipeline is being aligned with the protocols of the concurrent works (the CFHead reproduction is the first step), so that upcoming comparisons are like-for-like.
