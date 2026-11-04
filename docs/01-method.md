# Semantic Binary Codes and Mixtures of Bernoullis for Language-Model Output Layers

## Abstract

The output projection of an autoregressive language model reads a vocabulary-sized
weight matrix at each decoding step. MoB_Head replaces this projection with a
compact network predicting a mixture of factorized Bernoulli distributions over
semantic token codes. A balanced embedding tree assigns each token a unique
binary code. The resulting head supports exact code likelihoods, direct code
decoding, and candidate generation followed by optional native-head reranking.
The distinction between a normalized code-space density and its restriction to
valid vocabulary codes is explicit throughout training and evaluation.

## 1. Output representation

For vocabulary size $V$, each token $t$ is assigned an injective code
$c(t)\in\{0,1\}^{M}$, where $M=\lceil\log_2 V\rceil$.
The evaluated vocabularies contain 128,256 tokens for Llama-3.2-1B ($M=17$),
49,152 for SmolLM2-1.7B ($M=16$), and 248,320 for Qwen3.5-2B ($M=18$).
Codes are indexed by token ID; using a different tokenizer or token ordering
invalidates the correspondence even when vocabulary sizes match.

Construction recursively partitions L2-normalized input embeddings. At each node,
cosine two-means supplies a direction from the difference of its cluster centers.
Token projections onto this direction are sorted and split by rank at the median.
Left and right branches append a binary digit. Rank-based balancing prevents
ties and strongly anisotropic embedding geometry from producing long chains.
Unconstrained two-means bisection produced severely unbalanced partitions in the
evaluated embedding tables; it is not the construction used for the released codes.

The Llama codebook assigns all 128,256 tokens distinct vertices. On teacher top-40
target sets, weighted pairwise Hamming distance was 4.16 for semantic codes versus
6.43 for randomly assigned codes. This is a local target-set measurement, rather
than a claim that Hamming distance faithfully recovers the full embedding geometry.

## 2. Mixture density

Given final normalized backbone state $h$, the head produces bit logits
$z_{kj}(h)$ and mixture logits $a_k(h)$. With
$\pi_k=\operatorname{softmax}(a)_k$,

$$
q(c\mid h)=\sum_{k=1}^{K}\pi_k(h)
\prod_{j=1}^{M}\operatorname{Bernoulli}(c_j;\sigma(z_{kj}(h))).
$$

The mixture is exactly normalized on all $2^M$ binary vertices. If some vertices
are unused, its token-space normalization is

$$
Z(h)=\sum_{t=1}^{V}q(c(t)\mid h),\qquad
p(t\mid h)=q(c(t)\mid h)/Z(h).
$$

Reported full-vocabulary perplexities use $p$, including $\log Z$. Training
distillation uses the code-space likelihood $q$. It can therefore penalize
probability assigned to unused vertices through the training targets without
computing a full-vocabulary normalizer on every update.

The implementation evaluates
$\log\operatorname{Bernoulli}(c;\sigma(z))=cz+\log\sigma(-z)$,
then applies log-sum-exp across mixture components. Loss calculations use fp32.

## 3. Head architectures

All heads share the output contract: bit logits of shape `[N,K,M]` and mixture
logits of shape `[N,K]`. Implementations are in
[`mob/mob_common.py`](../mob/mob_common.py).

| Head | Hidden mapping |
|---|---|
| MLP | Three Linear–SiLU hidden layers |
| ResMLP | Input projection, RMS-normalized residual blocks, final RMSNorm |
| SwiGLU | Pre-normalized gated FFN residual block, final RMSNorm |
| Linear | Single readout, used as a diagnostic baseline |

The SwiGLU block is $x\leftarrow x+W_3(\operatorname{SiLU}(W_1\operatorname{RMSNorm}(x))\odot W_2\operatorname{RMSNorm}(x))$.
The common linear readout produces $K(M+1)$ logits. Architecture settings and
codebook dimensions are checkpoint-specific.

## 4. Exact top-k distillation

The teacher and backbone are frozen. At each position, the teacher's top-1024
logits are renormalized into weights $w_t$. The head minimizes

$$
\mathcal L_{KD}=-\sum_{t\in S_{1024}}w_t\log q(c(t)\mid h),
\qquad
\mathcal L=\mathcal L_{KD}+\lambda\sum_k\pi_k\log\pi_k.
$$

This is an exact weighted sum over the retained support, with no sampled targets
or continuous noise. It is not the full teacher KL: truncation and renormalization
omit the teacher tail. On the initial BabyLM measurements, top-40 retained about
78% of teacher mass and top-1024 about 95%.

Stability controls comprise frequency-weighted code-vertex initialization,
mixture-weight entropy regularization, a mixture temperature annealed from 2 to 1
over the first 10% of distillation, and gradient clipping. Component initialization
sets bit biases to approximately $\pm4$. Training monitors effective component
count $\exp H(\overline\pi)$ and components with mean weight below $10^{-4}$.

## 5. Decoding

**Component rounding** chooses a component and rounds each bit probability. A
packed vertex-to-token lookup returns the token; unused vertices use a nearest
valid-code fallback. This path avoids both the native output projection and a
vocabulary-wide softmax, but is not the exact mixture mode.

**Exact mixture ranking** scores valid codes. The `mixture_topk` routine prunes
using $\max_k A_k\leq\log\sum_k e^{A_k}\leq\max_k A_k+\log K$ and exactly
rescoring the survivors. It still scans vocabulary codes; eliminating the native
$h\times V$ projection does not eliminate all vocabulary-dependent computation.

**Mode-and-flip proposals** combine component modes with uncertain-bit flips.
`roundflip` prioritizes component diversity; `scoreflip` selects a mixture-mass
domain, enumerates uncertain-bit alternatives, and truncates by exact mixture
score. Candidate counts depend on the head and hidden-state distribution.

**Cascade reranking** gathers the original embedding rows for the proposed tokens
and computes their exact native logits. This is an approximate native decoder
whose fidelity is limited by candidate recall. It uses original weights and is
distinct from independent generation through the MoB density.

## 6. Computational interpretation

For a tied native head, the output matrix has no extra parameter allocation beyond
the embedding table, but reading it remains expensive. MoB targets this repeated
bandwidth cost. Adding a MoB head to a tied model does not by itself reduce total
parameter count. Pure code decoding can omit the output-table read; cascade
decoding reads selected rows. Input embeddings remain necessary.

## References

- [Bishop, Mixture Density Networks (1994)](https://publications.aston.ac.uk/id/eprint/373/).
- [Weiss et al., Spectral Hashing (2008)](https://proceedings.neurips.cc/paper/2008/hash/d58072be2820e8682c0a27c0518e805e-Abstract.html).
- [Shazeer, GLU Variants Improve Transformer (2020)](https://arxiv.org/abs/2002.05202).
- [CFHead, official implementation](https://github.com/HigherIdeal/CFHead).
