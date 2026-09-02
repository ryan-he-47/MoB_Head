# Prior Work Survey: Replacing/Restructuring the LLM Output (Unembedding + Softmax) Layer

> 归档说明：本文献调研由 librarian 子代理于 2026-08-31 完成，作为 `docs/03-experiment-plan.md` 可行性评估与新颖性定位的依据。所有事实于 2026-08-31 核实；无法核实的标 **UNVERIFIED**。来源：web 搜索、HuggingFace hub API/原始文件、GitHub、arXiv/ACL Anthology。

**Scope**: output-layer / vocab-space restructuring only (not KV cache, weight quantization, pruning).

---

## 1. Prior art on avoiding the h×V matmul / V-dim softmax at inference

### 1.1 Hierarchical softmax — **YES, decode-time speedup, full-vocab distribution**
- **Morin & Bengio 2005**, "Hierarchical Probabilistic Neural Network Language Model" (AISTATS) — replaces the V-way softmax with a sequence of O(log V) binary decisions along a tree; each word gets a **bit-vector code** (path). Notably, their tree was built from the **WordNet IS-A semantic hierarchy** — i.e., a semantic binary tree of words, exactly the "codes arranged by semantics" idea, but with fixed expert knowledge and per-node sigmoids, not a learned hypercube layout. ~200× speedup at train and test. [PDF](https://proceedings.mlr.press/r5/morin05a/morin05a.pdf)
- **Mnih & Hinton 2009**, "A Scalable Hierarchical Distributed Language Model" (NIPS) — automatic data-driven tree construction (feature-based), HLBL; outperformed flat NPLMs. [PDF](http://www.cs.toronto.edu/~amnih/papers/hlbl_final.pdf)
- **Mikolov et al. 2013** (word2vec) popularized HS with a **Huffman tree (frequency-based, not semantic)**. [arXiv:1301.3781]
- **Decode expressivity**: the model defines a valid distribution over the *full* vocabulary; exact argmax requires enumerating all V leaves (O(V log V)) or approximate greedy path descent (O(log V)). So: full-vocab expressivity **preserved**, exact decode **not** O(log V).

### 1.2 Adaptive softmax — **decode-time speedup, near-full expressivity**
- **Grave, Joulin, Cissé, Grangier, Jégou 2017**, "Efficient softmax approximation for GPUs" (ICML; arXiv:1609.04309) — frequency-clustered vocab: shortlist + cluster heads; optimizes an explicit GPU timing model; 2–10× speedup at accuracy parity; builds on **differentiated softmax** (Chen et al. 2015, per-frequency output dims). Their baselines included **"HSM with similarity-based binning"** (HSM_sim) — again, semantic clustering of vocab for the tree, but only as a baseline. Code: [facebookresearch/adaptive-softmax](https://github.com/facebookresearch/adaptive-softmax).
- **Decode expressivity**: still a full-vocab distribution; typical decode scores K clusters + shortlist then the top cluster (approximate greedy). Exact full-vocab scoring (their `getLogProb`) is still O(V).

### 1.3 Sampled / approximate softmax — **TRAINING-TIME ONLY**
- Importance sampling: **Bengio & Senécal 2003/2008**; NCE: **Gutmann & Hyvärinen 2010**, **Mnih & Teh 2012**; **Jean et al. 2015** ("On Using Very Large Target Vocabulary for NMT"); BlackOut: **Ji et al. 2016**.
- **Jozefowicz, Vinyals, Schuster, Shazeer, Wu 2016**, "Exploring the Limits of Language Modeling" (arXiv:1602.02410) — trained the 1B-Word SOTA with importance sampling, and explicitly notes that at evaluation "we need to compute the partition function" — i.e., **the full V-softmax is still required at test time**. They also replaced the softmax with a **character-level LSTM head** at decode (a restructured decode head, but over characters, not codes).
- **Decode expressivity**: nothing changes at decode — you still pay O(V). These methods do **not** address the inference bottleneck.

### 1.4 Error-correcting output codes (ECOC) over the vocabulary — **decode-time speedup; the closest "bit-coded vocab" prior art**
- Classic ECOC: **Dietterich & Bakiri 1995**. For vocabularies: **O'Neill & Bollegala 2019**, "Error-Correcting Neural Sequence Prediction" (ECOC-NSP, arXiv:1901.07002) — each token is a binary codeword (40–5000 bits, |c| ≪ V); each bit is a Bernoulli predicted from h (no h×V matmul); **codebook ordered by embedding cosine similarity** ("incorrect predictions are at least semantically close to the target"); decode = argmax codeword → token. Near-full-softmax LM perplexity with ≤1/20 the output parameters; applied to LM and image captioning.
- **Decode expressivity**: full vocab reachable (unique codewords), but the induced distribution over tokens is implicit/approximate (no guaranteed normalization), and codes are *redundant* (not minimal hypercube codes).

### 1.5 Binary / semantic hashing of the vocab into code bits
- The bit-path codes of hierarchical softmax (Morin & Bengio 2005) and ECOC-NSP (2019) are the only substantive generation-side uses found. Recent:
- **"Language Models Without a Trainable Input Embedding Table: Learning from Fixed Minimal Binary Token Codes"** (arXiv:2605.09751, 2026) — **input-side** minimal codes c(t) ∈ {0,1}^K, K = ⌈log2 V⌉, tiled to model width, with random affine recoding over F₂^K; matches learned-embedding perplexity at V=65,536 (K=16); output head left standard. Directly validates "minimal binary hypercube codes carry enough info for an LM" — but on the input side.
- **MultiHashFormer** (arXiv:2606.28057, 2026, Xue et al.) — decoder LM where each token is a **unique multi-hash signature** [h₁(w), h₂(w), …] and a cascaded "Hash Decoder" auto-regressively predicts the signature, deterministically mapped back to text; sub-linear vocab parameter footprint; collisions resolved by rehashing. Hash assignment is random, **not** semantic.
- **Decode expressivity**: full vocab, but MultiHashFormer codes are random (no semantic layout); the 2026 binary-input paper is input-side only.

### 1.6 VQ / quantized output heads — **decode-time speedup, expressivity capped by codebook**
- **VQ-Logits** (Shao et al., arXiv:2505.10202, 2025) — replace the V×d output matrix with a K≪V codebook (k-means over pretrained embeddings); predict logits over the codebook, then **scatter to the full vocab** and softmax; up to 99% output-param reduction, ~6× logit-computation speedup, ~4% PPL cost (K=2048, WikiText-103). Caveats: tokens sharing a codebook entry get **identical logits**; the scatter + softmax still touch V entries; no semantic ordering of codes beyond k-means proximity.
- **P-VQ** (Zhang, Tu, Yan, AAAI 2020) — partial VQ of tied embeddings for NMT softmax; replaces most multiplications with lookups; ~74% softmax FLOPs reduction, −0.04 BLEU. [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/17688/17495)
- **Decode expressivity**: full vocab coverage, but per-codebook-cell probability ties; PPL shows measurable loss vs full softmax.
- (GPTVQ, SoftWater etc. quantize the *weights* of the head — explicitly out of scope.)

### 1.7 Mixture-of-softmaxes / "softmax bottleneck" line — **expressivity, not efficiency**
- **Yang, Dai, Salakhutdinov, Cohen 2018**, "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model" (ICLR; arXiv:1711.03953) — shows softmax+dot-product logits are rank-limited (the "softmax bottleneck") and proposes **Mixture of Softmaxes (MoS)**: next-token distribution = Σ π_k softmax(h_k^T W). More expressive, better PPL; still K×V-way normalization — it does **not** reduce decode cost.
- **Kanai et al. 2018**, "Sigsoftmax: Reanalysis of the Softmax Bottleneck" (NeurIPS; arXiv:1805.10829) — rank-breaking activation without extra parameters; mixture variant MoSS > MoS.
- GitHub evidence of adoption: `MixtureOfSoftmaxes` in [tensorflow/mesh](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/vocab_embeddings.py#L245) and [ufal/perin](https://github.com/ufal/perin/blob/main/model/module/mixture_of_softmaxes.py).
- **Decode expressivity**: identical to full softmax (it *is* a full softmax per component) — no decode saving.

### 1.8 Continuous / bit-coded prediction instead of one-hot vocab — **the "continuous output" line**
- **Kumar & Tsvetkov 2019**, "Von Mises-Fisher Loss for Training Sequence to Seq Models with Continuous Outputs" (ICLR; arXiv:1812.04616) — the foundational "replace softmax with a **density over pretrained word embeddings**" paper: output a vector that parameterizes a **von Mises-Fisher (spherical)** distribution; decode = nearest-neighbor embedding lookup (cosine); up to 2.5× faster training, comparable BLEU, errors are semantically close to the reference. They explicitly cite **Bishop 1994** (MDN) as the correct probabilistic framework.
- **Li, Chen, Hsieh, Chang 2019**, "Efficient Contextual Representation Learning With Continuous Outputs" (TACL 7:611–624) — same idea for ELMo-style encoders; 4× speedup, −80% params. [ACL](https://aclanthology.org/Q19-1039/)
- Follow-ups: **Bhat, Kumar, Tsvetkov 2019** (syn-margin loss, WMT; [D19-5621](https://aclanthology.org/D19-5621.pdf)); **Tokarchuk & Niculae 2022** (target representation choice, [2022.repl4nlp-1.24](https://aclanthology.org/2022.repl4nlp-1.24/)); NAACL 2024 short "The Unreasonable Effectiveness of Random Target Embeddings for Continuous-Output NMT" ([2024.naacl-short.56](https://aclanthology.org/2024.naacl-short.56.pdf)); **Jauregi Unanue et al. 2019** (ReWE, auxiliary embedding regression, [N19-1041](https://aclanthology.org/N19-1041.pdf)).
- Spherical-loss family (training-side alternatives): **de Brébisson & Vincent 2016** (ICLR, arXiv:1511.05042) — spherical/Taylor softmax with O(d²) exact output updates.
- **Decode expressivity**: full vocab via nearest-neighbor lookup (O(V) search, or ANN index), but **no calibrated full-vocab probability distribution** at decode (greedy NN only); diversity is a known weakness (noted by GIVT's related-work review).

---

## 2. GMM / mixture-density output heads for next-token prediction (closest neighbors)

- **Bishop 1994**, "Mixture Density Networks" — the foundational conditional-GMM regression framework (cited by Kumar & Tsvetkov as the right probabilistic tool for continuous outputs).
- **Zen & Senior 2014**, "Deep mixture density networks for acoustic modeling in statistical parametric speech synthesis" (ICASSP) — MDN output layer for sequence generation (speech features): first major "GMM head" in autoregressive generative modeling. [DOI](https://doi.org/10.1109/icassp.2014.6854321)
- **Ha & Schmidhuber 2018**, "World Models" (NIPS) — MDN-RNN predicts a GMM over the next latent state; continuous-state autoregressive GMM head (not tokens).
- **GIVT** (Tschannen, Eastwood, Mentzer; arXiv:2312.02116, ECCV 2024) — decoder-only transformer whose head outputs **parameters of a d-variate GMM** over real-valued latents (initially per-channel scalar GMMs, then diagonal-covariance d-variate GMM); replaces logits entirely; temperature sampling becomes "variance scaling". Image/segmentation/depth domains, but architecturally the cleanest "GMM head on a transformer decoder".
- **TarFlowLM** (NeurIPS 2025; arXiv:2507.00425) — language modeling in a continuous latent space with transformer autoregressive flows; explicitly models each next latent as a **d-dimensional mixture of Gaussians with a shared codebook of V Gaussian components**, and proves MoG ⇔ Mixture-Rosenblatt flow equivalence. GMM next-"token" density for text — the nearest conceptual sibling in LM-land, but over VAE latents, not vocab bits.
- **ECOC-NSP** (2019) — its LVMS "Codeword Mixture Sampling" mixes discrete codewords at decode (mixture over codes, not a continuous GMM).
- **Kumar & Tsvetkov 2019** — single (unimodal) vMF, i.e., **spherical** density over word embeddings; the closest "spherical density for token prediction" — but one component, no bit codes, no semantics-by-construction.
- **Finding**: no prior work found (as of Aug 2026) that uses a **spherical GMM head over minimal bit-codes of vocab vertices** for next-token prediction. The ingredients exist separately (GMM heads: GIVT/TarFlowLM/Zen; spherical token densities: vMF; bit codes: HS/ECOC/MultiHashFormer; semantic code layouts: ECOC-NSP/Morin-Bengio) — the specific combination appears novel.

---

## 3. Assigning vocab to structured codes by embedding similarity

- **Semantic hashing (documents)**: **Salakhutdinov & Hinton 2009**, "Semantic Hashing" (Int. J. Approx. Reasoning 50:969–978) — deep autoencoder → binary codes; similar docs at nearby Hamming addresses; **trained by injecting noise at the code layer** (directly analogous to the noise-tolerance trick in this project's training). [PDF](http://www.cs.toronto.edu/~rsalakhu/papers/sdarticle.pdf)
- **Spectral hashing**: **Weiss, Torralba, Fergus 2008** (NIPS) — similarity-preserving codes via thresholded graph-Laplacian eigenvectors. [PDF](https://proceedings.neurips.cc/paper_files/paper/2008/file/d58072be2820e8682c0a27c0518e805e-Paper.pdf)
- **ITQ**: **Gong, Lazebnik, Gordo, Perronnin** (CVPR 2011 / TPAMI 2013) — rotate zero-centered data to minimize quantization error onto **binary hypercube vertices**; the canonical "hypercube-vertex codes preserving neighborhood structure" algorithm. [TPAMI](https://ieeexplore.ieee.org/document/6296665)
- **LSH lineage**: Indyk & Motwani 1998; **Charikar 2002** (SimHash). **NASH**: **Shen et al. 2018** (ACL P18-1190) — end-to-end semantic hashing with Bernoulli latents + **data-dependent noise injection** (rate–distortion motivated). **SemanticHash** (IEEE TNNLS 2021, [9389611](https://ieeexplore.ieee.org/document/9389611)) — class label prototypes on the K-dim **unit hypercube** for zero-shot classification (codes reflect label semantics).
- **Vocab-specific structured codes**:
  - **Morin & Bengio 2005** — binary tree from **WordNet semantics** (semantic tree, generation-side). ✔ exploited for generation.
  - **Mnih & Hinton 2009** — data-driven tree from features (semantic-ish clustering, generation-side). ✔
  - **Grave et al. 2017** — "HSM with similarity-based binning" baseline. ✔
  - **ECOC-NSP 2019** — codebook **rank-ordered by embedding cosine similarity**; used for LM + captioning, with the explicit claim that mispredictions land on semantically related tokens. ✔ exploited for generation (the closest match to "nearby codes = nearby semantics for generation").
  - **word2vec Huffman HS** — frequency-based only (not semantic). **LightRNN** (Li et al. 2016), **hash embeddings** (Svenstrup et al. 2017), **compositional code learning** (Shu & Nakayama 2018), **HashFormers** (Xue & Aletras, EMNLP 2022) — all **input-side** vocab structuring, not decode heads.
  - **MultiHashFormer 2026** — generation-side hash signatures, but **random** (no semantic layout).
- **Bottom line**: "nearby codes = nearby semantics" has been *demonstrated* for generation in exactly two lines: (i) semantic/embedding-ordered hierarchical structures (Morin-Bengio, Mnih-Hinton, Grave-sim, ECOC-NSP), and (ii) implicit semantic neighborhoods via continuous-output NMT (decode-to-nearest-embedding). But no prior work builds a **minimal M=⌈log2 V⌉ hypercube layout from a pretrained LLM's own embedding cosine-similarity tree** and then distills a frozen backbone into a GMM over those vertices.

---

## 4. Fact-check block (official sources)

### 4.1 meta-llama/Llama-3.2-1B
- **Weights gated: YES** — official HF API returns `"gated": "manual"`, license `llama3.2`, with the full **Llama 3.2 Community License Agreement** as the gating prompt (accept via form) — verified via [HF API](https://huggingface.co/api/models/meta-llama/Llama-3.2-1B) (field `cardData.extra_gated_prompt`). Raw `config.json` and `model.safetensors` return HTTP 401 without auth. Total params from official API: **1,235,814,400 (BF16)**.
- **Architecture facts** (verified from `config.json`; note: the *official* config is gated, so values were read from the byte-preserved public mirror [unsloth/Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B/raw/main/config.json), which matches the official release architecture):
  - `vocab_size`: **128,256** ⇒ M = ⌈log₂(128256)⌉ = **17 bits** (2¹⁷ = 131,072 ≥ 128,256; 2,816 unused vertices)
  - `hidden_size`: **2048**; `intermediate_size`: 8192; `num_hidden_layers`: **16**; heads 32 / KV-heads 8 / head_dim 64
  - `tie_word_embeddings`: **true** (input/output embeddings tied)
  - `hidden_act`: **silu**; `rope_theta`: 500,000; `max_position_embeddings`: 131,072; `torch_dtype`: bfloat16
  - Model type `LlamaForCausalLM`, `original/params.json` present in the official repo (gated).

### 4.2 BabyLM dataset
- **2023 round** (CoNLL 2023; tracks **Strict = 100M words, Strict-Small = 10M words, Loose = 100M + extra non-linguistic data**; findings paper: **Warstadt et al. 2023**, [2023.conll-babylm.1](https://aclanthology.org/2023.conll-babylm.1/)):
  - Official distribution: `babylm_data.zip` (240 MB zipped / ~700 MB unzipped) at [github.com/babylm/babylm.github.io/raw/main/babylm_data.zip](https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip) — contains `babylm_100M` (strict train), `babylm_10M` (strict-small train), `babylm_dev` (10M dev), `babylm_test` (10M test), each as per-domain `.txt` files (aochildes, bnc_spoken, cbt, children_stories, gutenberg, open_subtitles, qed, simple_wikipedia, switchboard, wikipedia). [README text verified](https://github.com/babylm/baseline-pretraining)
  - The commonly cited HF repo **`babylm/babylm`** exists in the literature; its current live status **UNVERIFIED** — API checks returned 401/429 (HF auth/rate limiting), and the datasets-server reports it inaccessible.
- **2024 round** (CoNLL 2024; tracks **Strict = 100M, Strict-Small = 10M, Multimodal = 100M text + 2.9M images, Paper**; findings paper: **Hu et al. 2024**, "Findings of the Second BabyLM Challenge", [2024.conll-babylm.1](https://aclanthology.org/2024.conll-babylm.1/), arXiv:2412.05149):
  - Official data: OSF **https://osf.io/ad7qg/** (per the 2024 CFP, [arXiv:2404.06214](https://doi.org/10.48550/arxiv.2404.06214)); updated corpus (CHILDES 29M, Gutenberg children's stories 26M, OpenSubtitles 20M, Simple Wikipedia 15M, BNC dialogue 8M, Switchboard 1M); multimodal = 50M text + Localized Narratives 27M + CC3M 23M.
  - HF repo **`babylm/babylm-2024`** is widely cited; live status **UNVERIFIED** (same 401/429 issue). Many community re-uploads exist (e.g., `nilq/babylm-100M`).
  - **Current official home (2026 round, "BabyLM Turns 4", EMNLP 2026)**: HF org **[BabyLM-community](https://huggingface.co/BabyLM-community)** — verified via HF API: `BabyLM-community/BabyLM-2026-Strict` (detoxified 100M, MIT, ~98.9M tokens), `BabyLM-community/BabyLM-2026-Strict-Small` (10M), plus `BabyLM-dev`, `BabyLM-Test`, `BabyLM-2026-Strict-Evals`, and the BabyBabelLM multilingual collection (e.g., `babylm-eng` 100M tier). Website: [babylm.github.io](https://babylm.github.io/) (round 4: EMNLP 2026, Budapest; new MultiLingual track; 10-epoch limit).
- **GitHub org URL**: **https://github.com/babylm** ("BabyLM Shared Task", co-hosted with CMCL & CoNLL 2023; 7 public repos incl. `evaluation-pipeline-2023/2024/2025`, `baseline-pretraining`, `babylm_data_preprocessing`, `babylm.github.io`); 2026 evaluation lives at **github.com/babylm-org/babylm-eval**. [Org page verified](https://github.com/babylm)

---

## 5. Novelty assessment

The closest prior art splits into three strands that this project is the first to fuse: **(i)** bit-coded vocab outputs with *semantic* code layout (ECOC-NSP 2019; Morin & Bengio 2005's WordNet tree), **(ii)** continuous/spherical token densities (Kumar & Tsvetkov's vMF head, 2019; GIVT's GMM head, 2024; TarFlowLM's MoG next-latent priors, 2025), and **(iii)** hypercube-vertex codes preserving similarity (ITQ, spectral hashing, semantic hashing). No prior work combines *all* of: a frozen pretrained backbone, distillation from the teacher's top-k next-token distribution, minimal M=17-bit hypercube codes assigned by a cosine-similarity binary tree built from the LLM's own embedding table, and a multi-component spherical GMM head sampled by component-weight temperature at decode. The two nearest neighbors — ECOC-NSP (semantic bit codes for generation, but trains from scratch, uses redundant error-checking codes rather than minimal hypercube codes, and has a Bernoulli not GMM head) and Kumar & Tsvetkov (spherical density over embeddings, but unimodal and not tied to bit-coded vertices) — each lack the other's key ingredient; VQ-Logits (2025) adds codebook quantization but keeps a V-sized softmax at decode with no semantic ordering. The novelty is therefore real but *incremental-by-composition*: the strongest citation to anchor against is ECOC-NSP 2019 (semantic bit-coded generation) and Kumar & Tsvetkov 2019 (spherical token density), with GIVT/TarFlowLM supplying the GMM-head machinery and NASH/Salakhutdinov-Hinton the noise-injection training trick.
