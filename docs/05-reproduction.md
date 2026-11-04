# Reproduction and Model Artifacts

## Software and layout

Run entry points as modules from the repository root. Python 3.10 or newer is
required by the source annotations. Core dependencies are torch, numpy,
transformers, safetensors, and huggingface_hub. Dataset builders additionally use
datasets and pyarrow; CPU proposal acceleration uses numba. Triton is optional.

```bash
python -m pip install -r requirements.txt
python -m tools.workspace_check
```

The recorded local environment used torch 2.11.0, transformers 5.16.1 and numpy
2.4.3 on an RTX 4060 Laptop with 8GB VRAM. Training runs used V100 32GB GPUs.
Select a torch build compatible with the actual driver and GPU; the local CUDA
build is not a cluster dependency lock.

Slurm templates use `MOB_ROOT` for the checkout, `PYTHON` for the interpreter and
`MOB_SCRATCH` for bulk resources. Create `logs/` before submission; Slurm output
directives use paths relative to the submission directory. Partition and GPU
resource names must match the target cluster.

| Directory | Contents |
|---|---|
| `mob/` | Shared heads, density math, decoding, Hub artifact loader |
| `train/` | Distillation, paired architecture study, joint adaptation, native pretraining |
| `eval/` | Normalized likelihood, candidate fidelity, generation and efficiency harnesses |
| `tools/` | Data/codebook builders and model export/publication |
| `slurm/` | Reproduction templates, configured through environment variables |
| `results/` | Numerical evaluation summaries |
| `release/` | Head manifest and artifact metadata |
| `data/` | Token-ID-indexed codebooks |

## Loading a head

The artifacts contain the head and its exact codebook, not backbone weights,
tokenizer files, or optimizer state. They are not full Transformers causal
language models and cannot be loaded with `AutoModelForCausalLM`.

```python
import torch
from mob.hub import load_head

head, codebook, config = load_head("dist/mob-qwen3.5-2b-swiglu")
# A published Hub repository ID can replace the local directory.
h = torch.zeros(2, config["head_kwargs"]["h"])
bit_logits, mixture_logits = head(h)
assert bit_logits.shape == (2, config["head_kwargs"]["k"], config["head_kwargs"]["m"])
```

The backbone must match `base_model` and the tokenizer's token-ID ordering.
Tap the final normalized hidden state corresponding to the native output-head
input. In cascade mode, retain access to the correct native output rows.
The head loader uses safetensors and validates the codebook checksum.

## Frozen-backbone distillation

This is the recorded Qwen recipe, with explicitly supplied local resources:

```bash
python -m train.p2_train --data-root . \
  --model-dir models/Qwen3.5-2B --head-type swiglu --hidden 7300 --blocks 1 \
  --codebook codebook_qwen35.npz \
  --token-cache qwen35_mix_train_tokens.npy \
  --dev-cache qwen35_dev_mix_tokens.npy \
  --data-format hf --dev-format hf --k 128 --epochs 1 --batch 16 \
  --topk 1024 --loss-chunk 2048 --warmup 4000 --schedule cosine --lr 1e-3 \
  --backbone-dtype float16 --ckpt-every 20000 --ckpt-keep 3 \
  --outdir runs/qwen35_distillation
```

The released checkpoint is from update 224,000. The full command's scheduled
duration is longer; reproducing the reported checkpoint requires comparing
that update, not assuming identical terminal training length.

The English builder is `python -m tools.build_fineweb_cache --help`; the
multilingual document-interleaver is `python -m tools.build_multimix_cache --help`.
The balanced semantic codebook builder is `python -m tools.build_codebook --help`.
Dataset and model access is subject to each source's own terms.

## Evaluation

```bash
python -m eval.p3_eval --data-root . --model-dir models/Qwen3.5-2B \
  --head runs/qwen35_head_latest.pt --head-type swiglu --hidden 7300 \
  --blocks 1 --k 128 --codebook codebook_qwen35.npz \
  --token-cache data/qwen35_dev_en_tokens.npy --backbone-dtype bfloat16 \
  --n-chunks 64 --batch 4 --outdir runs/qwen35_eval
```

Legacy evaluation entry points accept the original state-dict `.pt` files.
For a safetensors package, load it with `mob.hub.load_head` and supply that head
to custom inference code. Preserve fp32 likelihood computation and valid-code
normalization. Candidate recall, reranking agreement and normalized PPL measure
different properties.

CFHead cache construction and optional Triton acceleration load the official
upstream source from `CFHEAD_REPO`; see [third-party notices](../THIRD_PARTY_NOTICES.md).
The torch evaluation path can read existing compatible cache files without
the upstream kernels.

## Export and publication

```bash
python -m tools.export_heads --out-dir dist
hf auth login
python -m tools.publish_heads --namespace YOUR_NAMESPACE --directory dist
```

The exporter reads only the selected head state dicts listed in
[`release/heads.json`](../release/heads.json), preserves fp32 weights, and packages
config, safetensors, codebook, hashes, license and a model card. Export validation
strictly reloads every tensor and checks numerical forward equivalence on CPU.
Publication uploads only the package allowlist and verifies remote file hashes.
No backbone, training corpus, raw run directory or local account credential is
part of an artifact.

## Licensing

Original software and documentation are MIT-licensed. Backbone models, datasets,
and independently obtained third-party implementations retain their respective
licenses. Qwen/SmolLM2 packages contain original head parameters and a codebook
with source attribution. The Llama-derived artifact includes the Llama 3.2
Community License and required notice; it is labeled separately from the MIT
software. See [third-party notices](../THIRD_PARTY_NOTICES.md).
