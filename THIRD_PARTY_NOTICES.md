# Third-party notices

The MIT license at the repository root applies to the original MoB_Head software
and documentation. It does not relicense external models, datasets, or code.

## Model dependencies

- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B): Apache-2.0.
- [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B): Apache-2.0.
- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B):
  [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE).
  Built with Llama. The corresponding head artifact is named with a Llama prefix
  and includes the original agreement and attribution. Backbone parameters and
  tokenizer files are obtained from the respective providers, not this release.

Codebooks are constructed from each model's embedding geometry. Original codebook
construction software is MIT; use of the artifacts with the source models remains
subject to their model terms. Each exported model card identifies the source.

## CFHead

The baseline is [HigherIdeal/CFHead](https://github.com/HigherIdeal/CFHead),
*Fast LM Head via Coarse-Fine Decomposition and Tile-Wise Weight Clustering*.
Its upstream README describes a research-purpose release. Upstream code is not
included in this source distribution or relicensed as MIT. The optional wrappers
load a separately obtained checkout via `CFHEAD_REPO`:

```bash
git clone https://github.com/HigherIdeal/CFHead.git /path/to/CFHead
export CFHEAD_REPO=/path/to/CFHead
```

The numerical reports retain the original comparison settings. Reproducing
optimized CFHead timings requires its kernels and matching hardware/software;
the chunked torch fallback has different latency characteristics.

## Data

BabyLM, FineWeb-Edu, CCI3-HQ, The Stack v3, SmolTalk, and evaluation datasets
retain their source licenses and access conditions. This repository distributes
processing code and aggregate numerical measurements, not their text corpora.

## Software dependencies

PyTorch, Transformers, NumPy, safetensors, huggingface_hub, datasets, pyarrow,
numba and optional Triton are external dependencies under their own licenses.
