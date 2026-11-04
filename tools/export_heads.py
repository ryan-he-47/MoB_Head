"""Export the curated frozen-backbone heads as validated safetensors packages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from safetensors.torch import save_file
import torch

from mob.hub import HEAD_TYPES, load_head, sha256

ROOT = Path(__file__).resolve().parents[1]


def model_card(spec: dict, config: dict) -> str:
    language = "[en, zh]" if "qwen" in spec["name"] else "[en]"
    attribution = ""
    if spec["license"] == "llama3.2":
        attribution = "\nBuilt with Llama. This artifact is subject to the Llama 3.2 Community License.\n"
    name = spec["name"]
    return f'''---
license: {spec["license"]}
language: {language}
base_model: {spec["base_model"]}
tags:
- mob-head
- mixture-of-bernoullis
- knowledge-distillation
- semantic-binary-codes
- output-layer
---
# {name}

A compact Mixture-of-Bernoullis output head from
[MoB_Head](https://github.com/ryan-he-47/MoB_Head). This release contains
**head weights and a token-ID-indexed semantic codebook**, for use with
[{spec["base_model"]}](https://huggingface.co/{spec["base_model"]}).
{attribution}
## Architecture

- Head family: `{spec["head_type"]}`.
- Constructor arguments: `{json.dumps(spec["head_kwargs"], sort_keys=True)}`.
- Head parameters: {config["num_parameters"]:,}.
- Vocabulary: {config["vocab_size"]:,} tokens.
- Serialization: fp32 safetensors, without optimizer state.

The backbone and tokenizer are separate dependencies. This is not a complete
Transformers causal language model, and `AutoModelForCausalLM` cannot load it.
Do not substitute a different tokenizer or token-ID ordering.

## Training

The backbone is frozen. The head minimizes exact teacher-probability-weighted
code NLL on the teacher's renormalized top-1024 support. No sampled targets or
continuous target noise are used. Stability controls include vertex initialization,
mixture entropy regularization and mixture-temperature annealing.

```json
{json.dumps(spec["training"], indent=2)}
```

## Recorded evaluation

```json
{json.dumps(spec["metrics"], indent=2)}
```

Normalized perplexity uses a sum over valid vocabulary codes, not all hypercube
vertices. Candidate recall and reranking agreement are separate measurements.
See the repository's technical reports for datasets, position counts, precision,
and comparison settings. The Qwen artifact is update 224k/244k; reported results
are for that checkpoint, with no assumed completion of the schedule.

## Usage

Install the MoB_Head source and dependencies, then:

```python
from mob.hub import load_head
head, codebook, config = load_head("ryan1145/{name}")
# h: final normalized hidden states, shape [positions, hidden_size].
# bit_logits, mixture_logits = head(h)
```

The loader checks weight/codebook hashes and loads no remote Python code.
For independent decoding, use `mob.mob_common` code decoders. Cascade reranking
also requires the matching native output-table rows.

## Limitations

The head approximates its teacher and does not preserve its full distribution.
Independent long-form generation can drift; cascade decoding is constrained by
proposal recall. Speedups depend on hardware, vocabulary, batch and decode path.
The project comparison found performance close to CFHead in several settings,
but did not surpass its overall acceleration tradeoff.

## License and attribution

Artifact license: **{spec["license"]}**; see `LICENSE` and `NOTICE`.
Original loading code is MIT-licensed in MoB_Head. Source backbone and dataset
terms remain applicable. No backbone parameters or corpus text are included.
'''


def export_one(spec: dict, destination: Path, license_dir: Path) -> dict:
    output = destination / spec["name"]
    output.mkdir(parents=True, exist_ok=True)
    state = torch.load(ROOT / spec["checkpoint"], map_location="cpu", weights_only=True)
    head = HEAD_TYPES[spec["head_type"]](**spec["head_kwargs"])
    head.load_state_dict(state, strict=True)
    head.eval()
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError(f"Non-finite checkpoint: {spec['name']}")
    save_file({key: value.detach().cpu().contiguous() for key, value in state.items()},
              str(output / "model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(ROOT / spec["codebook"], output / "codebook.npz")
    with np.load(output / "codebook.npz", allow_pickle=False) as archive:
        vocab = len(archive["codes"])
    config = {"format_version": 1, "model_type": "mob_head",
              "head_type": spec["head_type"], "head_kwargs": spec["head_kwargs"],
              "base_model": spec["base_model"], "vocab_size": vocab,
              "num_parameters": sum(value.numel() for value in state.values()),
              "weight_dtype": "float32", "training": spec["training"], "metrics": spec["metrics"],
              "sha256": {name: sha256(output / name) for name in ("model.safetensors", "codebook.npz")}}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(model_card(spec, config), encoding="utf-8")
    if spec["license"] == "llama3.2":
        shutil.copyfile(license_dir / "llama3.2.txt", output / "LICENSE")
        shutil.copyfile(license_dir / "llama3.2-use-policy.md", output / "USE_POLICY.md")
        notice = "Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.\nBuilt with Llama.\n"
    else:
        shutil.copyfile(ROOT / "LICENSE", output / "LICENSE")
        shutil.copyfile(license_dir / "apache-2.0.txt", output / "BASE_MODEL_LICENSE")
        notice = (f"Original MoB head parameters: Copyright (c) 2026 MoB_Head contributors.\n"
                  f"Source backbone and embedding geometry: {spec['base_model']} (Apache-2.0).\n"
                  "Backbone parameters are not included. The codebook is derived from the source embedding geometry.\n")
    (output / "NOTICE").write_text(notice, encoding="utf-8")
    restored, codes, restored_config = load_head(output)
    for key, tensor in restored.state_dict().items():
        if not torch.equal(tensor, state[key]):
            raise AssertionError(f"Serialization changed {key}")
    generator = torch.Generator().manual_seed(42)
    inputs = torch.randn(3, spec["head_kwargs"]["h"], generator=generator)
    with torch.inference_mode():
        for expected, actual in zip(head(inputs), restored(inputs)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    files = {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
             for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_manifest.json"}
    manifest = {"format_version": 1, "name": spec["name"], "files": files}
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PASS {spec['name']}: {config['num_parameters']:,} parameters; exact reload/forward", flush=True)
    return {"name": spec["name"], "num_parameters": config["num_parameters"],
            "sha256": config["sha256"], "bytes": sum(item["bytes"] for item in files.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "release/heads.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--license-dir", type=Path, default=ROOT / "release/licenses")
    args = parser.parse_args()
    torch.set_num_threads(2)
    specs = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = [export_one(spec, args.out_dir, args.license_dir) for spec in specs]
    (ROOT / "release/artifacts.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
