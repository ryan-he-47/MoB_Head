"""Load original MoB head artifacts from a local directory or the Hugging Face Hub."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from mob.mob_common import MoBHead, ResMLPHead, SwiGLUHead

HEAD_TYPES = {"mob": MoBHead, "resmlp": ResMLPHead, "swiglu": SwiGLUHead}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_head(source: str | Path, *, revision: str | None = None,
              device: str | torch.device = "cpu", local_files_only: bool = False):
    """Return (eval-mode head, codebook arrays, config), without loading a backbone.

    Hub loads pin every artifact to the config's resolved commit. Optional
    ``revision`` accepts a commit/tag/branch. No remote Python code is executed.
    """
    source = str(source)
    directory = Path(source)
    if directory.is_dir():
        config_path = directory / "config.json"
        resolve = lambda filename: directory / filename
    else:
        from huggingface_hub import hf_hub_download
        config_path = Path(hf_hub_download(source, "config.json", revision=revision,
                                          local_files_only=local_files_only))
        commit = config_path.parent.name
        resolve = lambda filename: Path(hf_hub_download(
            source, filename, revision=commit, local_files_only=local_files_only))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format_version") != 1 or config.get("model_type") != "mob_head":
        raise ValueError("Unsupported MoB head artifact format")
    paths = {name: resolve(name) for name in ("model.safetensors", "codebook.npz")}
    for name, path in paths.items():
        if sha256(path) != config["sha256"][name]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    head = HEAD_TYPES[config["head_type"]](**config["head_kwargs"])
    state = load_file(str(paths["model.safetensors"]), device="cpu")
    head.load_state_dict(state, strict=True)
    with np.load(paths["codebook.npz"], allow_pickle=False) as archive:
        codebook = {key: archive[key].copy() for key in archive.files}
    codes = codebook["codes"]
    if codes.shape != (config["vocab_size"], config["head_kwargs"]["m"]):
        raise ValueError("Codebook dimensions disagree with artifact config")
    if not np.isin(codes, (0, 1)).all():
        raise ValueError("Codebook contains non-binary values")
    expected_vid = codes.astype(np.int64) @ (1 << np.arange(codes.shape[1] - 1, -1, -1))
    if len(np.unique(expected_vid)) != len(codes) or not np.array_equal(expected_vid, codebook["vertex_id"]):
        raise ValueError("Codebook is not an injective token-ID-to-vertex mapping")
    return head.to(device).eval(), codebook, config
