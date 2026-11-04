"""Build compatible baseline caches using the separately installed CFHead implementation.
Set CFHEAD_REPO to the official upstream checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

try:
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover
    raise SystemExit("safetensors required") from exc

CACHE_VERSION = 3


def find_embedding_tensor(model_dir: Path) -> tuple[torch.Tensor, str]:
    """Return (weight [V,H] fp32-on-CPU->caller casts, source dtype tag)."""
    index = model_dir / "model.safetensors.index.json"
    shards: list[Path]
    if index.exists():
        meta = json.loads(index.read_text())
        shards = sorted({model_dir / f for f in meta["weight_map"].values()})
    else:
        shards = sorted(model_dir.glob("*.safetensors"))
    candidates = ("model.embed_tokens.weight", "embed_tokens.weight", "lm_head.weight",
                  "model.language_model.embed_tokens.weight")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            hit = next((k for k in candidates if k in keys), None)
            if hit is None:  # fallback: any *embed_tokens.weight (multimodal wrappers nest)
                hit = next((k for k in keys if k.endswith("embed_tokens.weight")), None)
            if hit is not None:
                t = f.get_tensor(hit)
                return t, str(t.dtype)  # repo metadata keeps the "torch." prefix
    raise SystemExit(f"no embedding/lm_head tensor found in {shards}")


def fingerprint_weight(weight: torch.Tensor, samples: int = 4096) -> str:
    flat = weight.detach().reshape(-1)
    total = flat.numel()
    n = min(samples, total)
    idx = torch.arange(n, dtype=torch.long) * (total - 1) // (n - 1)
    sample = flat[idx].float().cpu().contiguous()
    return hashlib.sha1(sample.numpy().tobytes()).hexdigest()[:16]


@torch.no_grad()
def build_one(weight: torch.Tensor, tile: int, clusters: int, iters: int, seed: int,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    from types import SimpleNamespace
    from eval.cfhead_kernels import load_upstream
    upstream = load_upstream("algorithm/CFHead.py", "_mob_cfhead_algorithm")
    w = weight.to(device=device, dtype=torch.float32)
    context = SimpleNamespace(vocab_size=w.shape[0], hidden_size=w.shape[1],
                              tile_size=tile, cluster_size=clusters,
                              num_tiles=w.shape[1] // tile, seed=seed,
                              kmeans_iters=iters, _weight=lambda: w)
    if w.shape[1] % tile:
        raise ValueError("Hidden dimension must be divisible by tile width")
    return upstream.CFHeadSelector._build_codebook_and_index(context)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--clusters", type=int, nargs="+", default=[64, 4])
    ap.add_argument("--top-k", type=int, default=256, help="metadata tag only (k is a runtime knob)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    weight_cpu, dtype_tag = find_embedding_tensor(model_dir)
    V, H = weight_cpu.shape
    print(f"[cfhead-cache] weight [{V}, {H}] dtype={dtype_tag} load {time.time() - t0:.1f}s", flush=True)
    fingerprint = fingerprint_weight(weight_cpu)
    print(f"[cfhead-cache] fingerprint={fingerprint}", flush=True)

    for clusters in args.clusters:
        t1 = time.time()
        codebook, index_map = build_one(weight_cpu, args.tile, clusters, args.iters, args.seed, device)
        meta = {
            "version": CACHE_VERSION,
            "vocab_size": V,
            "hidden_size": H,
            "tile_size": args.tile,
            "cluster_size": clusters,
            "top_k": args.top_k,
            "kmeans_iters": args.iters,
            "seed": args.seed,
            "source_dtype": dtype_tag,
            "weight_fingerprint": fingerprint,
            "index_dtype": "uint8",
        }
        name = (f"cfhead_v{CACHE_VERSION}_V{V}_H{H}_T{args.tile}_C{clusters}"
                f"_K{args.top_k}_iters{args.iters}_seed{args.seed}_{dtype_tag.replace('torch.', '')}.pt")
        path = out_dir / name
        torch.save({"metadata": meta, "codebook": codebook.cpu(), "index_map": index_map.cpu()}, path)
        mib = (codebook.numel() * 4 + index_map.numel()) / 1024 / 1024
        print(f"[cfhead-cache] C={clusters} done {time.time() - t1:.1f}s -> {path} ({mib:.1f} MiB)",
              flush=True)


if __name__ == "__main__":
    main()
