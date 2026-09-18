"""Build CFHead-compatible tile-wise k-means caches for any model's tied LM head.

Faithful port of external/CFHead/algorithm/CFHead.py::_build_codebook_and_index
(seed 99, per-tile randperm init from one shared CPU generator, cdist+argmin+mean,
kmeans_iters=10 default). Cache format kept compatible: {"metadata", "codebook",
"index_map"} so the original repo's CFHeadSelector can load our files and vice
versa (cross-checkability for the head-to-head, docs/26).

Qwen3.5-2B: V=248,320, H=2048, tied embeddings -> lm_head == embed_tokens.weight.
T=8 -> N=256 tiles. Per-tile k-means is on [V, 8] slices: cheap on GPU.

Run from repo root:
  python -m tools.build_cfhead_cache --model-dir <model> --out-dir <dir> --clusters 64 4
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
    candidates = ("model.embed_tokens.weight", "embed_tokens.weight", "lm_head.weight")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in candidates:
                if key in f.keys():
                    t = f.get_tensor(key)
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
    V, H = weight.shape
    assert H % tile == 0
    N = H // tile
    codebook = torch.empty((N, clusters, tile), device=device, dtype=torch.float32)
    index_map = torch.empty((V, N), device=device, dtype=torch.uint8)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    w = weight.to(device=device, dtype=torch.float32)
    for tile_id in range(N):
        x = w[:, tile_id * tile:(tile_id + 1) * tile].contiguous()  # [V, tile]
        init_idx = torch.randperm(V, generator=generator)[:clusters].to(device)
        centroids = x.index_select(0, init_idx).contiguous()
        assign = None
        for _ in range(iters):
            dist = torch.cdist(x, centroids, p=2)
            assign = torch.argmin(dist, dim=1)
            new_centroids = torch.zeros_like(centroids)
            counts = torch.bincount(assign, minlength=clusters).float()
            new_centroids.index_add_(0, assign, x)
            nonempty = counts > 0
            new_centroids[nonempty] /= counts[nonempty].unsqueeze(1)
            new_centroids[~nonempty] = centroids[~nonempty]
            centroids = new_centroids.contiguous()
        if assign is None:
            assign = torch.argmin(torch.cdist(x, centroids, p=2), dim=1)
        codebook[tile_id] = centroids
        index_map[:, tile_id] = assign.to(torch.uint8)
        if (tile_id + 1) % max(1, N // 8) == 0 or tile_id == N - 1:
            print(f"  tile {tile_id + 1}/{N}", flush=True)
    return codebook, index_map


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
