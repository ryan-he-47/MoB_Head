# Build the semantic-binary-tree codebook for Llama-3.2-1B.
# Recursive 2-means (cosine) bisection of the tied embedding table, 17 levels.
# Artifacts: data/codebook.npz with codes[V,17] uint8, vertex_id[V] int32,
# valid_vertices[2^17] bool, depth_reached[V] uint8, plus validation report.
import json
import time

import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = r"E:\Trash_things\llm_codec\LLM-Research--Llama-3.2-1B\snapshots\master"
OUT_NPZ = r"E:\Trash_things\llm_codec\data\codebook.npz"
OUT_META = r"E:\Trash_things\llm_codec\data\codebook_meta.json"
M = 17
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_embeddings():
    with safe_open(MODEL_DIR + r"\model.safetensors", framework="pt") as f:
        key = [k for k in f.keys() if "embed_tokens" in k]
        assert len(key) == 1, f"unexpected embed keys: {key}"
        w = f.get_tensor(key[0])
    e = w.to(torch.float32).cuda()
    e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-8)
    print(f"embeddings: {tuple(e.shape)} (key={key[0]})")
    return e


def bisect_direction(x):
    """2-means (cosine) on unit-norm rows x:[n,d]; returns split direction w.
    (2-means on cone-shaped LLM embeddings gives lopsided clusters, so it is
    used only to FIND the split axis; the actual split is a median cut.)"""
    n = x.shape[0]
    restarts, iters = (3, 50) if n > 4096 else (1, 20)
    best = None
    for _ in range(restarts):
        i0 = torch.randint(n, (1,)).item()
        c0 = x[i0]
        c1 = x[(1 - x @ c0).argmax()]
        cent = torch.stack([c0, c1])
        assign = None
        for _ in range(iters):
            assign = (x @ cent.T).argmax(dim=1)
            new_cent = cent.clone()
            for j in range(2):
                mask = assign == j
                if mask.any():
                    cj = x[mask].mean(0)
                    new_cent[j] = cj / cj.norm().clamp_min(1e-8)
            if torch.allclose(new_cent, cent, atol=1e-6):
                cent = new_cent
                break
            cent = new_cent
        counts = torch.bincount(assign, minlength=2)
        if counts.min() == 0:
            continue
        inertia = ((x - cent[assign]) ** 2).sum().item()
        if best is None or inertia < best[0]:
            best = (inertia, cent)
    if best is None:  # identical points: any axis does, use first PC substitute
        return torch.ones(x.shape[1], device=x.device)
    return best[1][1] - best[1][0]


def two_means(x):
    """Balanced bisection: project onto the 2-means direction, cut at the
    median (rank-based, tie-safe). Guarantees sizes differ by <= 1, so every
    branch terminates at depth <= ceil(log2(n)) and all codes are unique.
    Bit 0 = lower-projection half (deterministic order)."""
    n = x.shape[0]
    if n == 2:
        return torch.tensor([0, 1], device=x.device)
    w = bisect_direction(x)
    proj = x @ w
    order = torch.argsort(proj, stable=True)
    assign = torch.empty(n, dtype=torch.long, device=x.device)
    assign[order[: n - n // 2]] = 0
    assign[order[n - n // 2 :]] = 1
    return assign


def build(e):
    v = e.shape[0]
    codes = np.zeros((v, M), dtype=np.uint8)
    depth_reached = np.zeros(v, dtype=np.uint8)
    stats = {"early_singleton": 0, "max_depth_hit": 0, "fallback_splits": 0}
    # stack of (indices_cpu, depth)
    stack = [(torch.arange(v, device=e.device), 0)]
    t0 = time.time()
    nodes = 0
    while stack:
        idx, depth = stack.pop()
        nodes += 1
        depth_reached[idx.cpu().numpy()] = depth
        if idx.numel() == 1:
            stats["early_singleton"] += 1
            continue
        if depth == M:
            # should not happen unless duplicate embeddings; leave remaining bits 0
            stats["max_depth_hit"] += idx.numel()
            continue
        assign = two_means(e[idx])
        bits = codes[idx.cpu().numpy()]
        bits[:, depth] = assign.cpu().numpy().astype(np.uint8)
        codes[idx.cpu().numpy()] = bits
        for b in (0, 1):
            child = idx[assign == b]
            if child.numel() > 0:
                stack.append((child, depth + 1))
        if nodes % 20000 == 0:
            print(f"nodes={nodes} stack={len(stack)} elapsed={time.time()-t0:.0f}s")
    print(f"tree done: nodes={nodes} elapsed={time.time()-t0:.0f}s stats={stats}")
    return codes, depth_reached, stats


def pack_vertex_id(codes):
    vid = np.zeros(codes.shape[0], dtype=np.int32)
    for j in range(M):
        vid |= codes[:, j].astype(np.int32) << (M - 1 - j)
    return vid


def validate(e, codes, vid):
    v = codes.shape[0]
    assert len(np.unique(vid)) == v, "vertex ids not unique!"
    print("uniqueness: OK (all 128256 vertices distinct)")

    ham = codes  # already a 0/1 bit array [V, M]
    n_pairs = 200_000
    g = np.random.default_rng(SEED)
    a = g.integers(0, v, n_pairs)
    b = g.integers(0, v, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    ham_d = (ham[a] != ham[b]).sum(1)
    cos_d = 1 - (e[torch.from_numpy(a).cuda()] * e[torch.from_numpy(b).cuda()]).sum(1).cpu().numpy()
    r_tree = np.corrcoef(ham_d, cos_d)[0, 1]

    # random codebook control
    perm = g.permutation(v)
    ham_d_rnd = (ham[perm[a]] != ham[perm[b]]).sum(1)
    r_rnd = np.corrcoef(ham_d_rnd, cos_d)[0, 1]
    print(f"corr(hamming, cosine_distance): tree={r_tree:.4f}  random-control={r_rnd:.4f}")

    # true nearest neighbor hamming distance
    n_probe = 500
    probe = torch.from_numpy(g.integers(0, v, n_probe)).cuda()
    sims = e[probe] @ e.T
    sims[torch.arange(n_probe), probe] = -1
    nn = sims.argmax(1).cpu().numpy()
    probe_np = probe.cpu().numpy()
    d_nn = (ham[probe_np] != ham[nn]).sum(1)
    nn_rnd = perm[nn]
    d_nn_rnd = (ham[perm[probe_np]] != ham[nn_rnd]).sum(1)
    print(f"cosine-NN hamming distance: tree mean={d_nn.mean():.2f}/17  "
          f"random-control mean={d_nn_rnd.mean():.2f}/17")
    return {"corr_tree": float(r_tree), "corr_random": float(r_rnd),
            "nn_hamming_tree": float(d_nn.mean()), "nn_hamming_random": float(d_nn_rnd.mean())}


def main():
    e = load_embeddings()
    codes, depth_reached, stats = build(e)
    vid = pack_vertex_id(codes)
    valid = np.zeros(1 << M, dtype=bool)
    valid[vid] = True
    print(f"valid vertices: {valid.sum()} / {1 << M} (unused {(1 << M) - valid.sum()})")
    report = validate(e, codes, vid)
    np.savez_compressed(
        OUT_NPZ,
        codes=codes,
        vertex_id=vid,
        valid_vertices=valid,
        depth_reached=depth_reached,
    )
    meta = {
        "model": "Llama-3.2-1B (ModelScope LLM-Research mirror)",
        "method": "recursive 2-means (cosine) bisection of tied embedding table",
        "M": M, "seed": SEED, "vocab": int(codes.shape[0]),
        "build_stats": stats, "validation": report,
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"saved {OUT_NPZ} + {OUT_META}")


if __name__ == "__main__":
    main()
