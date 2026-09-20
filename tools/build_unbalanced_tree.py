"""Unbalanced semantic tree exploration (docs/31 task 2): what does the
embedding-bisection tree look like if we do NOT enforce the median-balanced
cut? Pure 2-means natural splits, recursed to singletons (or a depth cap).

Builds ONE full tree (min-leaf 1), then derives truncated-tree statistics for
min-leaf thresholds {1,2,4,8,16,32,64} by cutting each token's path at the
first ancestor whose size <= threshold (split decisions are independent of the
threshold, so truncation == early stopping).

Answers:
  - how long do codes get (depth/length distribution, max depth, chain risk)?
  - how imbalanced are the splits (child-size ratio distribution)?
  - geometry: does shared-prefix length track embedding similarity
    (correlation, nearest-neighbor prefix depth)?
  - at each truncation threshold: code-length distribution.

Practical adaptation: construction in a global PCA-64 subspace (cost); rows
re-normalized so splits are cosine 2-means, matching the main builder's
geometry. CPU/numpy only.

Node table is keyed by (depth, path): the bit-0 child keeps the parent's path
value (appended bit is 0), so path alone does NOT identify a node.
"""

import argparse
import json
import time

import numpy as np

from tools.hidra_codebook import load_embeddings, pca_subspace, kmeans2_cosine


def build_full_tree(z, depth_cap, seed, max_nodes=3_000_000):
    v = z.shape[0]
    rng = np.random.default_rng(seed)
    codes = np.zeros((v, depth_cap), dtype=np.uint8)
    tok_leaf_depth = np.zeros(v, dtype=np.int16)
    nodes = {}  # (depth, path) -> size
    stack = [(np.arange(v), 0, 0)]
    t0 = time.time()
    n_degenerate = 0
    n_capped = 0
    while stack:
        idx, depth, path = stack.pop()
        n = idx.shape[0]
        nodes[(depth, path)] = n
        if n == 1:
            tok_leaf_depth[idx[0]] = depth
            continue
        if depth >= depth_cap:
            tok_leaf_depth[idx] = depth
            n_capped += n
            continue
        res = kmeans2_cosine(z[idx], rng, iters=15,
                             restarts=2 if n > 8192 else 1)
        if res is None:
            tok_leaf_depth[idx] = depth  # identical points: forced leaf
            n_degenerate += n
            continue
        assign, _cent = res
        codes[idx, depth] = assign.astype(np.uint8)
        stack.append((idx[assign == 0], depth + 1, path))
        stack.append((idx[assign == 1], depth + 1,
                      path | (1 << (depth_cap - 1 - depth))))
        if len(nodes) > max_nodes:
            raise RuntimeError(f"node explosion: >{max_nodes} nodes (chain risk)")
        if len(nodes) % 200000 == 0:
            print(f"  nodes={len(nodes)} stack={len(stack)} "
                  f"elapsed={time.time()-t0:.0f}s")
    print(f"full tree: nodes={len(nodes)} elapsed={time.time()-t0:.0f}s "
          f"degenerate_tokens={n_degenerate} capped_tokens={n_capped}")
    return codes, tok_leaf_depth, nodes


def prefix_of(path, to_depth, depth_cap):
    if to_depth == 0:
        return 0
    mask = ((1 << to_depth) - 1) << (depth_cap - to_depth)
    return path & mask


def truncated_lengths(tok_codes, tok_leaf_depth, nodes, threshold, depth_cap):
    """Each token's code length if we stop at the first ancestor with
    size <= threshold."""
    stop = set()
    for (depth, path), size in nodes.items():
        if size <= threshold:
            if depth == 0:
                stop.add((0, 0))
            else:
                parent = (depth - 1, prefix_of(path, depth - 1, depth_cap))
                if nodes[parent] > threshold:
                    stop.add((depth, path))
    v = tok_codes.shape[0]
    lengths = np.empty(v, dtype=np.int16)
    for t in range(v):
        d = int(tok_leaf_depth[t])
        bits = tok_codes[t, :d]
        path = 0
        out = d
        for j in range(d):
            if bits[j]:
                path |= (1 << (depth_cap - 1 - j))
            if (j + 1, path) in stop:
                out = j + 1
                break
        lengths[t] = out
    return lengths


def geometry_metrics(e, tok_codes, tok_leaf_depth, seed):
    v = e.shape[0]
    g = np.random.default_rng(seed)

    def shared_prefix(a, b):
        da = min(int(tok_leaf_depth[a]), int(tok_leaf_depth[b]))
        eq = tok_codes[a, :da] == tok_codes[b, :da]
        if eq.all():
            return da
        return int(np.argmin(eq))

    n_pairs = 20_000
    a = g.integers(0, v, n_pairs)
    b = g.integers(0, v, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    sp = np.array([shared_prefix(x, y) for x, y in zip(a, b)], dtype=np.float32)
    cos = (e[a] * e[b]).sum(1)
    r = float(np.corrcoef(sp, cos)[0, 1])

    n_probe = 500
    probe = g.integers(0, v, n_probe)
    sims = e[probe] @ e.T
    sims[np.arange(n_probe), probe] = -1
    nn = sims.argmax(1)
    nn_sp = np.array([shared_prefix(x, y) for x, y in zip(probe, nn)])
    rnd = g.integers(0, v, n_probe)
    rnd_sp = np.array([shared_prefix(x, y) for x, y in zip(probe, rnd)])
    return {
        "corr_shared_prefix_cosine": r,
        "nn_shared_prefix_mean": float(nn_sp.mean()),
        "nn_shared_prefix_median": float(np.median(nn_sp)),
        "random_shared_prefix_mean": float(rnd_sp.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=r"models/Qwen3.5-2B")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--depth-cap", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thresholds", default="1,2,4,8,16,32,64")
    ap.add_argument("--out", default=r"data/tree_unbalanced_qwen35.npz")
    ap.add_argument("--report", default=r"data/tree_unbalanced_report.json")
    args = ap.parse_args()

    e = load_embeddings(args.model_dir)
    z, _m, _b, _ev = pca_subspace(e, args.pca_dim)

    codes, leaf_depth, nodes = build_full_tree(z, args.depth_cap, args.seed)

    report = {"model": args.model_dir, "pca_dim": args.pca_dim,
              "depth_cap": args.depth_cap, "seed": args.seed,
              "full_tree": {
                  "n_nodes": len(nodes),
                  "depth_mean": float(leaf_depth.mean()),
                  "depth_median": float(np.median(leaf_depth)),
                  "depth_p90": float(np.percentile(leaf_depth, 90)),
                  "depth_p99": float(np.percentile(leaf_depth, 99)),
                  "depth_max": int(leaf_depth.max()),
                  "depth_hist": np.bincount(leaf_depth,
                                            minlength=args.depth_cap + 1).tolist(),
              },
              "imbalance": None, "thresholds": {}, "geometry": None}

    ratios = []
    for (depth, path), size in nodes.items():
        if size <= 1 or depth >= args.depth_cap:
            continue
        c0 = nodes.get((depth + 1, path))
        c1 = nodes.get((depth + 1, path | (1 << (args.depth_cap - 1 - depth))))
        if c0 and c1:
            ratios.append(max(c0, c1) / min(c0, c1))
    if ratios:
        ratios = np.array(ratios)
        report["imbalance"] = {
            "n_internal": int(ratios.size),
            "ratio_median": float(np.median(ratios)),
            "ratio_p90": float(np.percentile(ratios, 90)),
            "ratio_p99": float(np.percentile(ratios, 99)),
            "ratio_max": float(ratios.max()),
            "frac_worse_than_10x": float((ratios >= 10).mean()),
        }
        print("imbalance:", json.dumps(report["imbalance"], indent=1))

    for t in [int(x) for x in args.thresholds.split(",")]:
        t0 = time.time()
        lengths = truncated_lengths(codes, leaf_depth, nodes, t, args.depth_cap)
        leaves, counts = np.unique(lengths, return_counts=True)
        report["thresholds"][str(t)] = {
            "code_len_mean": float(lengths.mean()),
            "code_len_median": float(np.median(lengths)),
            "code_len_p90": float(np.percentile(lengths, 90)),
            "code_len_max": int(lengths.max()),
            "hist": {int(l): int(c) for l, c in zip(leaves, counts)},
        }
        print(f"threshold {t}: mean_len={lengths.mean():.2f} "
              f"max={lengths.max()} ({time.time()-t0:.0f}s)")

    report["geometry"] = geometry_metrics(e, codes, leaf_depth, args.seed)
    print("geometry:", json.dumps(report["geometry"], indent=1))

    np.savez_compressed(args.out, codes=codes, leaf_depth=leaf_depth)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved {args.out} + {args.report}")


if __name__ == "__main__":
    main()
