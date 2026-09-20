"""HiDRA-style codebook construction (ICLR 2026 TTU workshop paper) vs our
2-means-direction tree — construction-only head-to-head, CPU/numpy.

HiDRA paper construction (their section 3 + 4.1): "a Fisher LDA-motivated
median-split procedure on the input embedding matrix" — recursively:
get two classes, Fisher direction d* = S_w^{-1} (mu_R - mu_L), project,
median cut. Our builder (tools/build_codebook.py) is IDENTICAL except the
direction is the raw 2-means centroid difference mu_R - mu_L (i.e. LDA with
S_w = I). So the scientific delta being measured here is: does within-class
whitening of the split direction improve the resulting binary codes?

Practical adaptations (documented in docs/29):
  - All computation in a global top-D' PCA subspace of the (unit-normed)
    embedding table: per-node 2048x2048 covariances are ill-posed for small
    nodes and infeasible at scale. PCA + shrinkage is the standard practical
    form of LDA. Rows are re-normalized to unit norm inside the subspace so
    the cosine geometry matches our builder's.
  - lambda shrinkage: lam = 0.1 * trace(S_w)/D' + 1e-6.
  - Nodes of size 2 are split directly; degenerate 2-means falls back to the
    raw mean difference (which then equals our direction).

Arms:
  lda : per-node 2-means classes -> Fisher direction -> median cut
  pca : per-node local PC1 direction -> median cut (classical baseline)

Outputs: one npz per arm (codes + per-node hyperplanes for a later routing
eval), plus a json report with semantic metrics + cross-agreement vs our
existing codebook.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from safetensors import safe_open


def load_embeddings(model_dir, device="cpu"):
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    assert files, f"no safetensors under {model_dir}"
    key, path = None, None
    for p in files:
        with safe_open(p, framework="pt") as f:
            ks = [k for k in f.keys() if "embed_tokens" in k]
        if ks:
            key, path = ks[0], p
            break
    assert key, f"no embed_tokens key in {files}"
    with safe_open(path, framework="pt") as f:
        w = f.get_tensor(key)
    e = w.to(torch.float32).numpy()
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
    print(f"embeddings: {e.shape} (key={key}, shard={os.path.basename(path)})")
    return e.astype(np.float32)


def pca_subspace(e, dim, out_path=None):
    t0 = time.time()
    mean = e.mean(0, keepdims=True)
    ec = e - mean
    cov = (ec.T @ ec) / e.shape[0]
    w, V = np.linalg.eigh(cov)
    basis = V[:, -dim:][:, ::-1].astype(np.float32)  # [d, dim], descending variance
    ev = w[::-1][:dim]
    z = (ec @ basis).astype(np.float32)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    print(f"pca: dim={dim} var_captured={float(ev.sum() / w.sum()):.4f} "
          f"({time.time()-t0:.0f}s)")
    if out_path:
        np.savez(out_path, mean=mean.astype(np.float32), basis=basis,
                 eigenvalues=ev.astype(np.float32))
    return z, mean, basis, ev


def kmeans2_cosine(x, rng, iters=25, restarts=2):
    """2-means under cosine on unit-norm rows. Returns (labels, centers) or None."""
    n = x.shape[0]
    best = None
    for _ in range(restarts):
        i0 = rng.integers(n)
        c0 = x[i0].copy()
        c1 = x[np.argmin(x @ c0)].copy()   # farthest in cosine = min dot
        cent = np.stack([c0, c1])
        a = np.zeros(n, dtype=np.int64)
        for _ in range(iters):
            a = (x @ cent.T).argmax(1)
            nc = cent.copy()
            for j in (0, 1):
                m = a == j
                if m.any():
                    cj = x[m].mean(0)
                    nrm = np.linalg.norm(cj)
                    if nrm > 1e-8:
                        nc[j] = cj / nrm
            if np.allclose(nc, cent, atol=1e-6):
                cent = nc
                break
            cent = nc
        cnt = np.bincount(a, minlength=2)
        if cnt.min() == 0:
            continue
        inertia = (1 - (x * cent[a]).sum(1)).sum()
        if best is None or inertia < best[0]:
            best = (inertia, a.copy(), cent)
    if best is None:
        return None
    return best[1], best[2]


def node_direction(x, arm, rng, lam_frac=0.1):
    """Split direction for a node with unit-norm rows x:[n, D']."""
    if arm == "pca":
        xc = x - x.mean(0)
        cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        d = V[:, -1]
        nrm = np.linalg.norm(d)
        return d / nrm if nrm > 1e-12 else None
    # lda
    res = kmeans2_cosine(x, rng)
    if res is None:
        return None
    a, _cent = res
    muL = x[a == 0].mean(0)
    muR = x[a == 1].mean(0)
    d_raw = muR - muL
    if arm == "means":
        nrm = np.linalg.norm(d_raw)
        return (d_raw / nrm).astype(np.float32) if nrm > 1e-12 else None
    Sw = np.zeros((x.shape[1], x.shape[1]), dtype=np.float32)
    for j in (0, 1):
        m = a == j
        if m.sum() > 1:
            xc = x[m] - x[m].mean(0)
            Sw += xc.T @ xc
    Sw /= max(1, x.shape[0] - 2)
    lam = lam_frac * (np.trace(Sw) / x.shape[1]) + 1e-6
    try:
        d = np.linalg.solve(Sw + lam * np.eye(x.shape[1], dtype=np.float32),
                            d_raw.astype(np.float32))
    except np.linalg.LinAlgError:
        d = d_raw
    nrm = np.linalg.norm(d)
    if nrm < 1e-12:
        return None
    return (d / nrm).astype(np.float32)


def build_tree(z, arm, m_bits, seed):
    """Median-cut tree in subspace z. Returns codes [V,M], node arrays."""
    v = z.shape[0]
    rng = np.random.default_rng(seed)
    codes = np.zeros((v, m_bits), dtype=np.uint8)
    nodes = []  # (depth, path, size, dir[D'] or None, bias, child0, child1)
    stack = [(np.arange(v), 0, 0)]  # (indices, depth, path)
    my_node_idx = []
    t0 = time.time()
    n_done = 0
    while stack:
        idx, depth, path = stack.pop()
        n = idx.shape[0]
        node_id = len(nodes)
        nodes.append([depth, path, n, None, 0.0, -1, -1])
        if n == 1 or depth == m_bits:
            continue
        if n == 2:
            assign = np.array([0, 1])
            d2 = z[idx[1]] - z[idx[0]]
            nrm = np.linalg.norm(d2)
            if nrm > 1e-12:
                d = (d2 / nrm).astype(np.float32)
                proj2 = z[idx] @ d
                nodes[node_id][3] = d
                nodes[node_id][4] = -float(0.5 * (proj2[0] + proj2[1]))
        else:
            d = node_direction(z[idx], arm, rng)
            if d is None:
                # degenerate: arbitrary median cut on first coordinate
                proj = z[idx][:, 0]
            else:
                proj = z[idx] @ d
            order = np.argsort(proj, kind="stable")
            assign = np.empty(n, dtype=np.int64)
            assign[order[: n - n // 2]] = 0
            assign[order[n - n // 2:]] = 1
            if d is not None:
                thr = 0.5 * (proj[order[n - n // 2 - 1]] + proj[order[n - n // 2]])
                nodes[node_id][3] = d
                nodes[node_id][4] = -float(thr)
        codes[idx, depth] = assign.astype(np.uint8)
        for b in (0, 1):
            child = idx[assign == b]
            if child.shape[0] > 0:
                cpath = path | (int(b) << (m_bits - 1 - depth))
                stack.append((child, depth + 1, cpath))
                nodes[node_id][5 + b] = -2  # marker: has child (id filled later)
        n_done += 1
        if n_done % 20000 == 0:
            print(f"  nodes={n_done} stack={len(stack)} elapsed={time.time()-t0:.0f}s")
    print(f"tree({arm}): nodes={n_done} elapsed={time.time()-t0:.0f}s")
    return codes, nodes


def metrics(e, codes, ours_codes, seed):
    v = codes.shape[0]
    g = np.random.default_rng(seed)
    n_pairs = 200_000
    a = g.integers(0, v, n_pairs)
    b = g.integers(0, v, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    ham_d = (codes[a] != codes[b]).sum(1)
    cos_d = 1 - (e[a] * e[b]).sum(1)
    r_tree = float(np.corrcoef(ham_d, cos_d)[0, 1])

    n_probe = 500
    probe = g.integers(0, v, n_probe)
    sims = e[probe] @ e.T
    sims[np.arange(n_probe), probe] = -1
    nn = sims.argmax(1)
    d_nn = (codes[probe] != codes[nn]).sum(1)

    out = {
        "corr_hamming_cosine": r_tree,
        "nn_hamming_mean": float(d_nn.mean()),
        "nn_hamming_median": float(np.median(d_nn)),
    }
    if ours_codes is not None:
        xor = (codes != ours_codes).sum(1)
        out["vs_ours_hamming_mean"] = float(xor.mean())
        out["vs_ours_exact_match"] = float((xor == 0).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=r"models/Qwen3.5-2B")
    ap.add_argument("--ours", default=r"data/codebook_qwen35.npz")
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--arms", default="lda,pca")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default=r"data/codebook_qwen35")
    ap.add_argument("--report", default=r"data/hidra_build_report.json")
    ap.add_argument("--load-basis", default=None,
                    help="path to a saved *_pca_basis.npz to skip PCA recompute")
    args = ap.parse_args()

    e = load_embeddings(args.model_dir)
    v, m_bits = e.shape[0], int(np.ceil(np.log2(e.shape[0])))
    print(f"vocab {v} -> M={m_bits} bits")

    ours_codes = None
    if os.path.exists(args.ours):
        ours_codes = np.load(args.ours)["codes"]
        print(f"ours loaded: {ours_codes.shape}")

    basis_path = args.out_prefix + "_pca_basis.npz"
    if args.load_basis and os.path.exists(args.load_basis):
        b = np.load(args.load_basis)
        mean, basis, ev = b["mean"], b["basis"], b["eigenvalues"]
        ec = e - mean
        z = (ec @ basis).astype(np.float32)
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
        print(f"pca basis loaded from {args.load_basis}")
    else:
        z, mean, basis, ev = pca_subspace(e, args.pca_dim, out_path=basis_path)

    report = {"model": args.model_dir, "pca_dim": args.pca_dim,
              "var_captured": float(ev.sum()), "seed": args.seed, "arms": {}}
    if os.path.exists(args.report):
        with open(args.report, encoding="utf-8") as f:
            report = json.load(f)
        report.setdefault("arms", {})
    if ours_codes is not None:
        report["arms"]["ours_2means"] = metrics(e, ours_codes, None, args.seed)

    for arm in args.arms.split(","):
        codes, nodes = build_tree(z, arm, m_bits, args.seed)
        vid = np.zeros(v, dtype=np.int64)
        for j in range(m_bits):
            vid |= codes[:, j].astype(np.int64) << (m_bits - 1 - j)
        assert len(np.unique(vid)) == v, f"{arm}: vertex ids not unique!"
        nd = np.zeros((len(nodes), z.shape[1]), dtype=np.float32)
        nb = np.zeros(len(nodes), dtype=np.float32)
        npath = np.zeros(len(nodes), dtype=np.uint32)
        ndepth = np.zeros(len(nodes), dtype=np.uint8)
        nsize = np.zeros(len(nodes), dtype=np.int32)
        for i, (dep, pth, sz, d, b, _c0, _c1) in enumerate(nodes):
            npath[i], ndepth[i], nsize[i], nb[i] = pth, dep, sz, b
            if d is not None:
                nd[i] = d
        out = f"{args.out_prefix}_hidra_{arm}.npz"
        np.savez_compressed(out, codes=codes, vertex_id=vid.astype(np.int32),
                            node_dir=nd, node_bias=nb, node_path=npath,
                            node_depth=ndepth, node_size=nsize)
        report["arms"][arm] = metrics(e, codes, ours_codes, args.seed)
        print(arm, json.dumps(report["arms"][arm], indent=1))

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved report -> {args.report}")


if __name__ == "__main__":
    main()
