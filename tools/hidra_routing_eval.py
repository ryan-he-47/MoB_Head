"""HiDRA-style routing evaluation (docs/29 task 1, GPU part — run when GPU free).

Routes teacher-forced hidden states through the per-node hyperplanes of a
tree codebook (built by tools/hidra_codebook.py) and measures:

  - bit error rate per depth: pred bit vs the TRUE next token's path bit;
  - greedy-leaf fidelity: token at the greedily-routed leaf == native top-1
    (HiDRA's full-depth beam-1 operating point) and == gold next token;
  - beam-k containment @ depth l: true token (and native top-1) inside the
    union of subtree token sets of the top-k beam paths, plus candidate count
    (beam * V/2^l expectation vs measured).

Hyperplanes are built in the PCA-D' subspace; they are lifted back to the full
hidden space via d_full = B @ d_pca, b_full = b - mean @ d_full (exact for the
subspace component; the orthogonal complement of h is ignored, same as the
construction). h extraction follows train/p2_train.py: hidden_states[-1].

Node token-set queries use the sorted-vertex-id trick: tokens under a depth-l
prefix form a contiguous range in vid-sorted order (binary search).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tree(npz_path, basis):
    cb = np.load(npz_path)
    codes = cb["codes"]
    v, m_bits = codes.shape
    vid = cb["vertex_id"].astype(np.int64)
    nd, nb = cb["node_dir"], cb["node_bias"].astype(np.float64)
    ndepth, npath, nsize = cb["node_depth"], cb["node_path"].astype(np.int64), cb["node_size"]
    # lift directions to full space
    mean = basis["mean"].reshape(-1)
    B = basis["basis"]                       # [d_full, D']
    dir_full = (B @ nd.T).T                  # [n_nodes, d_full]
    bias_full = nb - dir_full @ mean
    node_idx = {(int(ndepth[i]), int(npath[i])): i for i in range(len(ndepth))}
    has_dir = np.abs(nd).sum(1) > 0
    order = np.argsort(vid, kind="stable")
    return {
        "codes": codes, "vid": vid, "m_bits": m_bits, "v": v,
        "dir": dir_full, "bias": bias_full, "node_idx": node_idx,
        "has_dir": has_dir, "nsize": nsize, "ndepth": ndepth,
        "sort": order, "sorted_vid": vid[order],
    }


def tokens_under(tree, depth, prefix):
    lo = prefix << (tree["m_bits"] - depth)
    hi = (prefix + 1) << (tree["m_bits"] - depth)
    a = np.searchsorted(tree["sorted_vid"], lo)
    b = np.searchsorted(tree["sorted_vid"], hi)
    return tree["sort"][a:b]


def route_eval(tree, h, gold, native_top1, depths, beams):
    """h [n,d] fp64-ish fp32 numpy; gold/native_top1 [n] int."""
    m_bits = tree["m_bits"]
    n = h.shape[0]
    codes = tree["codes"]
    node_idx = tree["node_idx"]
    max_d = max(depths) if depths else m_bits

    bit_err = np.zeros(m_bits, dtype=np.int64)
    bit_cnt = np.zeros(m_bits, dtype=np.int64)
    greedy_leaf = np.zeros(n, dtype=np.int64)
    first_err = np.full(n, -1, dtype=np.int64)
    contain = {(d, k): [0, 0, 0] for d in depths for k in beams}  # gold, native, cand_sum

    for i in range(n):
        hi = h[i]
        g = int(gold[i])
        true_bits = codes[g]
        # --- greedy walk + bit errors along the true path ---
        path = 0
        pred_path = 0
        err_seen = False
        for d in range(m_bits):
            j = node_idx.get((d, pred_path))
            if j is None or not tree["has_dir"][j]:
                break
            s = float(hi @ tree["dir"][j] + tree["bias"][j])
            pb = 1 if s > 0 else 0
            if d < m_bits:
                bit_cnt[d] += 1
                if pb != int(true_bits[d]) and not err_seen:
                    bit_err[d] += 1
                    err_seen = True
                    first_err[i] = d
                elif pb != int(true_bits[d]):
                    bit_err[d] += 1
            if pb:
                pred_path |= (1 << (m_bits - 1 - d))
        # leaf token of the greedy path = token whose vid == pred_path
        a = np.searchsorted(tree["sorted_vid"], pred_path)
        if a < tree["v"] and tree["sorted_vid"][a] == pred_path:
            greedy_leaf[i] = tree["sort"][a]
        else:
            greedy_leaf[i] = -1
        # --- beam containment at requested depths ---
        if depths:
            beam = [(0.0, 0)]  # (cum logp, path) at current depth
            for d in range(1, max_d + 1):
                cand = []
                for lp, pth in beam:
                    j = node_idx.get((d - 1, pth))
                    if j is None or not tree["has_dir"][j]:
                        cand.append((lp, pth))  # leaf already: carries on
                        continue
                    s = float(hi @ tree["dir"][j] + tree["bias"][j])
                    lp0 = lp + np.log1p(np.exp(-abs(s))) + min(s, 0.0)  # log sigmoid
                    # log sigmoid(s) for bit1, log sigmoid(-s) for bit0
                    l1 = -np.log1p(np.exp(-s))
                    l0 = -np.log1p(np.exp(s))
                    cand.append((lp + l1, pth | (1 << (m_bits - d))))
                    cand.append((lp + l0, pth))
                cand.sort(key=lambda x: -x[0])
                beam = cand[: max(beams)]
                if d in depths:
                    for k in beams:
                        gold_hit = nat_hit = 0
                        csum = 0
                        for _lp, pth in beam[:k]:
                            toks = tokens_under(tree, d, pth)
                            csum += toks.shape[0]
                            if gold_hit == 0 and (toks == g).any():
                                gold_hit = 1
                            if nat_hit == 0 and (toks == int(native_top1[i])).any():
                                nat_hit = 1
                        c = contain[(d, k)]
                        c[0] += gold_hit
                        c[1] += nat_hit
                        c[2] += csum
        if (i + 1) % 2048 == 0:
            print(f"  {i+1}/{n} positions")
    out = {
        "bit_err_per_depth": (bit_err / np.maximum(bit_cnt, 1)).round(4).tolist(),
        "greedy_leaf_eq_native_top1": float((greedy_leaf == native_top1).mean()),
        "greedy_leaf_eq_gold": float((greedy_leaf == gold).mean()),
        "first_err_depth_mean": float(first_err[first_err >= 0].mean())
        if (first_err >= 0).any() else None,
        "containment": {
            f"d{d}_b{k}": {"gold": c[0] / n, "native": c[1] / n,
                           "cand_mean": c[2] / n}
            for (d, k), c in sorted(contain.items())
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=r"models/Qwen3.5-2B")
    ap.add_argument("--tokens", default=r"data/qwen35/qwen35_dev_mix_tokens.npy")
    ap.add_argument("--basis", default=r"data/codebook_qwen35_pca_basis.npz")
    ap.add_argument("--codebooks", required=True,
                    help="comma name=path list, e.g. means=data/a.npz,lda=data/b.npz")
    ap.add_argument("--positions", type=int, default=8192)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--depths", default="4,8,12,16,18")
    ap.add_argument("--beams", default="1,2,4,8")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", default=r"runs/hidra_routing_report.json")
    args = ap.parse_args()

    basis = np.load(args.basis)
    depths = [int(x) for x in args.depths.split(",")]
    beams = [int(x) for x in args.beams.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ids = np.load(args.tokens)
    T = args.seq_len
    n_chunks = min(len(ids) // (T + 1),
                   int(np.ceil(args.positions * args.stride / T)))
    hs, golds = [], []
    t0 = time.time()
    with torch.no_grad():
        for c in range(n_chunks):
            chunk = torch.from_numpy(ids[c * T: (c + 1) * T + 1][None, :]).long().cuda()
            out = model(chunk[:, :-1], output_hidden_states=True)
            h = out.hidden_states[-1][0].float()
            hs.append(h.cpu().numpy())
            golds.append(chunk[0, 1:].cpu().numpy())
            if (c + 1) % 16 == 0:
                print(f"extract {c+1}/{n_chunks} ({time.time()-t0:.0f}s)")
    h = np.concatenate(hs, 0)[:: args.stride][: args.positions]
    gold = np.concatenate(golds, 0)[:: args.stride][: args.positions]
    print(f"h {h.shape}, positions {h.shape[0]} ({time.time()-t0:.0f}s)")

    with torch.no_grad():
        w = model.get_output_embeddings().weight.float()
        native_top1 = (torch.from_numpy(h).cuda() @ w.T).argmax(-1).cpu().numpy()
    del model
    torch.cuda.empty_cache()

    report = {"tokens": args.tokens, "positions": int(h.shape[0]),
              "stride": args.stride, "depths": depths, "beams": beams, "arms": {}}
    for spec in args.codebooks.split(","):
        name, path = spec.split("=", 1)
        tree = load_tree(path, basis)
        report["arms"][name] = route_eval(tree, h, gold, native_top1, depths, beams)
        print(name, json.dumps(report["arms"][name], indent=1)[:600])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
