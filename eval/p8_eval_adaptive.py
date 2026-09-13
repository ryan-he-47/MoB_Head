# P8: evaluate the adaptive candidate-generation strategy (user design
# 2026-09-07, docs/09 section 6.2): dynamic component scope (ratio/mass) +
# per-component uncertain-bit enumeration, vs the round-32 baseline, on a
# 2048-position dev subset. Reports recall vs native top-1, hit@top4/8,
# rerank agreement, and candidate-count stats (the Pareto cost axis).
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from mob.mob_common import H, MoBHead, ResMLPHead, SwiGLUHead, propose_adaptive

GRID = [
    {"mode": "ratio", "ratio": 0.05, "alpha": 0.15, "floor": 32},
    {"mode": "ratio", "ratio": 0.05, "alpha": 0.25, "floor": 32},
    {"mode": "ratio", "ratio": 0.10, "alpha": 0.15, "floor": 32},
    {"mode": "mass", "mass": 0.95, "alpha": 0.15, "floor": 32},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"E:\Trash_things\llm_codec")
    ap.add_argument("--head", default=r"E:\Trash_things\llm_codec\runs\head_final_wsd.pt")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--width", type=int, default=3072)
    ap.add_argument("--head-type", choices=["mob", "resmlp", "swiglu"], default="mob")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=7300)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--codebook", default="codebook.npz")
    ap.add_argument("--dev-cache", default="fineweb_dev_tokens.npy")
    ap.add_argument("--n-pos", type=int, default=2048)
    ap.add_argument("--out", default=r"E:\Trash_things\llm_codec\runs\p8_adaptive_eval.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    model_dir = args.model_dir or os.path.join(
        args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    cb = np.load(os.path.join(args.data_root, "data", args.codebook))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    CB_M = codes_all.shape[1]
    if args.head_type == "resmlp":
        head = ResMLPHead(k=args.k, width=args.width, depth=args.depth, m=CB_M)
    elif args.head_type == "swiglu":
        head = SwiGLUHead(k=args.k, hidden=args.hidden, blocks=args.blocks, m=CB_M)
    else:
        head = MoBHead(k=args.k, width=args.width, depth=args.depth, m=CB_M)
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    head = head.cuda().float().eval()
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    vid_to_tok = torch.full((1 << CB_M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().cuda()] = \
        torch.arange(len(cb["vertex_id"]), device="cuda")
    E = model.get_input_embeddings().weight

    dev = np.load(os.path.join(args.data_root, "data", args.dev_cache))
    T = 512
    n_ch = args.n_pos // T
    chunks = torch.from_numpy(dev[: n_ch * (T + 1)].reshape(n_ch, T + 1).copy()).long().cuda()
    with torch.no_grad():
        out = model(chunks[:, :-1], output_hidden_states=True)
        hs = out.hidden_states[-1].float().reshape(-1, H)
        logits = out.logits.float()
    V = logits.shape[-1]
    logits = logits.reshape(-1, V)
    nat_top1 = logits.argmax(-1)
    nat_top8 = logits.topk(8, -1).indices

    shifts = (1 << torch.arange(CB_M - 1, -1, -1, device="cuda")).long()
    results = {"n_positions": int(hs.shape[0]), "grid": []}
    # round-32 baseline on the same positions (matched ruler)
    base = {"recall": 0.0, "hit8": 0.0, "hit4": 0.0, "rerank": 0.0}
    for i in range(hs.shape[0]):
        bl, pl = head(hs[i:i + 1])
        lp = F.log_softmax(pl, -1)
        ktop = lp[0].topk(32, -1).indices
        bits = (bl[0, ktop] > 0).long()
        toks = vid_to_tok[(bits * shifts).sum(-1)]
        toks = toks[toks >= 0]
        base["recall"] += float((toks == nat_top1[i]).any().item())
        base["hit8"] += float((toks.unsqueeze(-1) == nat_top8[i]).any().item())
        base["hit4"] += float((toks.unsqueeze(-1) == nat_top8[i, :4]).any().item())
        if toks.numel():
            sc = hs[i].to(E.dtype) @ E[toks].T
            base["rerank"] += float((toks[sc.argmax()] == nat_top1[i]).item())
        if (i + 1) % 512 == 0:
            print(f"baseline {i + 1}/{hs.shape[0]}", flush=True)
    for k in base:
        base[k] /= hs.shape[0]
    results["round32_baseline"] = base

    # augmentation variant: round-32 PLUS flip-enumeration of the top-8
    # components' uncertain bits (the salvageable core of the adaptive idea)
    import os as _os
    AUG_ALPHA = float(_os.environ.get("AUG_ALPHA", "0.15"))
    AUG_UMAX = int(_os.environ.get("AUG_UMAX", "6"))
    AUG_PERCOMP = int(_os.environ.get("AUG_PERCOMP", "64"))
    AUG_TOPN = int(_os.environ.get("AUG_TOPN", "8"))
    aug = {"recall": 0.0, "hit8": 0.0, "hit4": 0.0, "rerank": 0.0}
    aug_counts = []
    for i in range(hs.shape[0]):
        bl, pl = head(hs[i:i + 1])
        lp = F.log_softmax(pl, -1)
        ktop = lp[0].topk(32, -1).indices
        bits = (bl[0, ktop] > 0).long()
        toks = vid_to_tok[(bits * shifts).sum(-1)]
        toks = toks[toks >= 0]
        extra = []
        for k in ktop[:AUG_TOPN].tolist():
            p = bl[0, k].sigmoid()
            unc = ((p - 0.5).abs() <= AUG_ALPHA).nonzero(as_tuple=True)[0]
            u = min(unc.numel(), AUG_UMAX)
            unc = unc[:u]
            if u == 0:
                continue
            base_v = (p > 0.5).long()
            base_v[unc] = 0
            base_vid = (base_v * shifts).sum()
            combos = ((torch.arange(2 ** u, device="cuda").unsqueeze(1)
                       >> torch.arange(u - 1, -1, -1, device="cuda")) & 1).long()
            vids = base_vid + (combos * shifts[unc]).sum(-1)
            t = vid_to_tok[vids]
            extra.append(t[t >= 0][:AUG_PERCOMP])
        if extra:
            toks = torch.unique(torch.cat([toks] + extra))
        aug_counts.append(toks.numel())
        if toks.numel() == 0:
            continue
        aug["recall"] += float((toks == nat_top1[i]).any().item())
        aug["hit8"] += float((toks.unsqueeze(-1) == nat_top8[i]).any().item())
        aug["hit4"] += float((toks.unsqueeze(-1) == nat_top8[i, :4]).any().item())
        sc = hs[i].to(E.dtype) @ E[toks].T
        aug["rerank"] += float((toks[sc.argmax()] == nat_top1[i]).item())
        if (i + 1) % 512 == 0:
            print(f"augment {i + 1}/{hs.shape[0]}", flush=True)
    for k in aug:
        aug[k] /= hs.shape[0]
    c = np.array(aug_counts)
    results["round32_plus_flips"] = {**aug, "alpha": AUG_ALPHA, "aug_topn": AUG_TOPN,
                                     "per_comp_cap": AUG_PERCOMP,
                                     "cand_mean": float(c.mean()),
                                     "cand_p95": float(np.percentile(c, 95))}
    print(f"augment done: {results['round32_plus_flips']}", flush=True)

    for gi, g in enumerate(GRID):
        rec = {"recall": 0.0, "hit8": 0.0, "hit4": 0.0, "rerank": 0.0}
        counts = []
        for i in range(hs.shape[0]):
            bl, pl = head(hs[i:i + 1])
            lp = F.log_softmax(pl, -1)
            toks = propose_adaptive(bl, lp, codes_all, vid_to_tok, **g)
            counts.append(toks.numel())
            if toks.numel() == 0:
                continue
            rec["recall"] += float((toks == nat_top1[i]).any().item())
            rec["hit8"] += float((toks.unsqueeze(-1) == nat_top8[i]).any().item())
            rec["hit4"] += float((toks.unsqueeze(-1) == nat_top8[i, :4]).any().item())
            sc = hs[i].to(E.dtype) @ E[toks].T
            rec["rerank"] += float((toks[sc.argmax()] == nat_top1[i]).item())
            if (i + 1) % 512 == 0:
                print(f"grid {gi} ({g}): {i + 1}/{hs.shape[0]}", flush=True)
        n = hs.shape[0]
        for k in rec:
            rec[k] /= n
        c = np.array(counts)
        results["grid"].append({**g, **rec, "cand_mean": float(c.mean()),
                                "cand_p50": float(np.percentile(c, 50)),
                                "cand_p95": float(np.percentile(c, 95)),
                                "cand_max": int(c.max())})
        print(f"grid {gi} done: {results['grid'][-1]}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
