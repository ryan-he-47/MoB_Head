# Gap analysis: decompose the MoB-vs-native PPL gap by (a) teacher entropy
# quartile and (b) the rank of the true next token in the teacher distribution.

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob.mob_common import H, M, MoBHead, mob_mix_logprob


def load_tokens(tok, data_glob, cache_path, eos_id):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    all_ids = []
    for path in sorted(glob.glob(data_glob)):
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for s in range(0, len(lines), 2000):
            for ids in tok(lines[s:s + 2000]).input_ids:
                all_ids.extend(ids + [eos_id])
    arr = np.asarray(all_ids, dtype=np.int32)
    np.save(cache_path, arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r".")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--head", default=r".\runs\p2_local_10m_e3\head_final.pt")
    ap.add_argument("--out", default=r".\runs\gap_analysis.json")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-chunks", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pos-chunk", type=int, default=64)
    ap.add_argument("--k", type=int, default=32, help="mixture components (must match checkpoint)")
    ap.add_argument("--width", type=int, default=2458, help="head MLP width (must match checkpoint)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_dir = args.model_dir or os.path.join(
        args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16).cuda().eval()
    head = MoBHead(k=args.k, width=args.width).cuda()
    head.load_state_dict(torch.load(args.head))
    head.eval()
    cb = np.load(os.path.join(args.data_root, "data", "codebook.npz"))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()

    dev = load_tokens(
        tok, os.path.join(args.data_root, "data", "babylm_data", "babylm_dev", "*.dev"),
        os.path.join(args.data_root, "data", "babylm_dev_tokens.npy"), tok.eos_token_id)
    T = args.seq_len
    n_chunks = min(args.n_chunks, len(dev) // (T + 1))
    chunks = torch.from_numpy(dev[: n_chunks * (T + 1)].reshape(n_chunks, T + 1).copy()).long().cuda()

    ENT, RANK, NNLL, MNLL = [], [], [], []
    with torch.no_grad():
        for i in range(0, n_chunks, args.batch):
            x = chunks[i:i + args.batch]
            out = model(x[:, :-1], output_hidden_states=True)
            tgt = x[:, 1:]
            h = out.hidden_states[-1].float().reshape(-1, H)
            logits = out.logits.float()
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            ent = -(p * logp).sum(-1).reshape(-1)
            lg = logits.reshape(-1, logits.shape[-1])
            tflat = tgt.reshape(-1, 1)
            logp_tgt = logp.reshape(-1, logp.shape[-1]).gather(-1, tflat).squeeze(-1)
            rank = (lg > lg.gather(-1, tflat)).sum(-1) + 1

            bl, pl = head(h)
            log_pi = F.log_softmax(pl, dim=-1)
            codes_true = codes_all[tgt.reshape(-1)].unsqueeze(1)
            logp_true = mob_mix_logprob(bl, log_pi, codes_true)[:, 0]
            n = h.shape[0]
            logZs = []
            for s in range(0, n, args.pos_chunk):
                b = bl[s:s + args.pos_chunk]
                term = torch.einsum("vm,nkm->nvk", codes_all, b)
                base = F.logsigmoid(-b).sum(-1).unsqueeze(1)
                mix = torch.logsumexp(log_pi[s:s + args.pos_chunk].unsqueeze(1) + term + base, dim=-1)
                logZs.append(torch.logsumexp(mix, dim=-1))
            mob_nll = torch.cat(logZs) - logp_true

            ENT.append(ent.cpu())
            RANK.append(rank.cpu())
            NNLL.append((-logp_tgt).cpu())
            MNLL.append(mob_nll.cpu())
            print(f"batch {i // args.batch + 1}/{(n_chunks + args.batch - 1) // args.batch}")

    ent = torch.cat(ENT).numpy()
    rank = torch.cat(RANK).numpy()
    nnll = torch.cat(NNLL).numpy()
    mnll = torch.cat(MNLL).numpy()
    gap = mnll - nnll
    total_gap = gap.sum()

    def bucket_stats(mask):
        return {
            "n_positions": int(mask.sum()),
            "frac_positions": float(mask.mean()),
            "mean_native_nll": float(nnll[mask].mean()),
            "mean_mob_nll": float(mnll[mask].mean()),
            "mean_gap": float(gap[mask].mean()),
            "gap_share": float(gap[mask].sum() / total_gap),
        }

    rank_buckets = {
        "rank_1": rank == 1,
        "rank_2_10": (rank >= 2) & (rank <= 10),
        "rank_11_40": (rank >= 11) & (rank <= 40),
        "rank_41_1024": (rank >= 41) & (rank <= 1024),
        "rank_>1024": rank > 1024,
    }
    qs = np.quantile(ent, [0.25, 0.5, 0.75])
    ent_buckets = {
        f"ent_q1(<{qs[0]:.2f})": ent <= qs[0],
        f"ent_q2({qs[0]:.2f}-{qs[1]:.2f})": (ent > qs[0]) & (ent <= qs[1]),
        f"ent_q3({qs[1]:.2f}-{qs[2]:.2f})": (ent > qs[1]) & (ent <= qs[2]),
        f"ent_q4(>{qs[2]:.2f})": ent > qs[2],
    }
    result = {
        "n_positions": int(len(gap)),
        "overall": {
            "mean_native_nll": float(nnll.mean()),
            "mean_mob_nll": float(mnll.mean()),
            "mean_gap": float(gap.mean()),
            "median_gap": float(np.median(gap)),
        },
        "by_true_token_rank": {k: bucket_stats(v) for k, v in rank_buckets.items()},
        "by_teacher_entropy": {k: bucket_stats(v) for k, v in ent_buckets.items()},
    }
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
