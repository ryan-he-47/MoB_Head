# Semantic eval for a MoB head on FineWeb-Edu dev (user request 2026-09-02):
#   - top-k recall (k=1,8,32) of the MoB mixture distribution, vs (a) the
#     ground-truth next token and (b) the native head's top-1 prediction
#   - normalized PPL (same math as p3_eval.py) side by side with native
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob_common import H, M, MoBHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"E:\Trash_things\llm_codec")
    ap.add_argument("--head", default=r"E:\Trash_things\llm_codec\runs\head_latest_wsd.pt")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--width", type=int, default=3072)
    ap.add_argument("--n-chunks", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pos-chunk", type=int, default=16,
                    help="positions per sub-chunk for the [pos,V,K] pass (K=128: 16 ~= 1GB)")
    ap.add_argument("--out", default=r"E:\Trash_things\llm_codec\runs\wsd_semantic_eval.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_dir = os.path.join(args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    head = MoBHead(k=args.k, width=args.width).cuda()
    head.load_state_dict(torch.load(args.head))
    head.eval()
    cb = np.load(os.path.join(args.data_root, "data", "codebook.npz"))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()  # [V,17]

    dev = np.load(os.path.join(args.data_root, "data", "fineweb_dev_tokens.npy"))
    T = 512
    n_chunks = min(args.n_chunks, len(dev) // (T + 1))
    chunks = torch.from_numpy(dev[: n_chunks * (T + 1)].reshape(n_chunks, T + 1).copy()).long().cuda()

    rank_gt, rank_nat, mnll, nnll = [], [], [], []
    cov_nat8, mass_nat8, rerank_hit = [], [], []
    V = codes_all.shape[0]
    with torch.no_grad():
        for i in range(0, n_chunks, args.batch):
            x = chunks[i:i + args.batch]
            out = model(x[:, :-1], output_hidden_states=True)
            tgt = x[:, 1:].reshape(-1)
            h = out.hidden_states[-1].float().reshape(-1, H)
            logits = out.logits.float()
            nnll.append(F.cross_entropy(logits.reshape(-1, V), tgt, reduction="none"))
            logits_flat = logits.reshape(-1, V)
            nat_top1 = logits_flat.argmax(-1)
            nat_top8 = logits_flat.topk(8, dim=-1).indices               # [n,8]
            nat_p8 = F.softmax(logits_flat, dim=-1).gather(1, nat_top8)  # [n,8] native mass

            bl, pl = head(h)
            log_pi = F.log_softmax(pl, dim=-1)
            n = h.shape[0]
            for s in range(0, n, args.pos_chunk):
                b = bl[s:s + args.pos_chunk]
                term = torch.einsum("vm,nkm->nvk", codes_all, b)          # [pc,V,K]
                base = F.logsigmoid(-b).sum(-1).unsqueeze(1)              # [pc,1,K]
                mix = torch.logsumexp(log_pi[s:s + args.pos_chunk].unsqueeze(1) + term + base, dim=-1)  # [pc,V]
                logZ = torch.logsumexp(mix, dim=-1)                        # [pc]
                ar = torch.arange(mix.shape[0], device="cuda")
                lp_gt = mix[ar, tgt[s:s + args.pos_chunk]]
                lp_nat = mix[ar, nat_top1[s:s + args.pos_chunk]]
                rank_gt.append(((mix > lp_gt.unsqueeze(1)).sum(-1) + 1).cpu())
                rank_nat.append(((mix > lp_nat.unsqueeze(1)).sum(-1) + 1).cpu())
                mnll.append((logZ - lp_gt).cpu())
                # task 2: coverage of the native head's top-8 by the MoB top-32
                cand32 = mix.topk(32, dim=-1).indices                     # [pc,32]
                hits8 = (cand32.unsqueeze(-1) == nat_top8[s:s + args.pos_chunk].unsqueeze(1)).any(1)  # [pc,8]
                cov_nat8.append(hits8.float().mean(-1).cpu())
                p8 = nat_p8[s:s + args.pos_chunk]
                mass_nat8.append(((hits8 * p8).sum(-1) / p8.sum(-1).clamp_min(1e-30)).cpu())
                # task 3: cascade rerank — native logits rescored on the 32
                # candidates; agreement == native top-1 in cand32 by construction
                pick = cand32.gather(1, logits_flat[s:s + args.pos_chunk]
                                     .gather(1, cand32).argmax(-1, keepdim=True)).squeeze(-1)
                rerank_hit.append((pick == nat_top1[s:s + args.pos_chunk]).cpu())
            print(f"batch {i // args.batch + 1}/{(n_chunks + args.batch - 1) // args.batch}")

    rank_gt = torch.cat(rank_gt).float().numpy()
    rank_nat = torch.cat(rank_nat).float().numpy()
    cov = torch.cat(cov_nat8).numpy()
    mob_nll = torch.cat(mnll).mean().item()
    nat_nll = torch.cat(nnll).mean().item()
    result = {
        "n_positions": int(len(rank_gt)),
        "topk_recall_vs_ground_truth": {f"top-{k}": float((rank_gt <= k).mean()) for k in (1, 8, 32)},
        "topk_recall_vs_native_top1": {f"top-{k}": float((rank_nat <= k).mean()) for k in (1, 8, 32)},
        "coverage_native_top8_by_mob_top32": float(cov.mean()),
        "coverage_native_top8_mean_count_of_8": float(cov.mean() * 8),
        "coverage_native_top8_mass_by_mob_top32": float(torch.cat(mass_nat8).mean().item()),
        "rerank_top1_agreement_vs_native": float(torch.cat(rerank_hit).float().mean().item()),
        "mob_head": {"nll": mob_nll, "ppl": float(np.exp(mob_nll))},
        "native_head": {"nll": nat_nll, "ppl": float(np.exp(nat_nll))},
        "ppl_ratio_mob_over_native": float(np.exp(mob_nll - nat_nll)),
    }
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
