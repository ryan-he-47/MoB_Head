"""Budget-256 proposal shootout (2026-09-19, docs/01-method.md follow-up).

Question: with the candidate budget unlocked to ~256 (CFHead-T4's operating
point), how close do cheap MoB proposals get to the exact mixture_topk@256
ceiling, and does the decoder-strategy gap close the CFHead gap?

Arms per position (all n_batch==1):
  mixture256   exact bound-pruned top-256 by mixture probability (GEMM ceiling)
  adaptive256  propose_adaptive(max_cand=256, floor=32)  — adopted adaptive
  varflip256   propose_varflip: all-K modes dedup + variance-ordered flips,
               per-comp cap 16 (; combos ordered by descending
               flipped-bit variance = per-component exact likelihood order)
  scoreflip256 propose_scoreflip: pi-mass scope + liberal flip emission +
               EXACT mixture rescoring + score-truncation (no ranking GEMM)
  roundflip85  propose_roundflip defaults (reference, the ~85-cand mainline)

Metrics per arm: recall vs native top-1, hit vs native top-8, candidate count,
and coverage = |cand ∩ mixture256| / min(256, |mix256|) (approximation quality
vs the exact ceiling). CFHead-T4 coarse containment@32/256 as the reference row.

Positions: stride-sampled chunks (--n-chunks 32 --chunk-stride 8 -> 2048/source).
Run from repo root: python -m eval.proposal_256_eval [options]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from mob.mob_common import (SwiGLUHead, mixture_topk, propose_adaptive,
                            propose_roundflip, propose_scoreflip, propose_varflip)
from eval.cfhead_rt import CFHeadRT


@torch.no_grad()
def run_source(name, dev, model, head, codes_all, vid_to_tok, rt_t4, args, V, CB_M):
    T = 512
    n_chunks = min(args.n_chunks, len(dev) // (T + 1))
    chunks = torch.from_numpy(dev[: n_chunks * (T + 1)].reshape(n_chunks, T + 1).copy()).long().cuda()
    chunks = chunks[:: args.chunk_stride]
    arms = ("mixture256", "adaptive256", "varflip256", "scoreflip256", "roundflip85")
    rec = {a: {"hit": [], "hit8": [], "n": [], "cov": []} for a in arms}
    t4_rank = []
    t0 = time.perf_counter()
    total = chunks.shape[0] * T

    for ci in range(chunks.shape[0]):
        x = chunks[ci:ci + 1]
        out = model(x[:, :-1], output_hidden_states=True)
        h = out.hidden_states[-1].float().reshape(-1, 2048)
        logits_bf = out.logits.reshape(-1, V)
        nat1 = logits_bf.argmax(-1)
        nat8 = logits_bf.topk(8, dim=-1).indices
        del out
        bl, pl = head(h)
        log_pi = F.log_softmax(pl, dim=-1)

        if rt_t4 is not None:
            rank = []
            for s in range(0, h.shape[0], 8):
                coarse = rt_t4.coarse(h[s:s + 8])
                rank.append((coarse > coarse.gather(
                    1, nat1[s:s + 8].unsqueeze(1))).sum(-1).cpu() + 1)
            t4_rank.append(torch.cat(rank))
            del coarse

        for j in range(h.shape[0]):
            bl_j, lp_j = bl[j:j + 1], log_pi[j:j + 1]
            n1, n8 = nat1[j], nat8[j]
            mix = mixture_topk(bl_j, lp_j, codes_all, 256)
            cands = {
                "mixture256": mix,
                "adaptive256": propose_adaptive(bl_j, lp_j, codes_all, vid_to_tok,
                                                max_cand=256, floor=32),
                "varflip256": propose_varflip(bl_j, lp_j, codes_all, vid_to_tok,
                                              per_comp_cap=16, max_cand=256),
                "scoreflip256": propose_scoreflip(bl_j, lp_j, codes_all, vid_to_tok,
                                                  per_comp_cap=32, max_cand=256),
                "roundflip85": propose_roundflip(bl_j, lp_j, codes_all, vid_to_tok),
            }
            for a, c in cands.items():
                rec[a]["hit"].append(bool((c == n1).any().item()))
                rec[a]["hit8"].append(bool((c.unsqueeze(-1) == n8).any().item()))
                rec[a]["n"].append(c.numel())
                rec[a]["cov"].append(float(torch.isin(c, mix).float().mean().item())
                                       * min(256, c.numel()) / min(256, mix.numel())
                                       if c.numel() else 0.0)
        done = (ci + 1) * T
        el = time.perf_counter() - t0
        print(f"[{name}] {done}/{total} pos ({done / el:.1f} pos/s, "
              f"eta {(total - done) / max(done / el, 1e-9) / 60:.1f}min)", flush=True)

    out = {"source": name, "n_positions": int(total)}
    for a in arms:
        out[a] = {
            "recall": float(np.mean(rec[a]["hit"])),
            "hit_at_native_top8": float(np.mean(rec[a]["hit8"])),
            "mean_candidates": float(np.mean(rec[a]["n"])),
            "p95_candidates": float(np.quantile(rec[a]["n"], 0.95)),
            "coverage_of_mixture256": float(np.mean(rec[a]["cov"])),
        }
    if t4_rank:
        r = torch.cat(t4_rank).float().numpy()
        out["cfhead_t4"] = {"containment_at_32": float((r <= 32).mean()),
                            "containment_at_256": float((r <= 256).mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--cfhead-t4-cache", default="")
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--dev-pattern", default="qwen35_dev_{name}_tokens.npy")
    ap.add_argument("--dev-names", nargs="+", default=["en", "zh", "code", "instruct"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=7300)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--n-chunks", type=int, default=32)
    ap.add_argument("--chunk-stride", type=int, default=8,
                    help="sample every Nth 512-token chunk (32/8 -> 2048 pos/source)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16).cuda().eval()
    cb = np.load(args.codebook)
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    V, CB_M = codes_all.shape
    vid_to_tok = torch.full((1 << CB_M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().cuda()] = torch.arange(V, device="cuda")
    head = SwiGLUHead(k=args.k, hidden=args.hidden, blocks=args.blocks, m=CB_M)
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    head.cuda().eval()

    rt_t4 = None
    if args.cfhead_t4_cache:
        rt_t4 = CFHeadRT(args.cfhead_t4_cache,
                         model.get_output_embeddings().weight.detach(), torch.device("cuda"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    allr = {}
    for name in args.dev_names:
        dev = np.load(Path(args.dev_dir) / args.dev_pattern.format(name=name))
        print(f"=== {name}: {dev.shape[0]} tokens ===", flush=True)
        res = run_source(name, dev, model, head, codes_all, vid_to_tok, rt_t4, args, V, CB_M)
        allr[name] = res
        with open(out_dir / f"proposal256_{name}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps({a: {k: round(v, 4) for k, v in res[a].items()
                              if k in ("recall", "mean_candidates")}
                          for a in ("mixture256", "adaptive256", "varflip256",
                                    "scoreflip256", "roundflip85")}, indent=1), flush=True)
    with open(out_dir / "proposal256_all.json", "w") as f:
        json.dump(allr, f, indent=2)
    print(f"saved {out_dir / 'proposal256_all.json'}")


if __name__ == "__main__":
    main()
