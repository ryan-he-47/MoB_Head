"""Sampling-shape eval: how much of the native top-k distribution does each
candidate set capture?

Theory (truncate+renormalize sampling): within the candidate set C the
renormalized distribution p' = p/mass(C) preserves ALL pairwise odds exactly;
the only distortions are (a) the missing mass 1 - mass(C) and (b) WHERE that
mass sits. So the suite is:
  mass_C            native probability mass captured (T=1.0 and T=0.7) — the
                    sampling hit rate (greedy containment's sampling analog)
  coverage curve    mass of native top-k captured, k = 1,2,4,8,...,256
  worst casualty    the single highest-probability missed token (prob, rank)
  missed buckets    missed mass in native-rank buckets 1-8 / 9-32 / 33-128 / 129+
  nucleus090        indicator: the native top-p=0.9 nucleus fully contained
Arms: mixture256 / scoreflip256 / adaptive256 / roundflip85 / cfhead_t4@256.
Positions: stride-sampled like proposal_256_eval (--chunk-stride 8).
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
                            propose_roundflip, propose_scoreflip)
from eval.cfhead_rt import CFHeadRT

K_CURVE = (1, 2, 4, 8, 16, 32, 64, 128, 256)
BUCKETS = ((0, 8), (8, 32), (32, 128), (128, 256))


@torch.no_grad()
def run_source(name, dev, model, head, codes_all, vid_to_tok, rt_t4, args, V, CB_M):
    T = 512
    n_chunks = min(args.n_chunks, len(dev) // (T + 1))
    chunks = torch.from_numpy(dev[: n_chunks * (T + 1)].reshape(n_chunks, T + 1).copy()).long().cuda()
    chunks = chunks[:: args.chunk_stride]
    arms = ("mixture256", "scoreflip256", "adaptive256", "roundflip85", "cfhead_t4")
    agg = {a: {"mass": [], "mass_t07": [], "cov": {k: [] for k in K_CURVE},
               "wc_prob": [], "wc_rank": [], "buckets": {b: [] for b in BUCKETS},
               "nuc": [], "n": []} for a in arms}
    t0 = time.perf_counter()
    total = chunks.shape[0] * T

    for ci in range(chunks.shape[0]):
        x = chunks[ci:ci + 1]
        out = model(x[:, :-1], output_hidden_states=True)
        h = out.hidden_states[-1].float().reshape(-1, 2048)
        logits = out.logits.reshape(-1, V).float()
        del out
        probs = F.softmax(logits, dim=-1)                       # [512,V]
        probs07 = F.softmax(logits / 0.7, dim=-1)
        sp, sids = probs.topk(256, dim=-1)                      # [512,256] sorted desc
        del logits
        bl, pl = head(h)
        log_pi = F.log_softmax(pl, dim=-1)

        for j in range(h.shape[0]):
            bl_j, lp_j = bl[j:j + 1], log_pi[j:j + 1]
            cands = {
                "mixture256": mixture_topk(bl_j, lp_j, codes_all, 256),
                "scoreflip256": propose_scoreflip(bl_j, lp_j, codes_all, vid_to_tok,
                                                  per_comp_cap=32, max_cand=256),
                "adaptive256": propose_adaptive(bl_j, lp_j, codes_all, vid_to_tok,
                                                max_cand=256, floor=32),
                "roundflip85": propose_roundflip(bl_j, lp_j, codes_all, vid_to_tok),
                "cfhead_t4": rt_t4.candidates(h[j:j + 1], 256)[0],
            }
            p_row, p07_row = probs[j], probs07[j]
            sp_row, sid_row = sp[j], sids[j]
            for a, c in cands.items():
                A = agg[a]
                A["mass"].append(float(p_row[c].sum().item()))
                A["mass_t07"].append(float(p07_row[c].sum().item()))
                in_c = torch.isin(sid_row, c)                     # [256] bool
                cum = torch.cumsum(sp_row, 0)
                for k in K_CURVE:
                    A["cov"][k].append(float((sp_row[:k] * in_c[:k]).sum().item()))
                if bool(in_c.all().item()):
                    A["wc_prob"].append(0.0)
                    A["wc_rank"].append(0)
                else:
                    fm = int((~in_c).int().argmax().item())
                    A["wc_prob"].append(float(sp_row[fm].item()))
                    A["wc_rank"].append(fm + 1)
                for lo, hi in BUCKETS:
                    A["buckets"][(lo, hi)].append(
                        float((sp_row[lo:hi] * (~in_c[lo:hi])).sum().item()))
                # native top-p=0.9 nucleus fully contained? (only checkable when
                # the nucleus ends inside the top-256 window)
                if float(cum[-1].item()) >= 0.9:
                    nuc_n = int((cum < 0.9).sum().item()) + 1
                    A["nuc"].append(bool(in_c[:nuc_n].all().item()))
                A["n"].append(c.numel())
        done = (ci + 1) * T
        el = time.perf_counter() - t0
        print(f"[{name}] {done}/{total} ({done / el:.1f} pos/s, "
              f"eta {(total - done) / max(done / el, 1e-9) / 60:.1f}min)", flush=True)

    out = {"source": name, "n_positions": int(total)}
    for a in arms:
        A = agg[a]
        out[a] = {
            "mass_captured_t1": float(np.mean(A["mass"])),
            "mass_captured_t07": float(np.mean(A["mass_t07"])),
            "coverage_mass_by_k": {str(k): float(np.mean(A["cov"][k])) for k in K_CURVE},
            "worst_casualty_prob_mean": float(np.mean(A["wc_prob"])),
            "worst_casualty_prob_p95": float(np.quantile(A["wc_prob"], 0.95)),
            "worst_casualty_rank_mean": float(np.mean([r for r in A["wc_rank"] if r] or [0])),
            "missed_mass_by_rank_bucket": {f"{lo + 1}-{hi}": float(np.mean(A["buckets"][(lo, hi)]))
                                           for lo, hi in BUCKETS},
            "nucleus090_fully_contained": float(np.mean(A["nuc"]) if A["nuc"] else float("nan")),
            "nucleus090_checkable_frac": float(len(A["nuc"]) / max(total, 1)),
            "mean_candidates": float(np.mean(A["n"])),
        }
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
    ap.add_argument("--chunk-stride", type=int, default=8)
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
    rt_t4 = CFHeadRT(args.cfhead_t4_cache,
                     model.get_output_embeddings().weight.detach(), torch.device("cuda"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    allr = {}
    for name in args.dev_names:
        dev = np.load(Path(args.dev_dir) / args.dev_pattern.format(name=name))
        print(f"=== {name} ===", flush=True)
        res = run_source(name, dev, model, head, codes_all, vid_to_tok, rt_t4, args, V, CB_M)
        allr[name] = res
        with open(out_dir / f"sampling_shape_{name}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps({a: res[a]["mass_captured_t1"] for a in
                          ("mixture256", "scoreflip256", "adaptive256",
                           "roundflip85", "cfhead_t4")}, indent=1), flush=True)
    with open(out_dir / "sampling_shape_all.json", "w") as f:
        json.dump(allr, f, indent=2)
    print(f"saved {out_dir / 'sampling_shape_all.json'}")


if __name__ == "__main__":
    main()
