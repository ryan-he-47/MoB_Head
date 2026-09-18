"""Teacher-forced paired fidelity eval: CFHead vs MoB on held-out dev (docs/26 §2).

The missing-metric harness the CFHead paper never reports: same positions, same
frozen backbone, all arms measured against the native head's own outputs.

Per position we record:
  native reference: top-1, top-8, top-1 prob, full-vocab entropy, NLL
  MoB head:         full-density NLL (normalized PPL vs native), mixture-rank of
                    native top-1, mixture-top32 set metrics, round-32 proposal set,
                    roundflip proposal set (per-position), cascade rerank agreement
  CFHead (per cache config): coarse-rank of native top-1 -> containment@k curve,
                    post-fine top-1 agreement at k=32 (budget-matched) and at the
                    config's default k (captures tie flips)

Bucketing: quartiles of the native top-1 probability (peaked vs diffuse positions)
per source. Run once per dev source, then a combined JSON is written.

Run from repo root, e.g.:
  python -m eval.cfhead_eval --model-dir ... --head ... --codebook ... \
      --dev-dir .../cache --dev-names en zh code instruct --out-dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401 (tokenizer for parity)

from mob.mob_common import H, MoBHead, ResMLPHead, SwiGLUHead, propose_roundflip
from eval.cfhead_rt import CFHeadRT

K_LIST = (8, 16, 32, 64, 128, 256, 512, 2048, 8192)


def load_head(args, cb_m: int):
    if args.head_type == "resmlp":
        head = ResMLPHead(k=args.k, width=args.width, depth=args.depth, m=cb_m)
    elif args.head_type == "swiglu":
        head = SwiGLUHead(k=args.k, hidden=args.hidden, blocks=args.blocks, m=cb_m)
    else:
        head = MoBHead(k=args.k, width=args.width, depth=args.depth, m=cb_m)
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    return head.cuda().eval()


def find_cfhead_caches(cache_dir: Path) -> dict[str, Path]:
    out = {}
    for tag, pat in (("cfhead_q", "*_T8_C64_*.pt"), ("cfhead_m", "*_T8_C4_*.pt")):
        hits = sorted(cache_dir.glob(pat))
        if hits:
            out[tag] = hits[0]
        else:
            print(f"[cfhead_eval] WARNING: no cache for {tag} ({pat}) in {cache_dir}")
    return out


@torch.no_grad()
def run_source(name: str, dev: np.ndarray, model, head, codes_all, vid_to_tok,
               cf: dict[str, CFHeadRT], args, V: int, CB_M: int) -> dict:
    T = 512
    n_chunks = min(args.n_chunks, len(dev) // (T + 1))
    chunks = torch.from_numpy(dev[: n_chunks * (T + 1)].reshape(n_chunks, T + 1).copy()).long().cuda()

    shifts = (1 << torch.arange(CB_M - 1, -1, -1, device="cuda")).long()
    per_pos: dict[str, list] = {k: [] for k in (
        "nat_p1", "nat_ent", "nat_nll", "mob_nll", "mix_rank_nat1", "mix_rerank_agree",
        "mix_hit8", "round_recall", "round_rerank_agree",
        "rf_recall", "rf_hit8", "rf_rerank_agree", "rf_count",
        "cfq_rank", "cfm_rank",
        "cfq_fine32", "cfq_fine_def", "cfm_fine32", "cfm_fine_def",
    )}
    k_def = {"cfq": 256, "cfm": 8192}

    for i in range(0, n_chunks, args.batch):
        x = chunks[i:i + args.batch]
        out = model(x[:, :-1], output_hidden_states=True)
        tgt = x[:, 1:].reshape(-1)
        h = out.hidden_states[-1].float().reshape(-1, H)
        logits_flat = out.logits.float().reshape(-1, V)

        nat_nll = F.cross_entropy(logits_flat, tgt, reduction="none")
        nat_top1 = logits_flat.argmax(-1)
        nat_top8 = logits_flat.topk(8, dim=-1).indices
        probs = F.softmax(logits_flat, dim=-1)
        nat_p1 = probs.gather(1, nat_top1.unsqueeze(1)).squeeze(1)
        nat_ent = -(probs * torch.log(probs.clamp_min(1e-45))).sum(-1)

        bl, pl = head(h)
        log_pi = F.log_softmax(pl, dim=-1)
        n = h.shape[0]
        for s in range(0, n, args.pos_chunk):
            pc = slice(s, s + args.pos_chunk)
            b = bl[pc]
            lp = log_pi[pc]
            term = torch.einsum("vm,nkm->nvk", codes_all, b)            # [pc,V,K]
            base = F.logsigmoid(-b).sum(-1).unsqueeze(1)                # [pc,1,K]
            mix = torch.logsumexp(lp.unsqueeze(1) + term + base, dim=-1)  # [pc,V]
            del term, base
            logZ = torch.logsumexp(mix, dim=-1)
            ar = torch.arange(mix.shape[0], device="cuda")
            lp_gt = mix[ar, tgt[pc]]
            lp_nat = mix[ar, nat_top1[pc]]
            per_pos["mob_nll"].append((logZ - lp_gt).cpu())
            per_pos["mix_rank_nat1"].append(((mix > lp_nat.unsqueeze(1)).sum(-1) + 1).cpu())
            cand32 = mix.topk(32, dim=-1).indices
            nt1 = nat_top1[pc]
            nt8 = nat_top8[pc]
            per_pos["mix_hit8"].append(
                (cand32.unsqueeze(-1) == nt8.unsqueeze(1)).any(1).any(-1).cpu())
            pick = cand32.gather(1, logits_flat[pc].gather(1, cand32).argmax(-1, keepdim=True)).squeeze(1)
            per_pos["mix_rerank_agree"].append((pick == nt1).cpu())
            del mix, cand32

            # round-32 proposal (vectorized)
            ktop = lp.topk(32, -1).indices
            bits = b.gather(1, ktop.unsqueeze(-1).expand(-1, -1, CB_M)) > 0
            rtok = vid_to_tok[(bits.long() * shifts).sum(-1)]           # [pc,32], -1 invalid
            per_pos["round_recall"].append((rtok == nt1.unsqueeze(1)).any(-1).cpu())
            sc = logits_flat[pc].gather(1, rtok.clamp_min(0))
            sc = sc.masked_fill(rtok < 0, float("-inf"))
            rpick = rtok.clamp_min(0).gather(1, sc.argmax(-1, keepdim=True)).squeeze(1)
            per_pos["round_rerank_agree"].append((rpick == nt1).cpu())
            del rtok, sc

            # roundflip proposal (per-position; n_batch==1 API)
            for j in range(s, min(s + args.pos_chunk, n)):
                cands = propose_roundflip(bl[j:j + 1], log_pi[j:j + 1], codes_all, vid_to_tok)
                nt1_j = nat_top1[j:j + 1]
                per_pos["rf_recall"].append((cands == nt1_j).any().cpu().view(1))
                per_pos["rf_hit8"].append(
                    (cands.unsqueeze(-1) == nat_top8[j]).any(0).any().cpu().view(1))
                sc_j = logits_flat[j].gather(0, cands)
                per_pos["rf_rerank_agree"].append(
                    (cands[sc_j.argmax()] == nt1_j).cpu().view(1))
                per_pos["rf_count"].append(torch.tensor([cands.numel()], dtype=torch.float32))

            # CFHead arms
            hh = h[pc]
            for key, rt in (("cfq", cf.get("cfhead_q")), ("cfm", cf.get("cfhead_m"))):
                if rt is None:
                    continue
                coarse = rt.coarse(hh)                                   # [pc,V] fp32
                rank = (coarse > coarse[ar, nat_top1[pc]].unsqueeze(1)).sum(-1) + 1
                per_pos[f"{key}_rank"].append(rank.cpu())
                for ktag, kv in (("fine32", 32), ("fine_def", k_def[key])):
                    cand = coarse.topk(min(kv, V), dim=-1, sorted=False).indices
                    fl = rt.fine_logits(hh, cand)
                    # repo tie semantics: among fine-max ties take the lowest token id
                    maxfl = fl.max(dim=-1, keepdim=True).values
                    fpick = cand.masked_fill(fl != maxfl, V).min(dim=-1).values
                    per_pos[f"{key}_{ktag}"].append((fpick == nat_top1[pc]).cpu())
                del coarse

        per_pos["nat_p1"].append(nat_p1.cpu())
        per_pos["nat_ent"].append(nat_ent.cpu())
        per_pos["nat_nll"].append(nat_nll.cpu())
        print(f"[{name}] batch {i // args.batch + 1}/{(n_chunks + args.batch - 1) // args.batch}", flush=True)

    d = {k: torch.cat(v).float().numpy() for k, v in per_pos.items() if v}
    n_pos = len(d["nat_nll"])
    mob_nll = float(d["mob_nll"].mean())
    nat_nll = float(d["nat_nll"].mean())

    def bucketed(metric: np.ndarray, agg: str = "mean") -> dict:
        qs = np.quantile(d["nat_p1"], [0.25, 0.5, 0.75])
        out = {"overall": float(getattr(metric, agg)())}
        for bi, (lo, hi) in enumerate([(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 2)]):
            m = (d["nat_p1"] > lo) & (d["nat_p1"] <= hi)
            out[f"q{bi + 1}"] = float(getattr(metric[m], agg)()) if m.any() else None
        return out

    result = {
        "source": name,
        "n_positions": int(n_pos),
        "native_head": {"nll": nat_nll, "ppl": float(np.exp(nat_nll)),
                        "mean_top1_prob": float(d["nat_p1"].mean()),
                        "mean_entropy": float(d["nat_ent"].mean())},
        "mob_head": {"nll": mob_nll, "ppl": float(np.exp(mob_nll)),
                     "ppl_ratio_vs_native": float(np.exp(mob_nll - nat_nll))},
        "mob_mixture": {
            "recall_at_1": bucketed((d["mix_rank_nat1"] <= 1)),
            "recall_at_32": bucketed((d["mix_rank_nat1"] <= 32)),
            "hit_at_native_top8": bucketed(d["mix_hit8"]),
            "rerank_agreement": bucketed(d["mix_rerank_agree"]),
        },
        "mob_round32": {
            "recall": bucketed(d["round_recall"]),
            "rerank_agreement": bucketed(d["round_rerank_agree"]),
        },
        "mob_roundflip": {
            "recall": bucketed(d["rf_recall"]),
            "hit_at_native_top8": bucketed(d["rf_hit8"]),
            "rerank_agreement": bucketed(d["rf_rerank_agree"]),
            "mean_candidates": float(d["rf_count"].mean()),
            "p95_candidates": float(np.quantile(d["rf_count"], 0.95)),
        },
    }
    for key, label in (("cfq", "cfhead_q"), ("cfm", "cfhead_m")):
        rk = f"{key}_rank"
        if rk not in d:
            continue
        result[label] = {
            "containment_at_k": {str(k): bucketed((d[rk] <= k)) for k in K_LIST},
            "fine_agreement_at_32": bucketed(d[f"{key}_fine32"]),
            f"fine_agreement_at_{k_def[key]}": bucketed(d[f"{key}_fine_def"]),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--cfhead-cache-dir", default="")
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--dev-pattern", default="qwen35_dev_{name}_tokens.npy")
    ap.add_argument("--dev-names", nargs="+", default=["en", "zh", "code", "instruct"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--width", type=int, default=3072)
    ap.add_argument("--head-type", choices=["mob", "resmlp", "swiglu"], default="swiglu")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=7300)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--backbone-dtype", default="bfloat16")
    ap.add_argument("--n-chunks", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pos-chunk", type=int, default=8,
                    help="positions per sub-chunk for the [pos,V,K] pass (V=248k,K=128: 8 ~= 1GB)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=getattr(torch, args.backbone_dtype)).cuda().eval()
    cb = np.load(args.codebook)
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    V, CB_M = codes_all.shape
    vid_to_tok = torch.full((1 << CB_M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().cuda()] = \
        torch.arange(V, device="cuda")
    head = load_head(args, CB_M)

    cf: dict[str, CFHeadRT] = {}
    if args.cfhead_cache_dir:
        weight = model.get_output_embeddings().weight.detach()
        for tag, path in find_cfhead_caches(Path(args.cfhead_cache_dir)).items():
            print(f"[cfhead_eval] loading {tag}: {path.name}", flush=True)
            cf[tag] = CFHeadRT(path, weight, torch.device("cuda"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name in args.dev_names:
        dev_path = Path(args.dev_dir) / args.dev_pattern.format(name=name)
        dev = np.load(dev_path)
        print(f"=== source {name}: {dev_path.name}, {len(dev)} tokens ===", flush=True)
        res = run_source(name, dev, model, head, codes_all, vid_to_tok, cf, args, V, CB_M)
        all_results[name] = res
        with open(out_dir / f"cfhead_eval_{name}.json", "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps({"source": name,
                          "mob_ppl_ratio": res["mob_head"]["ppl_ratio_vs_native"],
                          "rf_rerank": res["mob_roundflip"]["rerank_agreement"]["overall"]},
                         indent=2), flush=True)
    with open(out_dir / "cfhead_eval_all.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"saved {out_dir / 'cfhead_eval_all.json'}")


if __name__ == "__main__":
    main()
