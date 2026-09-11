# P7 local LR-comfort scan for the three-head race (docs/16 §5, user direction
# 2026-09-07): find each head variant's comfortable LR peak LOCALLY (Llama
# backbone, h=2048 — same hidden size as the SmolLM2 race backbone; LR comfort
# is architecture-driven, transfers across backbones of equal h).
# Resumable: one result JSON per (variant, peak) in the outdir; restart skips
# completed runs. Constant LR after 150-step warmup (the comfort question is
# peak tolerance; decay shapes are decided later on the cluster).
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from mob.mob_common import H, M, MoBHead, ResMLPHead, SwiGLUHead, mob_mix_logprob, decode_greedy
from train.p2_train import kmeans_vertex_init

VARIANTS = {
    "mlp3": lambda k, m: MoBHead(k=k, width=3072, depth=3, m=m),
    "resmlp5": lambda k, m: ResMLPHead(k=k, width=3072, depth=5, m=m),
    "swiglu1": lambda k, m: SwiGLUHead(k=k, hidden=7300, blocks=1, m=m),
    "swiglu2": lambda k, m: SwiGLUHead(k=k, hidden=3690, blocks=2, m=m),
}
PEAKS = [3e-4, 7e-4, 1.5e-3, 3e-3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"E:\Trash_things\llm_codec")
    ap.add_argument("--outdir", default=r"E:\Trash_things\llm_codec\runs\lr_scan")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--lambda-ent", type=float, default=0.01)
    ap.add_argument("--init-c", type=float, default=4.0)
    ap.add_argument("--loss-chunk", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default=",".join(VARIANTS.keys()))
    ap.add_argument("--peaks", default=",".join(str(p) for p in PEAKS))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    os.makedirs(args.outdir, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    peaks = [float(p) for p in args.peaks.split(",") if p.strip()]

    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master"),
        dtype=torch.float16).cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    cb = np.load(os.path.join(args.data_root, "data", "codebook.npz"))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    CB_M = codes_all.shape[1]
    vid_to_tok = torch.full((1 << CB_M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().cuda()] = \
        torch.arange(len(cb["vertex_id"]), device="cuda")

    cache = np.load(os.path.join(args.data_root, "data", "fineweb_probe_tokens.npz"))
    T = args.seq_len
    tr = cache["train"]
    n_chunks = len(tr) // (T + 1)
    chunks = tr[: n_chunks * (T + 1)].reshape(n_chunks, T + 1).astype(np.int64)
    dv = cache["dev"]
    n_dev = min(8, len(dv) // (T + 1))
    dev_chunks = torch.from_numpy(dv[: n_dev * (T + 1)].reshape(n_dev, T + 1).astype(np.int64)).cuda()

    def extract(x):
        with torch.no_grad():
            out = model(x[:, :-1], output_hidden_states=True)
            h = out.hidden_states[-1].float()
            top = out.logits.topk(args.topk, dim=-1)
            w = torch.softmax(top.values.float(), dim=-1)
        b, t, _ = h.shape
        return h.reshape(b * t, H), top.indices.reshape(b * t, args.topk), \
            w.reshape(b * t, args.topk)

    with torch.no_grad():
        dev_h, dev_tids, dev_tps = extract(dev_chunks)
        dev_top1 = dev_tids[:, 0].clone()

    order = np.random.RandomState(args.seed).permutation(n_chunks)
    for variant in variants:
        for peak in peaks:
            tag = f"{variant}__{peak:g}"
            result_path = os.path.join(args.outdir, f"{tag}.json")
            if os.path.exists(result_path):
                print(f"skip {tag} (done)", flush=True)
                continue
            torch.manual_seed(args.seed)
            head = VARIANTS[variant](args.k, CB_M).cuda()
            with torch.no_grad():
                warm = torch.from_numpy(chunks[: 4 * args.batch].copy()).long().cuda()
                wlists = [extract(warm[i:i + args.batch])
                          for i in range(0, 4 * args.batch, args.batch)]
                init_logits = kmeans_vertex_init(
                    codes_all, torch.cat([x[1] for x in wlists]),
                    torch.cat([x[2] for x in wlists]), args.init_c, args.k, seed=args.seed)
                head.out.bias.zero_()
                head.out.bias[: args.k * CB_M] = init_logits.reshape(-1)
            opt = torch.optim.AdamW(head.parameters(), lr=peak, weight_decay=0.0)
            warmup = 150
            t0 = time.time()
            nlls, spikes = [], 0
            log_path = os.path.join(args.outdir, f"{tag}.log.jsonl")
            with open(log_path, "w", encoding="utf-8") as flog:
                oi = 0
                for step in range(1, args.steps + 1):
                    if oi + args.batch > n_chunks:
                        oi = 0
                    for g in opt.param_groups:
                        g["lr"] = peak * min(1.0, step / warmup)
                    x = torch.from_numpy(chunks[order[oi:oi + args.batch]].copy()).long().cuda()
                    oi += args.batch
                    h, tids, tps = extract(x)
                    nll_parts, ent_parts = [], []
                    for s in range(0, h.shape[0], args.loss_chunk):
                        bl, pl = head(h[s:s + args.loss_chunk])
                        log_pi = F.log_softmax(pl, dim=-1)
                        codes = codes_all[tids[s:s + args.loss_chunk]]
                        mix = mob_mix_logprob(bl, log_pi, codes)
                        nll_parts.append(-(tps[s:s + args.loss_chunk] * mix).sum(-1))
                        ent_parts.append((log_pi.exp() * log_pi).sum(-1))
                    nll = torch.cat(nll_parts).mean()
                    loss = nll + args.lambda_ent * torch.cat(ent_parts).mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    opt.step()
                    v = nll.item()
                    nlls.append(v)
                    if step > 100 and v > 3 * float(np.median(nlls[-100:])):
                        spikes += 1
                    if step % 100 == 0:
                        rec = {"step": step, "nll": round(float(np.mean(nlls[-100:])), 4)}
                        flog.write(json.dumps(rec) + "\n")
                        flog.flush()
            # end-of-run panel: dev teacher-weighted NLL + decode agreement
            head.eval()
            with torch.no_grad():
                agrees, nll_dev = [], []
                for s in range(0, dev_h.shape[0], args.loss_chunk):
                    bl, pl = head(dev_h[s:s + args.loss_chunk])
                    log_pi = F.log_softmax(pl, dim=-1)
                    toks = decode_greedy(bl, log_pi, vid_to_tok, codes_all, m=CB_M)
                    agrees.append(toks == dev_top1[s:s + args.loss_chunk])
                    mix = mob_mix_logprob(bl, log_pi, codes_all[dev_tids[s:s + args.loss_chunk]])
                    nll_dev.append(-(dev_tps[s:s + args.loss_chunk] * mix).sum(-1))
            a = np.array(nlls)
            k = min(10, len(a) // 2)
            slope = float(np.polyfit(np.arange(k), a[-k:], 1)[0]) if k > 1 else 0.0
            result = {"variant": variant, "peak": peak, "steps": args.steps,
                      "final_nll": float(np.mean(a[-100:])),
                      "slope_last": slope, "spikes": spikes,
                      "dev_nll": round(torch.cat(nll_dev).mean().item(), 4),
                      "dev_agree": round(torch.cat(agrees).float().mean().item(), 4),
                      "elapsed_s": int(time.time() - t0)}
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"done {tag}: dev_nll {result['dev_nll']}, agree {result['dev_agree']}, "
                  f"spikes {spikes}, slope {slope:+.5f}, {result['elapsed_s']}s", flush=True)
            del head, opt
            torch.cuda.empty_cache()
    print("SCAN_DONE", flush=True)


if __name__ == "__main__":
    main()
