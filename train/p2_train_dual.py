
# (wide 4096x3 vs deep 3072x5, ~51M each) trained SIDE BY SIDE on ONE frozen
# Llama backbone. Distillation's expensive part — the frozen backbone forward
# (h + teacher top-k) — is shared, so two arms cost ~1.2x of one; and both
# arms see the IDENTICAL batch stream (paired samples, zero data-order
# variance in the comparison). Recipe = the proven WSD one (job 520825).
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from mob.mob_common import H, M, MoBHead, LinearHead, ResMLPHead, SwiGLUHead, mob_mix_logprob, decode_greedy
from train.p2_train import kmeans_vertex_init

ARM_SETS = {
    "scale": {
        "wide": {"type": "mob", "width": 4096, "depth": 3},
        "deep": {"type": "mob", "width": 3072, "depth": 5},
    },
    # probe set for 'is shallower actually better? — even linear?' (docs/01-method.md,

    # shallow1l width = 3072 (same as the original head, depth 1 vs 3 — isolates
    # depth alone; hidden width must exceed the 2304-dim output or it becomes a
    # rank bottleneck of its own —)
    "shallow": {
        "shallow1l": {"type": "mob", "width": 3072, "depth": 1},
        "linear": {"type": "linear"},
    },
    # three-head race on SmolLM2 (docs/01-method.md; COSINE per
    # — fixed-budget runs get a full-length decay tail,
    # no constant-LR plateau). Per-arm peaks from the local LR-comfort scan
    # (p7_lr_scan.py); swiglu peak is the fixed-head re-probe's output (override
    # with --peaks swiglu=... without touching code).
    "race3": {
        "control": {"type": "mob", "width": 3072, "depth": 3, "peak": 7e-4},
        "resmlp": {"type": "resmlp", "width": 3072, "depth": 5, "peak": 7e-4},
        "swiglu": {"type": "swiglu", "hidden": 7300, "blocks": 1, "peak": 3e-4},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r".")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--outdir", default="runs/p2_headscale_dual")
    ap.add_argument("--token-cache", default="fineweb_train_tokens.npy")
    ap.add_argument("--dev-cache", default="fineweb_dev_tokens.npy")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--warmup", type=int, default=2500)
    ap.add_argument("--decay-frac", type=float, default=0.35)
    ap.add_argument("--schedule", choices=["wsd", "cosine"], default="wsd")
    ap.add_argument("--peaks", default="",
                    help="per-arm peak override, e.g. control=7e-4,swiglu=5e-4")
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--lambda-ent", type=float, default=0.01)
    ap.add_argument("--init-c", type=float, default=4.0)
    ap.add_argument("--loss-chunk", type=int, default=2048)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--panel-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=20000)
    ap.add_argument("--ckpt-keep", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--arm-set", choices=list(ARM_SETS), default="scale")
    ap.add_argument("--codebook", default="codebook.npz", help="codebook filename under <data-root>/data/")
    ap.add_argument("--backbone-dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()
    ARMS = ARM_SETS[args.arm_set]
    # per-arm LR peaks: ARM_SET defaults, overridable via --peaks tag=value,...
    PEAKS = {tag: float(cfg.get("peak", args.lr)) for tag, cfg in ARMS.items()}
    for kv in args.peaks.split(","):
        if "=" in kv:
            _tag, _v = kv.split("=")
            PEAKS[_tag.strip()] = float(_v)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    model_dir = args.model_dir or os.path.join(
        args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=getattr(torch, args.backbone_dtype)).cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cb = np.load(os.path.join(args.data_root, "data", args.codebook))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    CB_M = codes_all.shape[1]
    vid_to_tok = torch.full((1 << CB_M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().cuda()] = \
        torch.arange(len(cb["vertex_id"]), device="cuda")

    ids_all = np.load(os.path.join(args.data_root, "data", args.token_cache))
    T = args.seq_len
    n_chunks = len(ids_all) // (T + 1)
    if args.max_chunks:
        n_chunks = min(n_chunks, args.max_chunks)
    chunks = np.lib.stride_tricks.as_strided(
        ids_all, shape=(n_chunks, T + 1),
        strides=(ids_all.strides[0] * T, ids_all.strides[0]))
    dev_ids = np.load(os.path.join(args.data_root, "data", args.dev_cache))
    dev_chunks = torch.from_numpy(dev_ids[: 8 * (T + 1)].reshape(8, T + 1).copy()).long().cuda()
    print(f"tokens={len(ids_all):,} chunks={n_chunks} x {T}", flush=True)

    def extract(chunk_batch):
        with torch.no_grad():
            out = model(chunk_batch[:, :-1], output_hidden_states=True)
            h = out.hidden_states[-1].float()
            top = out.logits.topk(args.topk, dim=-1)
            w = torch.softmax(top.values.float(), dim=-1)
        b, t, _ = h.shape
        return h.reshape(b * t, H), top.indices.reshape(b * t, args.topk), \
            w.reshape(b * t, args.topk)

    with torch.no_grad():
        dev_h, dev_tids, dev_tps = extract(dev_chunks)
        dev_top1 = dev_tids[:, 0].clone()

    # two arms: heads + optimizers, identical k-means init (same seed/data)
    heads, opts = {}, {}
    for tag, cfg_arm in ARMS.items():
        if cfg_arm.get("type") == "linear":
            head = LinearHead(k=args.k, m=CB_M).cuda()
        elif cfg_arm.get("type") == "resmlp":
            head = ResMLPHead(k=args.k, width=cfg_arm["width"],
                              depth=cfg_arm["depth"], m=CB_M).cuda()
        elif cfg_arm.get("type") == "swiglu":
            head = SwiGLUHead(k=args.k, hidden=cfg_arm["hidden"],
                              blocks=cfg_arm.get("blocks", 1), m=CB_M).cuda()
        else:
            head = MoBHead(k=args.k, width=cfg_arm["width"],
                           depth=cfg_arm["depth"], m=CB_M).cuda()
        with torch.no_grad():
            warm_chunks = torch.from_numpy(chunks[: 4 * args.batch].copy()).long().cuda()
            warm_n = min(4 * args.batch, n_chunks)
            wlists = [extract(warm_chunks[i:i + args.batch])
                      for i in range(0, warm_n, args.batch)]
            init_logits = kmeans_vertex_init(
                codes_all, torch.cat([x[1] for x in wlists]),
                torch.cat([x[2] for x in wlists]), args.init_c, args.k, seed=args.seed)
            head.out.bias.zero_()
            head.out.bias[: args.k * CB_M] = init_logits.reshape(-1)
        del warm_chunks, wlists
        torch.cuda.empty_cache()
        heads[tag] = head
        opts[tag] = torch.optim.AdamW(head.parameters(), lr=PEAKS[tag], weight_decay=0.0)
        if cfg_arm.get("type") == "linear":
            desc = "linear"
        elif cfg_arm.get("type") == "resmlp":
            desc = f"resmlp width {cfg_arm['width']} depth {cfg_arm['depth']}"
        elif cfg_arm.get("type") == "swiglu":
            desc = f"swiglu hidden {cfg_arm['hidden']} blocks {cfg_arm.get('blocks', 1)}"
        else:
            desc = f"width {cfg_arm['width']} depth {cfg_arm['depth']}"
        print(f"arm {tag}: {desc}, peak {PEAKS[tag]:.2e}, "
              f"{sum(p.numel() for p in head.parameters())/1e6:.1f}M params", flush=True)

    steps_per_epoch = n_chunks // args.batch
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step, peak):
        if step < args.warmup:
            return peak * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        if args.schedule == "cosine":
            # warmup -> cosine decay to 0.1x peak (fixed-budget runs: the whole
            # run is one long polishing tail, no constant-LR plateau)
            return peak * (0.1 + 0.45 * (1.0 + math.cos(math.pi * t)))
        decay_start = 1.0 - args.decay_frac
        if t < decay_start:
            return peak
        u = (t - decay_start) / args.decay_frac
        return peak * (0.1 + 0.9 * (1.0 - u))

    @torch.no_grad()
    def dev_metrics(head):
        agrees, pis, nlls = [], [], []
        for s in range(0, dev_h.shape[0], args.loss_chunk):
            bl, pl = head(dev_h[s:s + args.loss_chunk])
            log_pi = F.log_softmax(pl, dim=-1)
            toks = decode_greedy(bl, log_pi, vid_to_tok, codes_all, m=CB_M)
            agrees.append(toks == dev_top1[s:s + args.loss_chunk])
            pi = log_pi.exp()
            pis.append(pi.mean(0))
            codes = codes_all[dev_tids[s:s + args.loss_chunk]]
            mix = mob_mix_logprob(bl, log_pi, codes)
            nlls.append(-(dev_tps[s:s + args.loss_chunk] * mix).sum(-1))
        agree = torch.cat(agrees).float().mean().item()
        pi_mean = torch.stack(pis).mean(0)
        n_eff = float(torch.exp(-(pi_mean * (pi_mean + 1e-12).log()).sum()).item())
        dead = int((pi_mean < 1e-4).sum().item())
        return n_eff, dead, agree, torch.cat(nlls).mean().item()

    flog = open(os.path.join(args.outdir, "train_log.jsonl"), "w", encoding="utf-8")
    json.dump({"args": vars(args), "arms": ARMS},
              open(os.path.join(args.outdir, "config.json"), "w"))
    archives = {tag: [] for tag in ARMS}
    order = np.random.RandomState(args.seed).permutation(n_chunks)
    t0 = time.time()
    run = {tag: [0.0, 0] for tag in ARMS}
    step = 0
    for epoch in range(args.epochs):
        for i in range(0, steps_per_epoch * args.batch, args.batch):
            step += 1
            cb_idx = torch.from_numpy(chunks[order[i:i + args.batch]].copy()).long().cuda()
            h, tids, tps = extract(cb_idx)
            losses = {}
            for tag in ARMS:
                for g in opts[tag].param_groups:
                    g["lr"] = lr_at(step, PEAKS[tag])
                nll_parts, ent_parts = [], []
                for s in range(0, h.shape[0], args.loss_chunk):
                    bl, pl = heads[tag](h[s:s + args.loss_chunk])
                    log_pi = F.log_softmax(pl, dim=-1)
                    codes = codes_all[tids[s:s + args.loss_chunk]]
                    mix = mob_mix_logprob(bl, log_pi, codes)
                    nll_parts.append(-(tps[s:s + args.loss_chunk] * mix).sum(-1))
                    pi = log_pi.exp()
                    ent_parts.append((pi * log_pi).sum(-1))
                nll = torch.cat(nll_parts).mean()
                ent = torch.cat(ent_parts).mean()
                loss = nll + args.lambda_ent * ent
                opts[tag].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads[tag].parameters(), 1.0)
                opts[tag].step()
                losses[tag] = (nll.item(), ent.item())
                run[tag][0] += nll.item()
                run[tag][1] += 1
            if step % args.log_every == 0:
                rec = {"step": step, "elapsed_s": int(time.time() - t0)}
                for tag in ARMS:
                    rec[f"lr_{tag}"] = lr_at(step, PEAKS[tag])
                    rec[f"nll_{tag}"] = round(run[tag][0] / max(1, run[tag][1]), 4)
                    run[tag] = [0.0, 0]
                print(json.dumps(rec), flush=True)
                flog.write(json.dumps(rec) + "\n")
                flog.flush()
            if step % args.panel_every == 0:
                rec = {"step": step, "panel": True}
                for tag in ARMS:
                    heads[tag].eval()
                    n_eff, dead, agree, dev_nll = dev_metrics(heads[tag])
                    heads[tag].train()
                    rec[f"agree_{tag}"] = round(agree, 4)
                    rec[f"dev_nll_{tag}"] = round(dev_nll, 4)
                    rec[f"n_eff_{tag}"] = round(n_eff, 1)
                    rec[f"dead_{tag}"] = dead
                print(json.dumps(rec), flush=True)
                flog.write(json.dumps(rec) + "\n")
                flog.flush()
            if step % args.ckpt_every == 0:
                for tag in ARMS:
                    p = os.path.join(args.outdir, f"head_{tag}_step{step}.pt")
                    torch.save(heads[tag].state_dict(), p)
                    archives[tag].append(p)
                    while len(archives[tag]) > args.ckpt_keep:
                        os.remove(archives[tag].pop(0))
                for tag in ARMS:
                    torch.save(heads[tag].state_dict(),
                               os.path.join(args.outdir, f"head_{tag}_latest.pt"))
    for tag in ARMS:
        torch.save(heads[tag].state_dict(), os.path.join(args.outdir, f"head_{tag}_final.pt"))
    flog.close()
    print("DUAL_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
