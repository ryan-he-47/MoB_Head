# P5 / Rung-3 (docs/15, user-approved 2026-09-05): from-scratch twin pretrain —
# the decisive structural test for the MoB head. Two identical 212M Llama-mini
# models (h768/L12/H12, tied embeddings, V=128256) differ ONLY in the output
# head: native tied lm_head + softmax CE vs MoB head over 17-bit semantic-tree
# codes (pure MoB-NLL, T=1). Same data order, same seed, full fp32 (no
# half-precision variables — the hard-bones doctrine: one variable less, one
# failure mode less). Verdict lines and the failure playbook: docs/15 §1/§6.
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from mob.mob_common import M, MoBHead, mob_mix_logprob, decode_rerank_native
from train.p2_train import kmeans_vertex_init

V = 128256
PROMPTS = ["The little boy was playing with", "My mother said I should"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-type", choices=["native", "mob"], required=True)
    ap.add_argument("--data-root", default=r"E:\Trash_things\llm_codec")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--token-cache", default="fineweb_train_tokens.npy")
    ap.add_argument("--dev-cache", default="fineweb_dev_tokens.npy")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--final-frac", type=float, default=0.1,
                    help="cosine decays to this fraction of peak LR (0.02 for floor-probing)")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--width", type=int, default=922)
    ap.add_argument("--lambda-ent", type=float, default=0.01)
    ap.add_argument("--init-c", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--panel-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="weights-only resume: latest.pt with {model[,head]}; fresh optimizer+schedule")
    ap.add_argument("--skip-steps", type=int, default=0,
                    help="continue the SAME seeded data permutation past the already-consumed "
                         "prefix (skip_steps * batch chunks) — no repeats until cache exhausted")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    T = args.seq_len
    data = np.load(os.path.join(args.data_root, "data", args.token_cache))
    n_chunks = len(data) // (T + 1)
    if args.max_chunks:
        n_chunks = min(n_chunks, args.max_chunks)
    chunks = np.ascontiguousarray(data[: n_chunks * (T + 1)].reshape(n_chunks, T + 1))
    dev = np.load(os.path.join(args.data_root, "data", args.dev_cache))
    n_dev = min(64, len(dev) // (T + 1))
    dev_chunks = torch.from_numpy(dev[: n_dev * (T + 1)].reshape(n_dev, T + 1).copy()).long().cuda()
    print(f"train chunks {n_chunks}, dev {n_dev}", flush=True)

    cfg = LlamaConfig(hidden_size=768, intermediate_size=3072, num_hidden_layers=12,
                      num_attention_heads=12, num_key_value_heads=12,
                      vocab_size=V, tie_word_embeddings=True,
                      max_position_embeddings=2048)
    model = LlamaForCausalLM(cfg).cuda()  # fp32, from scratch
    n_bb = sum(p.numel() for p in model.parameters())
    print(f"backbone+tied-emb params: {n_bb/1e6:.1f}M", flush=True)

    cb = np.load(os.path.join(args.data_root, "data", "codebook.npz"))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    head = None
    if args.head_type == "mob":
        head = MoBHead(k=args.k, width=args.width, h=768).cuda()
    if args.resume:
        ck = torch.load(os.path.join(args.resume), map_location="cpu")
        model.load_state_dict(ck["model"])
        if head is not None:
            head.load_state_dict(ck["head"])
        print(f"resumed weights from {args.resume} (step {ck['step']}); "
              f"fresh optimizer, data offset {args.skip_steps}", flush=True)
    if head is not None and not args.resume:
        # unigram-weighted k-means vertex init (no teacher exists from scratch):
        # sampling each first-batch target occurrence uniformly == sampling by
        # empirical frequency (docs/15 section 3)
        warm = torch.from_numpy(chunks[: 4 * args.batch, 1:].copy()).reshape(-1).cuda()
        tps = torch.ones(warm.numel(), device="cuda") / warm.numel()
        init_logits = kmeans_vertex_init(codes_all, warm, tps, args.init_c, args.k,
                                         seed=args.seed)
        with torch.no_grad():
            head.out.bias.zero_()
            head.out.bias[: args.k * M] = init_logits.reshape(-1)
    if head is not None:
        params = list(model.parameters()) + list(head.parameters())
        print(f"mob head params: {sum(p.numel() for p in head.parameters())/1e6:.1f}M", flush=True)
    else:
        params = list(model.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = int(args.warmup_frac * args.steps)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return args.lr * (args.final_frac + (1 - args.final_frac) * 0.5 * (1 + np.cos(np.pi * t)))

    @torch.no_grad()
    def dev_mob_nll():
        nlls, pis, logzs = [], [], []
        for i in range(0, n_dev, 2):
            x = dev_chunks[i:i + 2]
            h = model.model(x[:, :-1], use_cache=False).last_hidden_state.reshape(-1, 768)
            bl, pl = head(h)
            log_pi = F.log_softmax(pl, dim=-1)
            tgt = x[:, 1:].reshape(-1)
            lp_true = mob_mix_logprob(bl, log_pi, codes_all[tgt].unsqueeze(1))[:, 0]
            lz = []
            for s in range(0, h.shape[0], 8):
                b = bl[s:s + 8]
                term = torch.einsum("vm,nkm->nvk", codes_all, b)
                base = F.logsigmoid(-b).sum(-1).unsqueeze(1)
                mix = torch.logsumexp(log_pi[s:s + 8].unsqueeze(1) + term + base, dim=-1)
                lz.append(torch.logsumexp(mix, dim=-1))
            nlls.append(torch.cat(lz) - lp_true)
            logzs.append(torch.cat(lz))
            pis.append(log_pi.exp().mean(0))
        pi_mean = torch.stack(pis).mean(0)
        n_eff = float(torch.exp(-(pi_mean * (pi_mean + 1e-12).log()).sum()).item())
        dead = int((pi_mean < 1e-4).sum().item())
        return (torch.cat(nlls).mean().item(), torch.cat(logzs).mean().item(),
                n_eff, dead)

    @torch.no_grad()
    def dev_native_nll():
        out = []
        for i in range(0, n_dev, 2):
            x = dev_chunks[i:i + 2]
            logits = model(x[:, :-1], use_cache=False).logits
            out.append(F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1),
                                       reduction="none"))
        return torch.cat(out).mean().item()

    def save(tag):
        sd = {"model": model.state_dict(), "step": tag}
        if head is not None:
            sd["head"] = head.state_dict()
        torch.save(sd, os.path.join(args.outdir, "latest.pt"))
        if tag % 20000 == 0 or tag == args.steps:
            torch.save(sd, os.path.join(args.outdir, f"step{tag}.pt"))

    flog = open(os.path.join(args.outdir, "train_log.jsonl"), "w", encoding="utf-8")
    json.dump({"args": vars(args)}, open(os.path.join(args.outdir, "config.json"), "w"))
    order = np.random.RandomState(args.seed).permutation(n_chunks)
    save(0)  # rollback target for the NaN rule
    t0 = time.time()
    oi = args.skip_steps * args.batch
    nan_count = 0
    lr_scale = 1.0
    step = 0
    while step < args.steps:
        step += 1
        if oi + args.batch > n_chunks:
            oi = 0
        x = torch.from_numpy(chunks[order[oi:oi + args.batch]].copy()).long().cuda()
        oi += args.batch
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * lr_scale
        if args.head_type == "native":
            logits = model(x[:, :-1], use_cache=False).logits.float()
            loss = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
        else:
            h = model.model(x[:, :-1], use_cache=False).last_hidden_state.reshape(-1, 768)
            bl, pl = head(h)
            log_pi = F.log_softmax(pl, dim=-1)
            tgt = x[:, 1:].reshape(-1)
            nll = -mob_mix_logprob(bl, log_pi, codes_all[tgt].unsqueeze(1))[:, 0].mean()
            pi = log_pi.exp()
            ent = (pi * log_pi).sum(-1).mean()
            loss = nll + args.lambda_ent * ent
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        if not torch.isfinite(loss):
            # NaN rule (docs/15 section 4): rollback once + halve LR; second time stop
            nan_count += 1
            print(f"NaN at step {step} (count {nan_count})", flush=True)
            if nan_count >= 2:
                print("second NaN — stopping for human diagnosis", flush=True)
                break
            ck = torch.load(os.path.join(args.outdir, "latest.pt"))
            model.load_state_dict(ck["model"])
            if head is not None and "head" in ck:
                head.load_state_dict(ck["head"])
            lr_scale *= 0.5
            opt = torch.optim.AdamW(params, lr=args.lr * lr_scale,
                                    betas=(0.9, 0.95), weight_decay=0.1)
            continue
        if step % args.log_every == 0:
            rec = {"step": step, "lr": lr_at(step) * lr_scale, "loss": round(loss.item(), 4),
                   "gnorm": round(gnorm, 3), "elapsed_s": int(time.time() - t0)}
            if args.head_type == "mob":
                rec["ent"] = round(ent.item(), 3)
                rec["nll"] = round(nll.item(), 4)
            print(json.dumps(rec), flush=True)
            flog.write(json.dumps(rec) + "\n")
            flog.flush()
        if step % args.panel_every == 0:
            model.eval()
            if args.head_type == "native":
                rec = {"step": step, "panel": True,
                       "dev_nll": round(dev_native_nll(), 4)}
            else:
                dn, lz, neff, ndead = dev_mob_nll()
                rec = {"step": step, "panel": True, "dev_nll": round(dn, 4),
                       "logZ": round(lz, 4), "n_eff": round(neff, 1), "dead": ndead}
            model.train()
            print(json.dumps(rec), flush=True)
            flog.write(json.dumps(rec) + "\n")
            flog.flush()
        if step % args.save_every == 0:
            save(step)
    save(step)
    # final generations (MoB twin: cascade rerank with its OWN embedding table)
    tok = AutoTokenizer.from_pretrained(os.path.join(
        args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master"))
    gens = {}
    model.eval()
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            for _ in range(24):
                if args.head_type == "native":
                    nt = model(ids, use_cache=False).logits[0, -1].argmax().item()
                else:
                    h1 = model.model(ids, use_cache=False).last_hidden_state[:, -1]
                    bl1, pl1 = head(h1)
                    nt = decode_rerank_native(bl1, F.log_softmax(pl1, -1), codes_all,
                                              model.get_input_embeddings().weight, h1)[0].item()
                ids = torch.cat([ids, torch.tensor([[nt]], device="cuda")], 1)
        gens[prompt] = tok.decode(ids[0].tolist())
    with open(os.path.join(args.outdir, "generations.json"), "w", encoding="utf-8") as f:
        json.dump(gens, f, ensure_ascii=False, indent=2)
    flog.close()
    print("P5_TWIN_DONE", flush=True)


if __name__ == "__main__":
    main()
