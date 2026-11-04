# P2 production distillation: MoB head on full BabyLM-10M, frozen Llama-3.2-1B.
# Verified pipeline pieces are carried over from scripts/p1_smoke.py.
# Usage (cluster): python scripts/p2_train.py --data-root /path/to/llm_codec --outdir runs/p2_10m
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob.mob_common import H, K, M, MoBHead, ResMLPHead, SwiGLUHead, mob_mix_logprob, decode_greedy, ecc_parity_matrix, ecc_extend


def load_or_pack_tokens(tok, data_glob, cache_path, eos_id):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    all_ids = []
    for path in sorted(glob.glob(data_glob)):
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for s in range(0, len(lines), 2000):  # batched fast-tokenizer calls
            for ids in tok(lines[s:s + 2000]).input_ids:
                all_ids.extend(ids + [eos_id])
        print(f"tokenized {path} ({len(lines)} lines)")
    arr = np.asarray(all_ids, dtype=np.int32)
    np.save(cache_path, arr)
    return arr


def load_or_pack_tokens_hf(tok, cache_path, eos_id, dataset, config, split,
                           field, max_docs, skip_docs=0):
    """Stream an HF dataset (e.g. HuggingFaceFW/fineweb-edu sample-10BT),
    tokenize the text field in batches, pack with EOS. If the dataset was
    pre-downloaded to HF cache, set HF_HUB_OFFLINE=1 to run without network.
    skip_docs: discard the first N docs (used to hold out a dev slice)."""
    if os.path.exists(cache_path):
        return np.load(cache_path)
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split, streaming=True)
    all_ids, buf, n_docs = [], [], 0
    for doc in ds:
        n_docs += 1
        if n_docs <= skip_docs:
            continue
        text = doc[field].strip()
        if text:
            buf.append(text)
        if len(buf) >= 1000:
            for ids in tok(buf).input_ids:
                all_ids.extend(ids + [eos_id])
            print(f"tokenized {n_docs} docs -> {len(all_ids):,} tokens")
            buf = []
        if max_docs and (n_docs - skip_docs) >= max_docs:
            break
    if buf:
        for ids in tok(buf).input_ids:
            all_ids.extend(ids + [eos_id])
    arr = np.asarray(all_ids, dtype=np.int32)
    np.save(cache_path, arr)
    return arr


def kmeans_vertex_init(codes_gpu, tids, tps, init_c, k, n_samp=20000, seed=42):
    g = torch.Generator().manual_seed(seed)  # CPU ops only
    # marginalize (position, topk) pair weights over token ids first: sampling
    # tokens from the V-vector is statistically identical to sampling pairs and
    # mapping to tokens, but V stays << 2^24 (torch.multinomial hard limit).
    # (batch-16 warmup broke the pair-flat path: 32768x1024 = 33.5M > 2^24.)
    V = codes_gpu.shape[0]
    w_v = torch.zeros(V, dtype=torch.float32, device=tids.device)
    w_v.scatter_add_(0, tids.reshape(-1), tps.reshape(-1).float())
    samp_tok = torch.multinomial(w_v.cpu(), n_samp, replacement=True, generator=g).cuda()
    X = codes_gpu[samp_tok]
    cent = [X[torch.randint(n_samp, (1,)).item()]]
    for _ in range(k - 1):
        d = torch.cdist(X, torch.stack(cent)).min(1).values.clamp_min(1e-12)
        cent.append(X[torch.multinomial(d, 1).item()])
    cent = torch.stack(cent)
    for _ in range(30):
        assign = torch.cdist(X, cent).argmin(1)
        new = cent.clone()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new[j] = X[mask].mean(0)
        if torch.allclose(new, cent, atol=1e-6):
            break
        cent = new
    return ((cent > 0.5).float() * 2 - 1) * init_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r".")
    ap.add_argument("--model-dir", default=None,
                    help="default: <data-root>/LLM-Research--Llama-3.2-1B/snapshots/master")
    ap.add_argument("--head-type", choices=["mob", "resmlp", "swiglu"], default="mob")
    ap.add_argument("--hidden", type=int, default=7300, help="swiglu FFN hidden width")
    ap.add_argument("--blocks", type=int, default=1, help="swiglu block count")
    ap.add_argument("--outdir", default="runs/p2_10m")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"],
                    help="wsd = warmup-stable-decay (hold peak, decay last --decay-frac)")
    ap.add_argument("--decay-frac", type=float, default=0.35,
                    help="WSD: fraction of total steps spent in the decay phase")
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--k", type=int, default=32, help="mixture components")
    ap.add_argument("--width", type=int, default=2458, help="head MLP width")
    ap.add_argument("--depth", type=int, default=3, help="head MLP layers")
    ap.add_argument("--lambda-ent", type=float, default=0.01)
    ap.add_argument("--daem-t0", type=float, default=2.0)
    ap.add_argument("--daem-frac", type=float, default=0.10)
    ap.add_argument("--init-c", type=float, default=4.0)
    ap.add_argument("--loss-chunk", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="save head_latest.pt every N steps (0 = off; use for long runs)")
    ap.add_argument("--ckpt-keep", type=int, default=3,
                    help="archived full-state checkpoints (head_stepN.pt) to keep")
    ap.add_argument("--panel-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="debug: cap number of training chunks (0 = all)")
    ap.add_argument("--backbone-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="V100 (Volta) has no bf16 tensor cores - use float16 there")
    ap.add_argument("--data-format", default="babylm", choices=["babylm", "hf"])
    ap.add_argument("--hf-dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--hf-config", default="sample-10BT")
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--hf-field", default="text")
    ap.add_argument("--hf-max-docs", type=int, default=0,
                    help="debug: cap streamed HF docs (0 = all)")
    ap.add_argument("--dev-format", default="babylm", choices=["babylm", "hf"],
                    help="dev slice source; hf = first --hf-dev-docs docs of the stream")
    ap.add_argument("--hf-dev-docs", type=int, default=2000,
                    help="docs reserved for the HF dev slice (training skips them)")
    ap.add_argument("--hf-skip-docs", type=int, default=0,
                    help="training stream skips this many leading docs (set = --hf-dev-docs when --dev-format hf)")
    ap.add_argument("--token-cache", default=None, help="override token cache filename under <data-root>/data/")
    ap.add_argument("--codebook", default="codebook.npz", help="codebook filename under <data-root>/data/")
    ap.add_argument("--dev-cache", default="fineweb_dev_tokens.npy",
                    help="dev token cache filename under <data-root>/data/ (hf dev-format)")
    ap.add_argument("--ecc-bits", type=int, default=0,
                    help=">0: online ECC extension — append this many parity bits to every "
                         "code (deterministic XOR of data bits; INFRASTRUCTURE, experiments "
                         "disabled by default)")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    model_dir = args.model_dir or os.path.join(
        args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=getattr(torch, args.backbone_dtype)).cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cb = np.load(os.path.join(args.data_root, "data", args.codebook))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    CB_M = codes_all.shape[1]  # code length from the codebook itself (multi-model support)
    HEAD_M = CB_M + args.ecc_bits
    if args.ecc_bits:
        pmat = ecc_parity_matrix(CB_M, args.ecc_bits, seed=args.seed).cuda()
        codes_all_full = ecc_extend(codes_all, pmat)              # [V, HEAD_M]
        vids = (codes_all_full.long()
                * (1 << torch.arange(HEAD_M - 1, -1, -1, device="cuda")).long()).sum(-1)
    else:
        codes_all_full = codes_all
        vids = torch.from_numpy(cb["vertex_id"]).long().cuda()
    vid_to_tok = torch.full((1 << HEAD_M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[vids] = torch.arange(codes_all_full.shape[0], device="cuda")

    # ---- data ----
    if args.data_format == "hf":
        cache_name = args.token_cache or "fineweb_tokens.npy"
        cache = os.path.join(args.data_root, "data", cache_name)
        ids_all = load_or_pack_tokens_hf(
            tok, cache, tok.eos_token_id, args.hf_dataset, args.hf_config,
            args.hf_split, args.hf_field, args.hf_max_docs, args.hf_skip_docs)
    else:
        cache_name = args.token_cache or "babylm_10m_tokens.npy"
        cache = os.path.join(args.data_root, "data", cache_name)
        ids_all = load_or_pack_tokens(
            tok, os.path.join(args.data_root, "data", "babylm_data", "babylm_10M", "*.train"),
            cache, tok.eos_token_id)
    T = args.seq_len
    n_chunks = len(ids_all) // (T + 1)
    if args.max_chunks:
        n_chunks = min(n_chunks, args.max_chunks)
    chunks = np.lib.stride_tricks.as_strided(
        ids_all, shape=(n_chunks, T + 1), strides=(ids_all.strides[0] * T, ids_all.strides[0]))
    print(f"tokens={len(ids_all):,} chunks={n_chunks} x {T} (+1 for shift)")

    # dev slice (held out) for per-epoch decode agreement
    if args.dev_format == "hf":
        dev_ids = load_or_pack_tokens_hf(
            tok, os.path.join(args.data_root, "data", args.dev_cache),
            tok.eos_token_id, args.hf_dataset, args.hf_config,
            args.hf_split, args.hf_field, args.hf_dev_docs)
    else:
        dev_ids = load_or_pack_tokens(
            tok, os.path.join(args.data_root, "data", "babylm_data", "babylm_dev", "*.dev"),
            os.path.join(args.data_root, "data", "babylm_dev_tokens.npy"), tok.eos_token_id)
    dev_chunks = torch.from_numpy(
        dev_ids[: 8 * (T + 1)].reshape(8, T + 1).copy()).long().cuda()

    def extract(chunk_batch):
        """chunk_batch [b, T+1] long cuda -> h [n,2048] fp32, top-k ids/probs [n,topk]."""
        with torch.no_grad():
            out = model(chunk_batch[:, :-1], output_hidden_states=True)
            h = out.hidden_states[-1].float()
            logits = out.logits
            top = logits.topk(args.topk, dim=-1)
            w = torch.softmax(top.values.float(), dim=-1)
        b, t, _ = h.shape
        return h.reshape(b * t, H), top.indices.reshape(b * t, args.topk), \
            w.reshape(b * t, args.topk)

    with torch.no_grad():
        dev_h, dev_tids, dev_tps = extract(dev_chunks)
        dev_top1 = dev_tids[:, 0].clone()

    if args.head_type == "resmlp":
        head = ResMLPHead(k=args.k, width=args.width, depth=args.depth, m=HEAD_M).cuda()
    elif args.head_type == "swiglu":
        head = SwiGLUHead(k=args.k, hidden=args.hidden, blocks=args.blocks, m=HEAD_M).cuda()
    else:
        head = MoBHead(k=args.k, width=args.width, depth=args.depth, m=HEAD_M).cuda()
    resume_state = None
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume)
        if isinstance(ck, dict) and "head" in ck:
            head.load_state_dict(ck["head"])
            resume_state = ck
            print(f"resuming FULL state from {args.resume} (step {ck['step']}, epoch {ck['epoch']})")
        else:
            head.load_state_dict(ck)
            print(f"resumed head (weights only) from {args.resume}")
    else:
        with torch.no_grad():
            # init from the first training batches' target distribution
            warm_chunks = torch.from_numpy(chunks[: 4 * args.batch].copy()).long().cuda()
            warm_n = min(4 * args.batch, n_chunks)
            wlists = [extract(warm_chunks[i:i + args.batch])
                      for i in range(0, warm_n, args.batch)]
            init_logits = kmeans_vertex_init(
                codes_all_full, torch.cat([x[1] for x in wlists]),
                torch.cat([x[2] for x in wlists]), args.init_c, args.k, seed=args.seed)
            head.out.bias.zero_()
            head.out.bias[: args.k * HEAD_M] = init_logits.reshape(-1)
        del warm_chunks, wlists
        torch.cuda.empty_cache()

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    if resume_state is not None and "opt" in resume_state:
        opt.load_state_dict(resume_state["opt"])
        print("optimizer state restored")
    steps_per_epoch = n_chunks // args.batch
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, total_steps - args.warmup)
        if args.schedule == "wsd":
            decay_start = 1.0 - args.decay_frac
            if t < decay_start:
                return args.lr
            u = (t - decay_start) / args.decay_frac
            return args.lr * (0.1 + 0.9 * (1.0 - u))
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t)))

    def dev_metrics():
        with torch.no_grad():
            agrees, pis, nlls = [], [], []
            for s in range(0, dev_h.shape[0], args.loss_chunk):
                bl, pl = head(dev_h[s:s + args.loss_chunk])
                log_pi = F.log_softmax(pl, dim=-1)
                pis.append(log_pi.exp())
                toks = decode_greedy(bl, log_pi, vid_to_tok, codes_all_full, m=HEAD_M)
                agrees.append(toks == dev_top1[s:s + args.loss_chunk])
                codes = codes_all_full[dev_tids[s:s + args.loss_chunk]]
                mix = mob_mix_logprob(bl, log_pi, codes)
                nlls.append(-(dev_tps[s:s + args.loss_chunk] * mix).sum(-1))
            pi_mean = torch.cat(pis).mean(0)
            h_pi = -(pi_mean * pi_mean.clamp_min(1e-12).log()).sum().item()
            return (float(np.exp(h_pi)), int((pi_mean < 1e-4).sum().item()),
                    torch.cat(agrees).float().mean().item(),
                    torch.cat(nlls).mean().item())

    print(f"total steps={total_steps} ({steps_per_epoch}/epoch x {args.epochs})")
    log_path = os.path.join(args.outdir, "train_log.jsonl")
    flog = open(log_path, "a", encoding="utf-8")
    start_step = int(resume_state["step"]) if resume_state is not None else 0
    start_epoch = start_step // steps_per_epoch
    resume_order = resume_state.get("order") if resume_state is not None else None
    if resume_order is not None and torch.is_tensor(resume_order):
        resume_order = resume_order.cpu().numpy()
    step = start_step
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        if epoch == start_epoch and resume_order is not None:
            order = resume_order
        else:
            order = np.random.permutation(n_chunks)
        skip_steps = (step - epoch * steps_per_epoch) if epoch == start_epoch else 0
        run_loss = run_nll = run_ent = 0.0
        nb = 0
        for i in range(skip_steps * args.batch, steps_per_epoch * args.batch, args.batch):
            cb_idx = torch.from_numpy(
                chunks[order[i:i + args.batch]].copy()).long().cuda()
            h, tids, tps = extract(cb_idx)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            temp = args.daem_t0 + (1.0 - args.daem_t0) * min(
                1.0, step / (args.daem_frac * total_steps))
            nll_parts, ent_parts = [], []
            for s in range(0, h.shape[0], args.loss_chunk):
                bl, pl = head(h[s:s + args.loss_chunk])
                log_pi = F.log_softmax(pl / temp, dim=-1)
                codes = codes_all_full[tids[s:s + args.loss_chunk]]
                mix = mob_mix_logprob(bl, log_pi, codes)
                nll_parts.append(-(tps[s:s + args.loss_chunk] * mix).sum(-1))
                pi = log_pi.exp()
                ent_parts.append((pi * log_pi).sum(-1))
            nll = torch.cat(nll_parts).mean()
            ent = torch.cat(ent_parts).mean()
            loss = nll + args.lambda_ent * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()
            run_nll += nll.item()
            run_ent += ent.item()
            nb += 1
            if step % args.log_every == 0:
                rec = {"step": step, "epoch": epoch, "lr": lr_at(step),
                       "nll": run_nll / nb, "ent": run_ent / nb, "T": temp,
                       "elapsed_s": int(time.time() - t0)}
                print(rec)
                flog.write(json.dumps(rec) + "\n")
                flog.flush()
                run_loss = run_nll = run_ent = 0.0
                nb = 0
            if step % args.panel_every == 0 and step > 0:
                n_eff, dead, agree, dev_nll = dev_metrics()
                rec = {"step": step, "panel": True, "n_eff": round(n_eff, 1),
                       "dead": dead, "dev_top1_agree": round(agree, 4),
                       "dev_nll": round(dev_nll, 4)}
                print(rec)
                flog.write(json.dumps(rec) + "\n")
                flog.flush()
            if args.ckpt_every and step % args.ckpt_every == 0 and step > 0:
                # weights-only copy for eval scripts + full-state archive for true resume
                torch.save(head.state_dict(), os.path.join(args.outdir, "head_latest.pt"))
                torch.save({
                    "head": head.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step + 1,  # resume starts at this update index
                    "epoch": epoch,
                    "order": torch.from_numpy(np.asarray(order)),
                }, os.path.join(args.outdir, f"head_step{step + 1}.pt"))
                archived = sorted(
                    (f for f in os.listdir(args.outdir)
                     if f.startswith("head_step") and f.endswith(".pt")),
                    key=lambda f: int(f[len("head_step"):-len(".pt")]))
                for f in archived[: max(0, len(archived) - args.ckpt_keep)]:
                    os.remove(os.path.join(args.outdir, f))
            step += 1
        ckpt = os.path.join(args.outdir, f"head_epoch{epoch}.pt")
        torch.save(head.state_dict(), ckpt)
        n_eff, dead, agree, dev_nll = dev_metrics()
        print(f"[epoch {epoch}] saved {ckpt} | n_eff={n_eff:.1f} dead={dead} "
              f"dev_top1_agree={agree:.4f} dev_nll={dev_nll:.4f}")
    torch.save(head.state_dict(), os.path.join(args.outdir, "head_final.pt"))
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    flog.close()
    print("done.")


if __name__ == "__main__":
    main()
