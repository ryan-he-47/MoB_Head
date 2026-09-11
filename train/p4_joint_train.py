# P4 joint fine-tune (docs/14, user decision 2026-09-03): LoRA on the backbone
# (r=32, last 8 layers) + final norm trainable + MoB head, trained with the
# PURE MoB-NLL loss (no teacher) — the MoB head is the system's ONLY output
# head and the native lm_head never enters forward or loss.
# Acceptance framing (user-reframed 2026-09-03): MoB-system dev PPL vs the
# pure-native baseline; native-drift PPL is DIAGNOSTIC ONLY; OOD (BabyLM) MoB
# PPL is the catastrophic-forgetting canary.
# Mixed-precision recipe validated in the local probe: bf16 frozen trunk +
# fp32 trainable params (LoRA, norm, head) + autocast forward, head/loss in
# fp32 outside autocast (raw fp16+Adam NaNs within 50 steps).
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob.mob_common import H, M, MoBHead, mob_mix_logprob, decode_rerank_native

PROMPTS = ["The little boy was playing with", "My mother said I should"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"E:\Trash_things\llm_codec")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--outdir", default="runs/p4_joint")
    ap.add_argument("--token-cache", default="fineweb_train_tokens.npy")
    ap.add_argument("--dev-cache", default="fineweb_dev_tokens.npy")
    ap.add_argument("--ood-cache", default="babylm_dev_tokens.npy")
    ap.add_argument("--head", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-layers", type=int, default=8)
    ap.add_argument("--lambda-ent", type=float, default=0.001)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--panel-every", type=int, default=2500)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--resume-adapter", default=None,
                    help="peft adapter dir to continue from (head via --head)")
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="constant",
                    help="cosine decays both LR groups to ~5%% after warmup (polish runs)")
    ap.add_argument("--mode", choices=["lora", "unfreeze"], default="lora",
                    help="lora = peft adapter; unfreeze = full-parameter last N layers + norm "
                         "(escalation ladder rung 1, docs/14: LoRA's low-rank assumption does "
                         "not fit reshape-h-geometry tasks)")
    ap.add_argument("--n-layers", type=int, default=4, help="unfreeze mode: last N layers")
    ap.add_argument("--compute-dtype", choices=["bfloat16", "float16"], default="bfloat16",
                    help="trunk/autocast dtype. V100 (pre-Ampere): use float16 + GradScaler — "
                         "bf16 there runs through emulation paths that NaN'd full-parameter "
                         "training within 400 warmup steps (job 524864, docs/14)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    os.makedirs(args.outdir, exist_ok=True)
    model_dir = args.model_dir or os.path.join(
        args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master")

    tok = AutoTokenizer.from_pretrained(model_dir)
    T = args.seq_len
    data = np.load(os.path.join(args.data_root, "data", args.token_cache))
    n_chunks = len(data) // (T + 1)
    if args.max_chunks:
        n_chunks = min(n_chunks, args.max_chunks)
    chunks = np.ascontiguousarray(data[: n_chunks * (T + 1)].reshape(n_chunks, T + 1))
    dev = np.load(os.path.join(args.data_root, "data", args.dev_cache))
    n_dev = min(64, len(dev) // (T + 1))
    dev_chunks = torch.from_numpy(dev[: n_dev * (T + 1)].reshape(n_dev, T + 1).copy()).long().cuda()
    ood = np.load(os.path.join(args.data_root, "data", args.ood_cache))
    n_ood = min(64, len(ood) // (T + 1))
    ood_chunks = torch.from_numpy(ood[: n_ood * (T + 1)].reshape(n_ood, T + 1).copy()).long().cuda()
    print(f"train chunks {n_chunks}, dev {n_dev}, ood {n_ood}")

    compute_dtype = getattr(torch, args.compute_dtype)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=compute_dtype).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig, PeftModel, get_peft_model
    n_layers = len(model.model.layers)
    if args.mode == "unfreeze":
        backbone = model.model
        for layer in backbone.layers[-args.n_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
            layer.float()  # fp32 trainable layers (Adam-safe), trunk stays bf16
    elif args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        backbone = model.get_base_model().model
    else:
        cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"],
                         layers_to_transform=list(range(n_layers - args.lora_layers, n_layers)),
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        backbone = model.get_base_model().model
    model.enable_input_require_grads() if args.mode == "lora" else None
    for p in backbone.norm.parameters():  # final norm trainable (geometry lever)
        p.requires_grad_(True)
        p.data = p.data.float()
    bb_trainable = [p for p in model.parameters() if p.requires_grad]
    for p in bb_trainable:  # LoRA params fp32 for Adam stability
        p.data = p.data.float()
        print(f"backbone trainable params: {sum(p.numel() for p in bb_trainable) / 1e6:.2f}M")

    head = MoBHead(k=128, width=3072)
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    head = head.cuda().float()
    cb = np.load(os.path.join(args.data_root, "data", "codebook.npz"))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    E = model.get_input_embeddings().weight

    opt = torch.optim.AdamW([
        {"params": bb_trainable, "lr": args.lr_backbone},
        {"params": list(head.parameters()), "lr": args.lr_head},
    ], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(compute_dtype == torch.float16))

    def lr_at(step, base):
        if step < args.warmup:
            return base * (step + 1) / args.warmup
        if args.schedule == "cosine":
            t = (step - args.warmup) / max(1, args.steps - args.warmup)
            return base * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * t)))
        return base

    @torch.no_grad()
    def mob_nll(eval_chunks, max_b=2):
        nlls = []
        for i in range(0, len(eval_chunks), max_b):
            x = eval_chunks[i:i + max_b]
            with torch.autocast("cuda", dtype=compute_dtype):
                h = backbone(x[:, :-1], use_cache=False).last_hidden_state
            h = h.float().reshape(-1, H)
            bl, pl = head(h)
            log_pi = F.log_softmax(pl, dim=-1)
            tgt = x[:, 1:].reshape(-1)
            lp_true = mob_mix_logprob(bl, log_pi, codes_all[tgt].unsqueeze(1))[:, 0]
            logzs = []
            for s in range(0, h.shape[0], 8):
                b = bl[s:s + 8]
                term = torch.einsum("vm,nkm->nvk", codes_all, b)
                base = F.logsigmoid(-b).sum(-1).unsqueeze(1)
                mix = torch.logsumexp(log_pi[s:s + 8].unsqueeze(1) + term + base, dim=-1)
                logzs.append(torch.logsumexp(mix, dim=-1))
            nlls.append(torch.cat(logzs) - lp_true)
        return torch.cat(nlls).mean().item()

    @torch.no_grad()
    def native_nll(eval_chunks, max_b=2):
        out = []
        for i in range(0, len(eval_chunks), max_b):
            x = eval_chunks[i:i + max_b]
            with torch.autocast("cuda", dtype=compute_dtype):
                lg = model(x[:, :-1], use_cache=False).logits
            out.append(F.cross_entropy(lg.float().reshape(-1, 128256),
                                       x[:, 1:].reshape(-1), reduction="none"))
        return torch.cat(out).mean().item()

    @torch.no_grad()
    def panel(tag):
        model.eval()
        rec = {"step": tag, "mob_dev_nll": round(mob_nll(dev_chunks), 4),
               "native_dev_nll_diag": round(native_nll(dev_chunks), 4),
               "mob_ood_nll": round(mob_nll(ood_chunks), 4)}
        model.train()
        print(json.dumps(rec))
        flog.write(json.dumps({"panel": True, **rec}) + "\n")
        flog.flush()
        return rec

    def save(tag):
        torch.save(head.state_dict(), os.path.join(args.outdir, f"head_{tag}.pt"))
        if args.mode == "lora":
            model.save_pretrained(os.path.join(args.outdir, f"adapter_{tag}"))
        else:
            torch.save({f"layer{i}": backbone.layers[i].state_dict()
                        for i in range(n_layers - args.n_layers, n_layers)}
                       | {"norm": backbone.norm.state_dict()},
                       os.path.join(args.outdir, f"backbone_{tag}.pt"))

    flog = open(os.path.join(args.outdir, "train_log.jsonl"), "w", encoding="utf-8")
    json.dump({"args": vars(args)}, open(os.path.join(args.outdir, "config.json"), "w"))
    panel(0)
    save("step0")
    order = np.random.RandomState(args.seed).permutation(n_chunks)
    t0 = time.time()
    oi = 0
    model.train()
    for step in range(1, args.steps + 1):
        if oi + args.batch > n_chunks:
            oi = 0
        x = torch.from_numpy(chunks[order[oi:oi + args.batch]].copy()).long().cuda()
        oi += args.batch
        for g, base in zip(opt.param_groups, (args.lr_backbone, args.lr_head)):
            g["lr"] = lr_at(step, base)
        with torch.autocast("cuda", dtype=compute_dtype):
            h = backbone(x[:, :-1], use_cache=False).last_hidden_state
        h = h.float().reshape(-1, H)
        bl, pl = head(h)
        log_pi = F.log_softmax(pl, dim=-1)
        tgt = x[:, 1:].reshape(-1)
        nll = -mob_mix_logprob(bl, log_pi, codes_all[tgt].unsqueeze(1))[:, 0].mean()
        pi = log_pi.exp()
        ent = (pi * log_pi).sum(-1).mean()
        loss = nll + args.lambda_ent * ent
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(bb_trainable + list(head.parameters()), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % args.log_every == 0:
            rec = {"step": step, "lr_bb": opt.param_groups[0]["lr"],
                   "nll": round(nll.item(), 4), "ent": round(ent.item(), 3),
                   "elapsed_s": int(time.time() - t0)}
            print(json.dumps(rec))
            flog.write(json.dumps(rec) + "\n")
            flog.flush()
        if step % args.panel_every == 0:
            panel(step)
        if step % args.save_every == 0:
            save(f"step{step}")
    panel(args.steps)
    save("final")
    # final generations: cascade rerank (system output) + native greedy (diagnostic)
    gens = {}
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            for _ in range(32):
                with torch.autocast("cuda", dtype=compute_dtype):
                    h1 = backbone(ids, use_cache=False).last_hidden_state
                h1 = h1[:, -1].float()
                bl1, pl1 = head(h1)
                nt = decode_rerank_native(bl1, F.log_softmax(pl1, -1), codes_all, E, h1)[0].item()
                ids = torch.cat([ids, torch.tensor([[nt]], device="cuda")], 1)
        gens[f"mob_rerank: {prompt!r}"] = tok.decode(ids[0].tolist())
    with open(os.path.join(args.outdir, "generations.json"), "w", encoding="utf-8") as f:
        json.dump(gens, f, ensure_ascii=False, indent=2)
    flog.close()
    print("JOINT_TRAIN_DONE")


if __name__ == "__main__":
    main()
