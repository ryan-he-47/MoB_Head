# P4 feasibility probe (docs/14 section 4, user decision 2026-09-03):
# can the PURE MoB-NLL loss (no teacher) train the BACKBONE through the
# 17-bit code bottleneck? System: Llama-3.2-1B fp16 + WSD final MoB head fp32,
# native lm_head unused. Loss = -log p_mix(code(gt)) + small pi-entropy reg
# (self-normalizing over the hypercube: each component's Bernoulli product
# sums to 1 over all vertices, so no logZ term is needed for training —
# same property the p2 distillation loss relies on; logZ only appears in the
# eval metric for comparability with the renormalized PPL baseline).
# Modes: A = unfreeze last-N layers + final norm; B = peft LoRA on last-N layers.
# Health checks at intervals: MoB dev NLL (logZ-renormalized), native-head PPL
# of the DRIFTED backbone (tied embeddings let us plug the native head back in;
# >2x degradation = backbone bending, rollback criterion), short generations.
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob_common import H, M, MoBHead, mob_mix_logprob, decode_rerank_native

ROOT = r"E:\Trash_things\llm_codec"
MODEL_DIR = os.path.join(ROOT, "LLM-Research--Llama-3.2-1B", "snapshots", "master")
CACHE = os.path.join(ROOT, "data", "fineweb_probe_tokens.npz")
PROMPTS = ["The little boy was playing with", "My mother said I should"]


def build_cache(tok, n_train_docs=4000, n_dev_docs=200):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    tr, dv = [], []
    for i, d in enumerate(ds):
        ids = tok(d["text"]).input_ids + [tok.eos_token_id]
        (tr if i < n_train_docs else dv).extend(ids)
        if i + 1 >= n_train_docs + n_dev_docs:
            break
        if (i + 1) % 500 == 0:
            print(f"  cached {i + 1} docs")
    np.savez(CACHE, train=np.array(tr, dtype=np.int32), dev=np.array(dv, dtype=np.int32))
    print(f"cache built: train {len(tr)} tokens, dev {len(dv)} tokens -> {CACHE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unfreeze", "lora"], required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--n-layers", type=int, default=2, help="unfreeze mode: last N layers + norm")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-layers", type=int, default=4)
    ap.add_argument("--lambda-ent", type=float, default=0.001)
    ap.add_argument("--freeze-head", action="store_true",
                    help="freeze the MoB head entirely (control arm: backbone-only co-adaptation)")
    ap.add_argument("--norm-trainable", action="store_true",
                    help="lora mode: also unfreeze the final norm (mirrors the cluster recipe)")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--head", default=os.path.join(ROOT, "runs", "head_final_wsd.pt"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_path = args.out or os.path.join(ROOT, "runs", f"p4_probe_{args.mode}.json")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if not os.path.exists(CACHE):
        build_cache(tok)
    cache = np.load(CACHE)
    T = args.seq_len
    tr = cache["train"]
    n_chunks = len(tr) // (T + 1)
    chunks = torch.from_numpy(tr[: n_chunks * (T + 1)].reshape(n_chunks, T + 1).astype(np.int64))
    dv = cache["dev"]
    n_dev = min(64, len(dv) // (T + 1))
    dev_chunks = torch.from_numpy(dv[: n_dev * (T + 1)].reshape(n_dev, T + 1).astype(np.int64)).cuda()

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for p in model.parameters():
        p.requires_grad_(False)
    if args.mode == "unfreeze":
        for layer in model.model.layers[-args.n_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in model.model.norm.parameters():
            p.requires_grad_(True)
        # mixed precision done right: trainable layers in fp32 (Adam-safe),
        # frozen trunk stays bf16; autocast casts on the fly during forward
        for layer in model.model.layers[-args.n_layers:]:
            layer.float()
        model.model.norm.float()
        bb_trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        from peft import LoraConfig, get_peft_model
        n_layers = len(model.model.layers)
        cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"],
                         layers_to_transform=list(range(n_layers - args.lora_layers, n_layers)),
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.enable_input_require_grads()
        backbone = model.get_base_model().model  # LlamaModel inside the wrapped CausalLM
        if args.norm_trainable:
            for p in backbone.norm.parameters():
                p.requires_grad_(True)
                p.data = p.data.float()
        bb_trainable = [p for p in model.parameters() if p.requires_grad]
        for p in bb_trainable:  # LoRA params in fp32 for Adam stability
            p.data = p.data.float()
    if args.mode == "unfreeze":
        backbone = model.model
    print(f"backbone trainable params: {sum(p.numel() for p in bb_trainable) / 1e6:.1f}M")

    head = MoBHead(k=128, width=3072)
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    head = head.cuda().float()
    if args.freeze_head:
        for p in head.parameters():
            p.requires_grad_(False)
    cb = np.load(os.path.join(ROOT, "data", "codebook.npz"))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    E = model.get_input_embeddings().weight

    head_params = [p for p in head.parameters() if p.requires_grad]
    groups = [{"params": bb_trainable, "lr": args.lr_backbone}]
    if head_params:
        groups.append({"params": head_params, "lr": args.lr_head})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    @torch.no_grad()
    def evaluate(tag):
        model.eval()
        # MoB dev NLL with logZ renormalization (chunked; K=128 -> chunk 8)
        nlls = []
        for i in range(0, n_dev, 2):
            x = dev_chunks[i:i + 2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
        mob_nll = torch.cat(nlls).mean().item()
        # native-head PPL of the (drifted) backbone
        nat = []
        for i in range(0, n_dev, 2):
            x = dev_chunks[i:i + 2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x[:, :-1], use_cache=False).logits
            nat.append(F.cross_entropy(logits.float().reshape(-1, 128256), x[:, 1:].reshape(-1),
                                       reduction="none"))
        nat_nll = torch.cat(nat).mean().item()
        # short generations: cascade rerank (MoB head) + native greedy (drifted backbone)
        gens = {}
        for prompt in PROMPTS:
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                for _ in range(24):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h1 = backbone(ids, use_cache=False).last_hidden_state
                    h1 = h1[:, -1].float()
                    bl1, pl1 = head(h1)
                    nt = decode_rerank_native(bl1, F.log_softmax(pl1, -1), codes_all, E, h1)[0].item()
                    ids = torch.cat([ids, torch.tensor([[nt]], device="cuda")], 1)
            gens[f"mob_rerank: {prompt!r}"] = tok.decode(ids[0].tolist())
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                for _ in range(24):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = model(ids, use_cache=False).logits
                    nt = lg[0, -1].float().argmax().item()
                    ids = torch.cat([ids, torch.tensor([[nt]], device="cuda")], 1)
            gens[f"native_greedy: {prompt!r}"] = tok.decode(ids[0].tolist())
        model.train()
        rec = {"tag": tag, "mob_dev_nll": round(mob_nll, 4),
               "mob_dev_ppl": round(float(np.exp(mob_nll)), 2),
               "native_dev_nll": round(nat_nll, 4),
               "native_dev_ppl": round(float(np.exp(nat_nll)), 2), "gens": gens}
        print(json.dumps(rec, ensure_ascii=False))
        return rec

    history = [evaluate("step-0")]
    order = np.random.RandomState(123).permutation(n_chunks)
    t0 = time.time()
    oi = 0
    for step in range(1, args.steps + 1):
        if oi + args.batch > n_chunks:
            oi = 0
        x = chunks[order[oi:oi + args.batch]].cuda()
        oi += args.batch
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = backbone(x[:, :-1], use_cache=False).last_hidden_state
        h = h.float().reshape(-1, H)
        bl, pl = head(h)
        log_pi = F.log_softmax(pl, dim=-1)
        tgt = x[:, 1:].reshape(-1)
        lp_true = mob_mix_logprob(bl, log_pi, codes_all[tgt].unsqueeze(1))[:, 0]
        nll = -lp_true.mean()
        pi = log_pi.exp()
        ent = (pi * log_pi).sum(-1).mean()
        loss = nll + args.lambda_ent * ent
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bb_trainable + head_params, 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step}/{args.steps}  nll {nll.item():.4f}  ent {ent.item():.3f}  "
                  f"{(time.time() - t0) / step:.2f}s/step")
        if step % args.eval_every == 0:
            history.append(evaluate(f"step-{step}"))
    result = {"args": vars(args), "history": history}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
