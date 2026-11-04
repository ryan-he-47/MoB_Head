# P3 acceptance eval: normalized PPL of the MoB head vs the native head (same
# frozen backbone, same dev positions) + decode agreement + generation samples.
# Spec: docs/01-method.md (architecture study).
# Usage: python -m eval.p3_eval --head runs/p2_10m/head_final.pt --outdir runs/p3_eval
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob.mob_common import H, M, MoBHead, ResMLPHead, SwiGLUHead, mob_mix_logprob, decode_greedy, decode_mixture_argmax, decode_rerank_native


def load_tokens(tok, data_glob, cache_path, eos_id):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    all_ids = []
    for path in sorted(glob.glob(data_glob)):
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for s in range(0, len(lines), 2000):
            for ids in tok(lines[s:s + 2000]).input_ids:
                all_ids.extend(ids + [eos_id])
        print(f"tokenized {path} ({len(lines)} lines)")
    arr = np.asarray(all_ids, dtype=np.int32)
    np.save(cache_path, arr)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r".")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--head", required=True)
    ap.add_argument("--outdir", default="runs/p3_eval")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-chunks", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pos-chunk", type=int, default=64,
                    help="positions per sub-chunk for the logZ pass")
    ap.add_argument("--backbone-dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="V100 (Volta) has no bf16 tensor cores - use float16 there")
    ap.add_argument("--k", type=int, default=32, help="mixture components (must match checkpoint)")
    ap.add_argument("--width", type=int, default=2458, help="head MLP width (must match checkpoint)")
    ap.add_argument("--depth", type=int, default=3, help="head MLP layers (must match checkpoint)")
    ap.add_argument("--head-type", choices=["mob", "resmlp", "swiglu"], default="mob",
                    help="head architecture (must match checkpoint)")
    ap.add_argument("--hidden", type=int, default=7300, help="swiglu FFN hidden width")
    ap.add_argument("--blocks", type=int, default=1, help="swiglu block count")
    ap.add_argument("--token-cache", default=None,
                    help="eval directly on this token cache (default: babylm_dev_tokens.npy)")
    ap.add_argument("--codebook", default="codebook.npz",
                    help="codebook filename under <data-root>/data/")
    ap.add_argument("--gen-tokens", type=int, default=40)
    ap.add_argument("--decode", choices=["round", "mixture", "rerank"], default="round",
                    help="greedy decode path for generation samples: round = pick-component+round "
                         "(default, comparable with earlier runs); mixture = mixture-argmax over all "
                         "valid vertices; rerank = mixture top-32 proposal + exact native-logit "
                         "rescoring (cascade; see docs/01-method.md). "
                         "The agreement metric always uses round; mixture-argmax agreement is "
                         "recall@1 in eval_topk_recall.py")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK console safety
    os.makedirs(args.outdir, exist_ok=True)

    model_dir = args.model_dir or os.path.join(
        args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=getattr(torch, args.backbone_dtype)).cuda()
    model.eval()

    cb = np.load(os.path.join(args.data_root, "data", args.codebook))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    CB_M = codes_all.shape[1]
    vid_to_tok = torch.full((1 << CB_M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().cuda()] = \
        torch.arange(len(cb["vertex_id"]), device="cuda")

    if args.head_type == "resmlp":
        head = ResMLPHead(k=args.k, width=args.width, depth=args.depth, m=CB_M).cuda()
    elif args.head_type == "swiglu":
        head = SwiGLUHead(k=args.k, hidden=args.hidden, blocks=args.blocks, m=CB_M).cuda()
    else:
        head = MoBHead(k=args.k, width=args.width, depth=args.depth, m=CB_M).cuda()
    head.load_state_dict(torch.load(args.head))
    head.eval()

    dev_ids = None
    if args.token_cache:
        dev = np.load(args.token_cache)
    else:
        dev = load_tokens(
            tok, os.path.join(args.data_root, "data", "babylm_data", "babylm_dev", "*.dev"),
            os.path.join(args.data_root, "data", "babylm_dev_tokens.npy"), tok.eos_token_id)
    T = args.seq_len
    n_chunks = min(args.n_chunks, len(dev) // (T + 1))
    chunks = torch.from_numpy(dev[: n_chunks * (T + 1)].reshape(n_chunks, T + 1).copy()).long().cuda()
    print(f"eval: {n_chunks} chunks x {T} positions")

    native_nlls, mob_nlls, agrees = [], [], []
    with torch.no_grad():
        for i in range(0, n_chunks, args.batch):
            x = chunks[i:i + args.batch]
            out = model(x[:, :-1], output_hidden_states=True)
            tgt = x[:, 1:]
            h = out.hidden_states[-1].float().reshape(-1, H)
            logits = out.logits.float()
            native_nlls.append(F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none"))
            teacher_top1 = logits.argmax(-1).reshape(-1)

            bl, pl = head(h)
            log_pi = F.log_softmax(pl, dim=-1)
            codes_true = codes_all[tgt.reshape(-1)].unsqueeze(1)        # [n,1,17]
            logp_true = mob_mix_logprob(bl, log_pi, codes_true)[:, 0]   # [n]
            n = h.shape[0]
            logZs = []
            for s in range(0, n, args.pos_chunk):
                b = bl[s:s + args.pos_chunk]
                term = torch.einsum("vm,nkm->nvk", codes_all, b)        # [pc,V,K]
                base = F.logsigmoid(-b).sum(-1).unsqueeze(1)            # [pc,1,K]
                mix = torch.logsumexp(
                    log_pi[s:s + args.pos_chunk].unsqueeze(1) + term + base, dim=-1)
                logZs.append(torch.logsumexp(mix, dim=-1))              # [pc]
            logZ = torch.cat(logZs)
            mob_nlls.append(logZ - logp_true)
            agrees.append(decode_greedy(bl, log_pi, vid_to_tok, codes_all, m=CB_M) == teacher_top1)
            print(f"batch {i // args.batch + 1}/{(n_chunks + args.batch - 1) // args.batch} done")

    native_nll = torch.cat(native_nlls).mean().item()
    mob_nll = torch.cat(mob_nlls).mean().item()
    agree = torch.cat(agrees).float().mean().item()
    metrics = {
        "n_positions": int(n_chunks * T),
        "native_head": {"nll": native_nll, "ppl": float(np.exp(native_nll))},
        "mob_head": {"nll": mob_nll, "ppl": float(np.exp(mob_nll))},
        "ppl_ratio_mob_over_native": float(np.exp(mob_nll - native_nll)),
        "decode_top1_agreement": agree,
    }
    print(json.dumps(metrics, indent=2))
    with open(os.path.join(args.outdir, "p3_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- generation samples (milestone material) ----
    prompts = [
        "The little boy was playing with",
        "Once upon a time there was a",
        "My mother said I should",
        "The dog ran to the park and",
    ]

    def mob_step(ids, tau=None):
        out = model(ids, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1].float()
        bl, pl = head(h)
        log_pi = F.log_softmax(pl, dim=-1)
        if tau is None:  # greedy
            if args.decode == "mixture":
                return decode_mixture_argmax(bl, log_pi, codes_all)[0].item()
            if args.decode == "rerank":
                E = model.get_input_embeddings().weight
                return decode_rerank_native(bl, log_pi, codes_all, E, h)[0].item()
            return decode_greedy(bl, log_pi, vid_to_tok, codes_all, m=CB_M)[0].item()
        probs = F.softmax(pl / tau, dim=-1)
        k = torch.multinomial(probs, 1).item()
        bits = (bl[0, k] > 0).float()
        vid = 0
        for j in range(CB_M):
            vid |= bits[j].long().item() << (CB_M - 1 - j)
        t = vid_to_tok[vid].item()
        if t < 0:
            d = (codes_all != bits.unsqueeze(0)).sum(-1)
            t = d.argmin().item()
        return t

    lines = []
    for prompt in prompts:
        lines.append(f"prompt: {prompt!r}")
        for tag, tau in [("greedy", None), ("tau=0.7", 0.7), ("tau=1.0", 1.0)]:
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                for _ in range(args.gen_tokens):
                    nt = mob_step(ids, tau)
                    ids = torch.cat([ids, torch.tensor([[nt]], device="cuda")], 1)
            lines.append(f"  MoB {tag}: {tok.decode(ids[0].tolist())!r}")
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            for _ in range(args.gen_tokens):
                nt = model(ids).logits[0, -1].argmax().item()
                ids = torch.cat([ids, torch.tensor([[nt]], device="cuda")], 1)
        lines.append(f"  teacher greedy: {tok.decode(ids[0].tolist())!r}")
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(args.outdir, "generations.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
