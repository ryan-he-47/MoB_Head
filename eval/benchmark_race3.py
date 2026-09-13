# Race3 efficiency benchmark (user request 2026-09-13): comprehensive re-test
# of the three race heads on SmolLM2-1.7B, RTX 4060 Laptop (8GB).
#   A. head-only latency: GPU bf16 batch1 / batch256, CPU fp32 batch1 (native
#      lm_head matvec as reference)
#   B. decode-path step cost: round / mixture_topk(32) / roundflip / cascade
#      rerank(roundflip) — GPU + CPU, batch 1
#   C. e2e greedy generation tok/s: native vs MoB-round vs MoB-rerank-roundflip
#      at batch 1 / 32 / 128 (KV cache)
#   D. analytic table: params, bytes/step, FLOPs/step
# Honest-bandwidth note: SmolLM2 V=49152 (~2.6x smaller than Llama 128256), so
# the native-side bandwidth is proportionally smaller; report both.
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob.mob_common import (MoBHead, ResMLPHead, SwiGLUHead, mixture_topk,
                            propose_roundflip, decode_rerank_native)


def bench(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def build_head(args, tag):
    if tag == "control":
        return MoBHead(k=args.k, width=3072, depth=3, m=args.m)
    if tag == "resmlp":
        return ResMLPHead(k=args.k, width=3072, depth=5, m=args.m)
    if tag == "swiglu":
        return SwiGLUHead(k=args.k, hidden=7300, blocks=1, m=args.m)
    raise ValueError(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"E:\Trash_things\llm_codec")
    ap.add_argument("--model-dir", default=r"E:\Trash_things\llm_codec\models\SmolLM2-1.7B")
    ap.add_argument("--codebook", default="codebook_smollm2.npz")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--heads", default="control,resmlp,swiglu")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--out", default=r"E:\Trash_things\llm_codec\runs\race3_benchmark.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(dev).eval()
    E = model.get_input_embeddings().weight  # [V,h] bf16 tied
    V, h_dim = E.shape
    cb = np.load(os.path.join(args.data_root, "data", args.codebook))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).to(dev)
    args.m = int(codes_all.shape[1])  # codebook is the source of truth for M
    vid_to_tok = torch.full((1 << args.m,), -1, dtype=torch.long, device=dev)
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().to(dev)] = \
        torch.arange(len(cb["vertex_id"]), device=dev)

    heads = {}
    for tag in args.heads.split(","):
        path = os.path.join(args.data_root, "runs", f"race3_head_{tag}_final.pt")
        hd = build_head(args, tag)
        hd.load_state_dict(torch.load(path, map_location="cpu"))
        heads[tag] = hd.to(dev).eval()
    params = {tag: sum(p.numel() for p in hd.parameters()) for tag, hd in heads.items()}

    R = {"model": "SmolLM2-1.7B", "V": V, "h": h_dim, "device": torch.cuda.get_device_name(0),
         "head_params_M": {t: round(p / 1e6, 1) for t, p in params.items()}}

    # ---------- D. analytic table ----------
    table = {"native": {"params": V * h_dim,
                        "bytes_fp16": V * h_dim * 2, "bytes_fp32": V * h_dim * 4,
                        "flops": 2 * V * h_dim}}
    for tag, p in params.items():
        table[tag] = {"params": p, "bytes_fp16": p * 2, "bytes_fp32": p * 4, "flops": 2 * p}
    R["analytic_per_step"] = {k: {kk: (round(vv / 1e6, 2) if kk != "params" else vv)
                                  for kk, vv in v.items()} for k, v in table.items()}

    # ---------- A. head-only latency ----------
    h1 = torch.randn(1, h_dim, device=dev, dtype=torch.bfloat16)
    h256 = torch.randn(256, h_dim, device=dev, dtype=torch.bfloat16)
    lat = {}
    with torch.no_grad():
        # native lm_head = F.linear(h, E)
        lat["native_gpu_b1_ms"] = bench(lambda: F.linear(h1, E), 200) * 1e3
        lat["native_gpu_b256_ms"] = bench(lambda: F.linear(h256, E), 50) * 1e3
        h1f, h256f = h1.float(), h256.float()   # heads run fp32 (as in training)
        for tag, hd in heads.items():
            lat[f"{tag}_gpu_b1_ms"] = bench(lambda: hd(h1f), 200) * 1e3
            lat[f"{tag}_gpu_b256_ms"] = bench(lambda: hd(h256f), 50) * 1e3
    # CPU fp32
    E32 = E.float().cpu()
    h1c = torch.randn(1, h_dim)
    lat["native_cpu_b1_ms"] = bench(lambda: F.linear(h1c, E32), 30) * 1e3
    for tag in args.heads.split(","):
        hd_cpu = build_head(args, tag)
        hd_cpu.load_state_dict(torch.load(os.path.join(args.data_root, "runs", f"race3_head_{tag}_final.pt"),
                                          map_location="cpu"))
        hd_cpu.float().eval()
        with torch.no_grad():
            lat[f"{tag}_cpu_b1_ms"] = bench(lambda: hd_cpu(h1c), 30) * 1e3
        del hd_cpu
    R["head_latency"] = {k: round(v, 3) for k, v in lat.items()}
    print("A done", flush=True)

    # ---------- B. decode-path step cost (best head = swiglu) ----------
    hd = heads["swiglu"]
    with torch.no_grad():
        bl, pl = hd(h1.float())
        log_pi = F.log_softmax(pl, -1)
        codes_i = codes_all  # [V,m] fp32

        def round_step():
            ktop = log_pi[0].topk(1, -1).indices
            bits = (bl[0, ktop] > 0).long()
            shifts = (1 << torch.arange(args.m - 1, -1, -1, device=dev)).long()
            return vid_to_tok[(bits * shifts).sum(-1)]

        def mixture_step():
            return mixture_topk(bl, log_pi, codes_i, 32)

        def roundflip_step():
            return propose_roundflip(bl, log_pi, codes_i, vid_to_tok)

        def rerank_step():
            toks = propose_roundflip(bl, log_pi, codes_i, vid_to_tok)  # 1-D tensor, n==1
            return toks.gather(0, (E[toks].float() @ h1[0].float()).argmax(0, keepdim=True))

        dec = {
            "round_gpu_ms": bench(round_step, 100) * 1e3,
            "mixture_topk32_gpu_ms": bench(mixture_step, 100) * 1e3,
            "roundflip_gpu_ms": bench(roundflip_step, 50) * 1e3,
            "rerank_roundflip_gpu_ms": bench(rerank_step, 50) * 1e3,
        }
    R["decode_step_gpu"] = {k: round(v, 3) for k, v in dec.items()}
    print("B done", flush=True)

    # ---------- C. e2e greedy generation ----------
    prompt = "The little boy was playing with a"
    ids0 = tok(prompt, return_tensors="pt").input_ids.to(dev)

    def gen_native(bs, n_tok):
        ids = ids0.repeat(bs, 1)
        with torch.no_grad():
            out = model(ids, use_cache=True)
            past = out.past_key_values
            cur = out.logits[:, -1].argmax(-1, keepdim=True)
            for _ in range(n_tok - 1):
                out = model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                cur = out.logits[:, -1].argmax(-1, keepdim=True)

    def gen_mob(hd, bs, n_tok, rerank=False):
        ids = ids0.repeat(bs, 1)
        with torch.no_grad():
            out = model(ids, use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            hs_last = out.hidden_states[-1][:, -1]
            for _ in range(n_tok):
                bl_, pl_ = hd(hs_last.float())
                lp_ = F.log_softmax(pl_, -1)
                if rerank:
                    rows = []
                    for b in range(bs):  # propose_roundflip is n_batch==1 only
                        tk = propose_roundflip(bl_[b:b + 1], lp_[b:b + 1], codes_all, vid_to_tok)
                        rows.append(tk.gather(0, (E[tk].float() @ hs_last[b].float()).argmax(0, keepdim=True)))
                    cur = torch.stack(rows)
                else:
                    ktop = lp_.topk(1, -1).indices.squeeze(1)          # [B]
                    bits = (bl_.gather(1, ktop.view(-1, 1, 1).expand(-1, 1, args.m)) > 0)
                    shifts = (1 << torch.arange(args.m - 1, -1, -1, device=dev)).long()
                    cur = vid_to_tok[(bits.long() * shifts).sum(-1)]    # [B]
                    cur = cur.clamp_min(0).view(-1, 1)
                out = model(cur, past_key_values=past, use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                hs_last = out.hidden_states[-1][:, -1]

    e2e = {}
    for bs, nt in ((1, args.gen_tokens), (32, args.gen_tokens), (128, 32)):
        t = bench(lambda: gen_native(bs, nt), 3, warmup=1)
        e2e[f"native_b{bs}_tokps"] = round(bs * nt / t, 1)
        hd = heads["swiglu"]
        t = bench(lambda: gen_mob(hd, bs, nt, rerank=False), 3, warmup=1)
        e2e[f"mob_round_b{bs}_tokps"] = round(bs * nt / t, 1)
        if bs <= 32:
            t = bench(lambda: gen_mob(hd, bs, nt, rerank=True), 3, warmup=1)
            e2e[f"mob_rerank_roundflip_b{bs}_tokps"] = round(bs * nt / t, 1)
        print(f"C bs={bs} done", flush=True)
    R["e2e_tok_per_s"] = e2e

    print(json.dumps(R, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
