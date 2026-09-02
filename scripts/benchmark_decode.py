# Inference benchmark: native lm_head+softmax vs distilled MoB head.
# Three layers: (1) head-only step latency on GPU (bandwidth-bound regime) and
# CPU (edge-like), (2) end-to-end greedy decode tokens/sec with KV cache,
# (3) analytic per-token bandwidth & FLOPs table.
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob_common import H, M, MoBHead, decode_greedy

V = 128256
MODEL_DIR = r"E:\Trash_things\llm_codec\LLM-Research--Llama-3.2-1B\snapshots\master"
PROMPT = "The little boy was playing with"


def bench_gpu(fn, warmup=20, reps=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def bench_cpu(fn, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--cpu-e2e", action="store_true",
                    help="also run end-to-end greedy decode on CPU (fp32, slow)")
    ap.add_argument("--cpu-tokens", type=int, default=32)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).cuda().eval()

    cb = np.load(r"E:\Trash_things\llm_codec\data\codebook.npz")
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    vid_to_tok = torch.full((1 << M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().cuda()] = \
        torch.arange(len(cb["vertex_id"]), device="cuda")

    heads = {}
    for tag, path, k, w in [
        ("MoB-K32", r"E:\Trash_things\llm_codec\runs\p2_local_10m_e3\head_final.pt", 32, 2458),
        ("MoB-K128", r"E:\Trash_things\llm_codec\runs\head_latest_wsd.pt", 128, 3072),
    ]:
        head = MoBHead(k=k, width=w).cuda()
        head.load_state_dict(torch.load(path))
        head.eval()
        heads[tag] = head

    E = model.model.embed_tokens.weight  # [V, 2048] bf16 (tied = lm_head)

    # ---------- analytic table ----------
    n_params_total = sum(p.numel() for p in model.parameters())
    n_backbone = n_params_total - V * H  # tied: table counted once
    print("== analytic per-decode-step (native vs MoB) ==")
    for tag, head in [("native", None)] + list(heads.items()):
        if head is None:
            head_bytes = V * H * 2  # bf16 table read
            head_flops = 2 * V * H
        else:
            head_bytes = sum(p.numel() for p in head.parameters()) * 4  # fp32
            k, w = head.k, head.trunk[0].out_features
            head_flops = 2 * (H * w + 2 * w * w + w * (k * M + k))
        total_bytes = n_backbone * 2 + head_bytes
        print(f"{tag:>9}: head reads {head_bytes/2**20:7.1f} MiB, head FLOPs {head_flops/1e6:6.1f}M, "
              f"total bytes/token {total_bytes/2**30:.3f} GiB")

    # ---------- head-only latency, GPU ----------
    print("\n== head-only step latency (GPU, bf16 in, fp32 head) ==")
    print(f"{'batch':>6} | {'native':>10} | {'MoB-K32':>10} | {'MoB-K128':>10}")
    for B in [1, 16, 64, 256]:
        h = torch.randn(B, H, dtype=torch.bfloat16, device="cuda")

        def f_native():
            logits = F.linear(h, E)
            torch.softmax(logits.float(), dim=-1).argmax(-1)

        def f_mob32():
            bl, pl = heads["MoB-K32"](h.float())
            decode_greedy(bl, F.log_softmax(pl, -1), vid_to_tok, codes_all)

        def f_mob128():
            bl, pl = heads["MoB-K128"](h.float())
            decode_greedy(bl, F.log_softmax(pl, -1), vid_to_tok, codes_all)

        t_n = bench_gpu(f_native)
        t_32 = bench_gpu(f_mob32)
        t_128 = bench_gpu(f_mob128)
        print(f"{B:>6} | {t_n*1e3:9.3f}ms | {t_32*1e3:9.3f}ms | {t_128*1e3:9.3f}ms")

    # ---------- head-only latency, CPU ----------
    print("\n== head-only step latency (CPU, fp32, batch 1) ==")
    E_cpu = E.float().cpu()
    h_cpu = torch.randn(1, H)
    head32_cpu = MoBHead(k=32, width=2458)
    head32_cpu.load_state_dict(torch.load(
        r"E:\Trash_things\llm_codec\runs\p2_local_10m_e3\head_final.pt"))
    head32_cpu.eval()
    head128_cpu = MoBHead(k=128, width=3072)
    head128_cpu.load_state_dict(torch.load(
        r"E:\Trash_things\llm_codec\runs\head_latest_wsd.pt"))
    head128_cpu.eval()
    vid_cpu = vid_to_tok.cpu()
    codes_cpu = codes_all.cpu()

    def c_native():
        torch.softmax(F.linear(h_cpu, E_cpu), dim=-1).argmax(-1)

    def c_mob32():
        bl, pl = head32_cpu(h_cpu)
        decode_greedy(bl, F.log_softmax(pl, -1), vid_cpu, codes_cpu)

    def c_mob128():
        bl, pl = head128_cpu(h_cpu)
        decode_greedy(bl, F.log_softmax(pl, -1), vid_cpu, codes_cpu)

    t_cn = bench_cpu(c_native)
    t_cm = bench_cpu(c_mob32)
    t_cm128 = bench_cpu(c_mob128)
    print(f"native: {t_cn*1e3:9.3f}ms   MoB-K32: {t_cm*1e3:9.3f}ms ({t_cn/t_cm:.1f}x)   "
          f"MoB-K128-wsd: {t_cm128*1e3:9.3f}ms ({t_cn/t_cm128:.1f}x)")

    # ---------- end-to-end greedy decode ----------
    print("\n== end-to-end greedy decode (KV cache, prefill excluded) ==")
    ids0 = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    n_new = args.gen_tokens

    def run_native():
        with torch.no_grad():
            out = model(ids0, use_cache=True)
            past = out.past_key_values
            nt = out.logits[0, -1].argmax().item()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_new):
                out = model(torch.tensor([[nt]], device="cuda"),
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                nt = out.logits[0, -1].argmax().item()
            torch.cuda.synchronize()
            return n_new / (time.perf_counter() - t0)

    def run_mob(head):
        with torch.no_grad():
            out = model.model(ids0, use_cache=True)
            past = out.past_key_values
            h = out.last_hidden_state[:, -1].float()
            bl, pl = head(h)
            nt = decode_greedy(bl, F.log_softmax(pl, -1), vid_to_tok, codes_all)[0].item()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_new):
                out = model.model(torch.tensor([[nt]], device="cuda"),
                                  past_key_values=past, use_cache=True)
                past = out.past_key_values
                h = out.last_hidden_state[:, -1].float()
                bl, pl = head(h)
                nt = decode_greedy(bl, F.log_softmax(pl, -1), vid_to_tok, codes_all)[0].item()
            torch.cuda.synchronize()
            return n_new / (time.perf_counter() - t0)

    r_n = run_native()
    r_32 = run_mob(heads["MoB-K32"])
    r_128 = run_mob(heads["MoB-K128"])
    print(f"native: {r_n:.1f} tok/s   MoB-K32: {r_32:.1f} tok/s ({r_32/r_n:.3f}x)   "
          f"MoB-K128: {r_128:.1f} tok/s ({r_128/r_n:.3f}x)")

    # ---------- end-to-end greedy decode on CPU (edge-like) ----------
    if args.cpu_e2e:
        print("\n== end-to-end greedy decode on CPU (fp32) ==")
        model_cpu = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
        vid_cpu = vid_to_tok.cpu()
        codes_cpu = codes_all.cpu()
        heads_cpu = {}
        for tag, path, k, w in [
            ("MoB-K32", r"E:\Trash_things\llm_codec\runs\p2_local_10m_e3\head_final.pt", 32, 2458),
            ("MoB-K128", r"E:\Trash_things\llm_codec\runs\head_latest_wsd.pt", 128, 3072),
        ]:
            hd = MoBHead(k=k, width=w)
            hd.load_state_dict(torch.load(path, map_location="cpu"))
            hd.eval()
            heads_cpu[tag] = hd
        ids0c = tok(PROMPT, return_tensors="pt").input_ids
        n_new = args.cpu_tokens

        def run_native_cpu():
            with torch.no_grad():
                out = model_cpu(ids0c, use_cache=True)
                past = out.past_key_values
                nt = out.logits[0, -1].argmax().item()
                t0 = time.perf_counter()
                for _ in range(n_new):
                    out = model_cpu(torch.tensor([[nt]]), past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nt = out.logits[0, -1].argmax().item()
                return n_new / (time.perf_counter() - t0)

        def run_mob_cpu(head):
            with torch.no_grad():
                out = model_cpu.model(ids0c, use_cache=True)
                past = out.past_key_values
                h = out.last_hidden_state[:, -1].float()
                bl, pl = head(h)
                nt = decode_greedy(bl, F.log_softmax(pl, -1), vid_cpu, codes_cpu)[0].item()
                t0 = time.perf_counter()
                for _ in range(n_new):
                    out = model_cpu.model(torch.tensor([[nt]]), past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    h = out.last_hidden_state[:, -1].float()
                    bl, pl = head(h)
                    nt = decode_greedy(bl, F.log_softmax(pl, -1), vid_cpu, codes_cpu)[0].item()
                return n_new / (time.perf_counter() - t0)

        rc_n = run_native_cpu()
        rc_32 = run_mob_cpu(heads_cpu["MoB-K32"])
        rc_128 = run_mob_cpu(heads_cpu["MoB-K128"])
        print(f"native: {rc_n:.2f} tok/s   MoB-K32: {rc_32:.2f} tok/s ({rc_32/rc_n:.3f}x)   "
              f"MoB-K128: {rc_128:.2f} tok/s ({rc_128/rc_n:.3f}x)")


if __name__ == "__main__":
    main()
