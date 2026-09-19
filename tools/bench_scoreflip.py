"""Equal-bandwidth benchmark: MoB scoreflip-256 cascade vs CFHead-T4 (2026-09-19).

Both sides read ~95 MiB/step of method state (CFHead-T4: 512-tile INT6 index
map theoretical 95.3 MiB / uint8 runtime 127.1 MiB; MoB swiglu head fp16
95.0 MiB / fp32 190 MiB). Both then gather ~256 rows of the tied table (~1 MiB).

Phase A (fidelity, paired positions): per position compute native top-1 (fp32),
then for each arm: candidate recall AND the actual cascade/fine-stage pick
agreement vs native argmax:
  t4:        CFHeadRT T4 cache, k=256, repo tie semantics
  scoreflip: propose_scoreflip(256) + E-row rerank
  roundflip: propose_roundflip defaults + rerank (85-class reference)
  mixture:   mixture_topk(256) + rerank (exact ceiling)
Phase B (latency): per-step wall time of each decode path on GPU (cuda events,
128 steps) and CPU (32 steps), proposal included, rerank included; native full
lm_head matvec as the reference.

Run from repo root: python -m tools.bench_scoreflip [options]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from mob.mob_common import (SwiGLUHead, mixture_topk, propose_roundflip,
                            propose_scoreflip)
from eval.cfhead_rt import CFHeadRT


def rerank_pick(E, h_row, cand):
    scores = h_row.to(E.dtype) @ E[cand].T
    return cand[scores.argmax()]


@torch.no_grad()
def phase_fidelity(model, head, codes_all, vid_to_tok, rt_t4, dev, args, V):
    T = 512
    n_chunks = min(args.n_chunks, len(dev) // (T + 1))
    chunks = torch.from_numpy(dev[: n_chunks * (T + 1)].reshape(n_chunks, T + 1).copy()).long().cuda()
    chunks = chunks[:: args.chunk_stride]
    E = model.get_output_embeddings().weight.detach()
    arms = ("t4", "scoreflip", "roundflip", "mixture")
    rec = {a: {"hit": [], "agree": [], "n": []} for a in arms}
    t0 = time.perf_counter()
    total = chunks.shape[0] * T
    for ci in range(chunks.shape[0]):
        x = chunks[ci:ci + 1]
        out = model(x[:, :-1], output_hidden_states=True)
        h = out.hidden_states[-1].float().reshape(-1, 2048)
        nat1 = out.logits.reshape(-1, V).float().argmax(-1)
        del out
        bl, pl = head(h)
        log_pi = F.log_softmax(pl, dim=-1)
        for j in range(h.shape[0]):
            hj = h[j]
            bl_j, lp_j = bl[j:j + 1], log_pi[j:j + 1]
            n1 = nat1[j]
            # CFHead-T4
            cand = rt_t4.candidates(hj.unsqueeze(0), 256)[0]
            fl = rt_t4.fine_logits(hj.unsqueeze(0), cand.unsqueeze(0))[0]
            maxf = fl.max()
            pick = cand.masked_fill(fl != maxf, V).min()
            rec["t4"]["hit"].append(bool((cand == n1).any().item()))
            rec["t4"]["agree"].append(bool((pick == n1).item()))
            rec["t4"]["n"].append(256)
            # MoB arms
            for a, c in (("scoreflip", propose_scoreflip(bl_j, lp_j, codes_all, vid_to_tok,
                                                         per_comp_cap=32, max_cand=256)),
                         ("roundflip", propose_roundflip(bl_j, lp_j, codes_all, vid_to_tok)),
                         ("mixture", mixture_topk(bl_j, lp_j, codes_all, 256))):
                rec[a]["hit"].append(bool((c == n1).any().item()))
                rec[a]["agree"].append(bool((rerank_pick(E, hj, c) == n1).item()))
                rec[a]["n"].append(c.numel())
        done = (ci + 1) * T
        print(f"  [fid] {done}/{total} ({done / (time.perf_counter() - t0):.1f} pos/s)",
              flush=True)
    return {a: {"recall": float(np.mean(r["hit"])),
                "cascade_agreement": float(np.mean(r["agree"])),
                "mean_candidates": float(np.mean(r["n"]))}
            for a, r in rec.items()}


@torch.no_grad()
def phase_latency(model, head, codes_all, vid_to_tok, rt_t4, h_steps_gpu, h_steps_cpu, V, args):
    E_gpu = model.get_output_embeddings().weight.detach()
    E_cpu = E_gpu.cpu()
    w_gpu = E_gpu
    w_cpu_fp32 = E_cpu.float()

    def time_step(fn, device, steps, warm=8):
        for f in steps[:warm]:
            fn(f)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for f in steps[warm:]:
            fn(f)
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / max(len(steps) - warm, 1) * 1000.0

    def make_paths(hs, head_dev, codes, v2t, rt, E, dev):
        def p_native(h):
            if dev == "cuda":
                F.linear(h.to(w_gpu.dtype), w_gpu).argmax(-1)
            else:
                F.linear(h, w_cpu_fp32).argmax(-1)
        def p_t4(h):
            rt.select(h, 256)
        def p_scoreflip(h):
            bl, pl = head_dev(h)
            lp = F.log_softmax(pl, -1)
            cand = propose_scoreflip(bl, lp, codes, v2t, per_comp_cap=32, max_cand=256)
            scores = h[0].to(E.dtype) @ E[cand].T
            cand[scores.argmax()]
        def p_roundflip(h):
            bl, pl = head_dev(h)
            lp = F.log_softmax(pl, -1)
            cand = propose_roundflip(bl, lp, codes, v2t)
            scores = h[0].to(E.dtype) @ E[cand].T
            cand[scores.argmax()]
        def p_mixture(h):
            bl, pl = head_dev(h)
            lp = F.log_softmax(pl, -1)
            cand = mixture_topk(bl, lp, codes, 256)
            scores = h[0].to(E.dtype) @ E[cand].T
            cand[scores.argmax()]
        return {"native": p_native, "cfhead_t4": p_t4, "mob_scoreflip256": p_scoreflip,
                "mob_roundflip85": p_roundflip, "mob_mixture256": p_mixture}

    head_cpu = SwiGLUHead(k=args.k, hidden=args.hidden, blocks=args.blocks,
                          m=codes_all.shape[1])
    head_cpu.load_state_dict(torch.load(args.head, map_location="cpu"))
    head_cpu.eval()

    out = {}
    paths = make_paths(None, head, codes_all, vid_to_tok, rt_t4, E_gpu, "cuda")
    for name, fn in paths.items():
        out[f"gpu_{name}"] = round(time_step(fn, "cuda", h_steps_gpu), 3)
    codes_cpu = codes_all.cpu()
    v2t_cpu = vid_to_tok.cpu()
    rt_cpu = CFHeadRT(args.cfhead_t4_cache, E_cpu, torch.device("cpu"))
    paths = make_paths(None, head_cpu, codes_cpu, v2t_cpu, rt_cpu, E_cpu, "cpu")
    for name, fn in paths.items():
        out[f"cpu_{name}"] = round(time_step(fn, "cpu", h_steps_cpu), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--cfhead-t4-cache", required=True)
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--dev-pattern", default="qwen35_dev_{name}_tokens.npy")
    ap.add_argument("--dev-names", nargs="+", default=["en", "zh"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=7300)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--n-chunks", type=int, default=32)
    ap.add_argument("--chunk-stride", type=int, default=16)
    ap.add_argument("--lat-steps", type=int, default=128)
    ap.add_argument("--lat-steps-cpu", type=int, default=32)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16).cuda().eval()
    cb = np.load(args.codebook)
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    V, CB_M = codes_all.shape
    vid_to_tok = torch.full((1 << CB_M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().cuda()] = torch.arange(V, device="cuda")
    head = SwiGLUHead(k=args.k, hidden=args.hidden, blocks=args.blocks, m=CB_M)
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    head.cuda().eval()
    rt_t4 = CFHeadRT(args.cfhead_t4_cache,
                     model.get_output_embeddings().weight.detach(), torch.device("cuda"))

    result = {"fidelity": {}}
    for name in args.dev_names:
        dev = np.load(Path(args.dev_dir) / args.dev_pattern.format(name=name))
        print(f"=== fidelity {name} ===", flush=True)
        result["fidelity"][name] = phase_fidelity(
            model, head, codes_all, vid_to_tok, rt_t4, dev, args, V)

    print("=== latency ===", flush=True)
    # collect real hidden states for latency: one chunk forward, keep h rows
    dev = np.load(Path(args.dev_dir) / args.dev_pattern.format(name=args.dev_names[0]))
    x = torch.from_numpy(dev[: 2 * 513].reshape(2, 513)[:, :-1].copy()).long().cuda()
    with torch.no_grad():
        h_all = model(x, output_hidden_states=True).hidden_states[-1].float().reshape(-1, 2048)
    idx = torch.randint(0, h_all.shape[0], (max(args.lat_steps, args.lat_steps_cpu) + 8,))
    h_gpu_steps = [h_all[i: i + 1] for i in idx[: args.lat_steps + 8]]
    h_cpu_steps = [h_all[i: i + 1].cpu() for i in idx[: args.lat_steps_cpu + 8]]
    result["latency_ms_per_step"] = phase_latency(
        model, head, codes_all, vid_to_tok, rt_t4, h_gpu_steps, h_cpu_steps, V, args)

    print(json.dumps(result, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
