"""CPU-offload latency benchmark.

Split: GPU runs the coarse stage; the small coarse outputs cross PCIe; CPU
runs the cheap proposal + gather from the RAM-resident vocab matrix + GEMV.

Paths:
  mob_offload    GPU: MoB head forward -> bl/pl [128,19] fp32 to CPU
                 CPU: numba scoreflip(256) -> cand -> E_ram[cand] -> GEMV rerank
  cfhead_offload GPU: CFHead-T4 coarse (Triton) -> top-256 ids to CPU
                 CPU: E_ram[cand] -> GEMV fine
  native_gpu     full lm_head matvec on GPU (reference)
  native_cpu     full lm_head GEMV on CPU from the RAM table (naive offload)

Metrics: per-step stage latencies (gpu / transfer / cpu) and end-to-end
greedy decode tok/s on one prompt (KV cache, backbone always on GPU).
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob.mob_common import SwiGLUHead
from mob.cpu_proposals import CPUProposals
from eval.cfhead_rt import CFHeadRT
from eval.cfhead_kernels import TRITON_AVAILABLE


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--cfhead-t4-cache", required=True)
    ap.add_argument("--prompt", default="The future of artificial intelligence is")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16).cuda().eval()
    cb = np.load(args.codebook)
    M = int(cb["codes"].shape[1])
    head = SwiGLUHead(k=128, hidden=7300, blocks=1, m=M)
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    head.cuda().eval()
    rt_t4 = CFHeadRT(args.cfhead_t4_cache,
                     model.get_output_embeddings().weight.detach(), torch.device("cuda"))
    cp = CPUProposals(args.codebook)
    E_gpu = model.get_output_embeddings().weight.detach()       # [V,H] bf16 GPU
    E_ram = E_gpu.float().cpu()                                  # [V,H] fp32 RAM (2GB)
    V, H = E_gpu.shape
    w_gpu = E_gpu
    print(f"V={V} H={H} triton={TRITON_AVAILABLE} E_ram={E_ram.numel() * 4 / 2 ** 20:.0f} MiB",
          flush=True)

    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()

    # ------------------------------------------------------------------
    # per-step workers (return next token id; record stage times)
    # ------------------------------------------------------------------
    stage = {"gpu": [], "xfer": [], "cpu": []}

    def sync():
        torch.cuda.synchronize()

    def mob_offload(h):
        sync(); t0 = time.perf_counter()
        bl, pl = head(h)
        log_pi = F.log_softmax(pl, dim=-1)
        sync(); stage["gpu"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        bl_c, lp_c = bl.cpu(), log_pi.cpu()
        stage["xfer"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        cand = cp.scoreflip(bl_c, lp_c)
        rows = E_ram[cand]
        pick = cand[(rows @ h[0].float().cpu()).argmax()]
        stage["cpu"].append(time.perf_counter() - t0)
        return int(pick)

    def cfhead_offload(h):
        sync(); t0 = time.perf_counter()
        cand = rt_t4.candidates(h, 256)
        sync(); stage["gpu"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        cand_c = cand[0].cpu()
        stage["xfer"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        rows = E_ram[cand_c]
        pick = cand_c[(rows @ h[0].float().cpu()).argmax()]
        stage["cpu"].append(time.perf_counter() - t0)
        return int(pick)

    def native_gpu(h):
        return int(F.linear(h.to(w_gpu.dtype), w_gpu).argmax(-1).item())

    def native_cpu(h):
        t0 = time.perf_counter()
        pick = int((E_ram @ h[0].float().cpu()).argmax().item())
        stage["cpu"].append(time.perf_counter() - t0)
        return pick

    paths = {"mob_offload": mob_offload, "cfhead_offload": cfhead_offload,
             "native_gpu": native_gpu, "native_cpu": native_cpu}

    # ------------------------------------------------------------------
    # end-to-end greedy decode per path (fresh prefill each path)
    # ------------------------------------------------------------------
    result = {"paths": {}, "triton": TRITON_AVAILABLE, "V": V, "H": H}
    for name, fn in paths.items():
        stage["gpu"], stage["xfer"], stage["cpu"] = [], [], []
        input_ids = ids.clone()
        past = None
        cur = input_ids
        t_start = time.perf_counter()
        n_gen = 0
        for step in range(args.steps):
            out = model(cur, past_key_values=past, use_cache=True,
                        output_hidden_states=True)
            past = out.past_key_values
            h = out.hidden_states[-1][:, -1]
            nt = fn(h)
            cur = torch.tensor([[nt]], device="cuda")
            n_gen += 1
        sync()
        total = time.perf_counter() - t_start
        entry = {
            "tok_per_s": round(n_gen / total, 2),
            "ms_per_step_e2e": round(total / n_gen * 1000, 2),
        }
        for k, v in (("gpu", stage["gpu"]), ("xfer", stage["xfer"]), ("cpu", stage["cpu"])):
            if v:
                entry[f"head_{k}_ms_mean"] = round(float(np.mean(v)) * 1000, 3)
                entry[f"head_{k}_ms_p95"] = round(float(np.quantile(v, 0.95)) * 1000, 3)
        result["paths"][name] = entry
        print(f"{name}: {json.dumps(entry)}", flush=True)
        del past
        torch.cuda.empty_cache()

    print(json.dumps(result, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
