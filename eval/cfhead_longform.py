# Long-form autoregressive head-to-head: native vs CFHead-Q vs CFHead-M vs
# MoB cascade on Qwen3.5-2B (docs/01-method.md,
# protocol A). J4 of the head-to-head plan.
#
# Protocol ported from the CFHead reproduction (E:/CFHead/CFHead/repro/
# longform_hit.py, repro/REPRO_REPORT.md section 3.6), extended to a multi-arm
# self-trajectory race:
#   * 6 prompts (2 ShareGPT + 2 LMSYS-Chat-1M + 2 LongBench hotpotqa/samsum),
#     flattened `role: content` transcripts, input fixed to 256 tokens
#     (truncate; left-pad only if a short prompt slips through the >=256-token
#     selection filter). Seed 99.
#   * Greedy decoding, batch size 1, HF KV cache, EOS disabled -> exactly
#     `--steps` tokens per (arm, prompt). One forward per step.
#   * Every arm generates its OWN trajectory ("self-context" bias is symmetric
#     across arms). At every step the native full lm_head argmax on the current
#     context is the ground truth for THAT trajectory. It is computed as an
#     fp32 shadow matmul from hidden_states[-1][:, -1] against a resident fp32
#     copy of the tied lm_head weight (fp32 because bf16 logit ties are
#     frequent, docs/01-method.md; [1,V] fp32 output ~1 MB, no chunking
#     needed at batch 1).
#       - hit       = native argmax in the arm's candidate set
#                     (native arm: candidate set = CFHead-Q coarse top-256 on
#                     the native trajectory -- the paper's Table-II
#                     containment).
#       - agreement = the arm's chosen token == native argmax.
#   * Repetition bucketing (docs/01-method.md): a step is
#     `repeat` if the chosen token already appeared in the previous
#     `--repeat-window` generated tokens of the same trajectory, else `novel`;
#     hit/agreement are reported split by bucket.
#
# Arms (--arms, comma list, subsettable):
#   native      full lm_head argmax (agreement == 1 by construction)
#   cfhead_q    CFHeadRT quality cache (T8/C64), coarse top-k=256 + exact fine
#               rescoring (same math as CFHeadRT.select, candidates reused for
#               the hit metric)
#   cfhead_m    CFHeadRT storage cache (T8/C4), coarse top-k=8192 + fine
#   mob_cascade SwiGLU MoB head (fp32) -> propose_roundflip candidates ->
#               decode_rerank_native(proposal="roundflip") pick
#
# Outputs in --out-dir:
#   longform_{arm}_p{idx}.jsonl   per-step records, flushed every
#                                 --checkpoint-every steps
#   longform_summary_{arm}.json   per-prompt + overall + per-bucket rates,
#                                 miss step indices, mean candidate count,
#                                 generated token ids (drift analysis)
#   longform_summary_all.json     all completed arms combined
#
# Run (cluster, from the repo root -- see slurm/slurm_cfhead_longform.sbatch):
#   python -m eval.cfhead_longform --steps 8192
from __future__ import annotations

import os

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from eval.cfhead_rt import CFHeadRT
from mob.mob_common import SwiGLUHead, decode_rerank_native, propose_roundflip

SCRATCH = os.path.join(os.environ.get("MOB_SCRATCH", "."), "")
CFHEAD_Q_CACHE_NAME = "cfhead_v3_V248320_H2048_T8_C64_K256_iters10_seed99_bfloat16.pt"
CFHEAD_M_CACHE_NAME = "cfhead_v3_V248320_H2048_T8_C4_K256_iters10_seed99_bfloat16.pt"
ALL_ARMS = ("native", "cfhead_q", "cfhead_m", "mob_cascade")
CANDIDATE_SOURCE = {
    "native": "cfhead_q coarse top-256 on the native trajectory (Table-II containment)",
    "cfhead_q": "cfhead_q coarse top-256",
    "cfhead_m": "cfhead_m coarse top-8192",
    "mob_cascade": "propose_roundflip (round-32 modes + uncertain-bit flips, cap 8/comp)",
}


# ---------------------------------------------------------------------------
# Input construction (ported from the repro protocol)
# ---------------------------------------------------------------------------
def transcript_sharegpt(convs):
    return "\n".join(f"{c.get('from', '')}: {c.get('value', '')}" for c in convs)


def transcript_roles(convs):
    return "\n".join(f"{c.get('role', '')}: {c.get('content', '')}" for c in convs)


def read_json_or_jsonl(path):
    """Cluster exports may be a JSON list (sharegpt/lmsys dumps) or JSONL."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    return obj if isinstance(obj, list) else [obj]


def load_sharegpt(path, pool):
    items = read_json_or_jsonl(path)
    out = []
    for idx in range(min(pool, len(items))):
        it = items[idx]
        out.append({
            "dataset": "sharegpt",
            "sample_id": f"{it.get('id', idx)}#{idx}",
            "index": idx,
            "source": str(path),
            "text": transcript_sharegpt(it.get("conversations", [])),
        })
    return out


def load_lmsys(path, pool):
    out = []
    for idx, row in enumerate(read_json_or_jsonl(path)):
        if len(out) >= pool:
            break
        out.append({
            "dataset": "lmsys_chat_1m",
            "sample_id": f"{row.get('conversation_id', idx)}#{idx}",
            "index": idx,
            "source": str(path),
            "text": transcript_roles(row.get("conversation", [])),
        })
    return out


def load_longbench(path, pool, task):
    out = []
    for idx, row in enumerate(read_json_or_jsonl(path)):
        if idx >= pool:
            break
        ctx = (row.get("context") or "").strip()
        inp = (row.get("input") or "").strip()
        # Standard LongBench task prompt; `input` alone is only ~20 tokens for
        # hotpotqa/samsum, so `context` supplies the fixed 256-token input.
        text = f"{ctx}\n\n{inp}".strip() if inp else ctx
        out.append({
            "dataset": "longbench",
            "sample_id": f"{task}#{idx}",
            "index": idx,
            "source": str(path),
            "text": text,
        })
    return out


def tokenize_fixed(tokenizer, text, length):
    """Exactly `length` tokens: truncate long, left-pad short (mask=0) so the
    final position is always a real token."""
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    if len(ids) >= length:
        return ids[:length], [1] * length
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    padn = length - len(ids)
    return [pad] * padn + ids, [0] * padn + [1] * len(ids)


def raw_token_len(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def select_samples(tokenizer, pool, n, length):
    """First `n` pool items with >= length real tokens, else fill with short ones."""
    long_ok = [s for s in pool if raw_token_len(tokenizer, s["text"]) >= length]
    rest = [s for s in pool if raw_token_len(tokenizer, s["text"]) < length]
    chosen = long_ok[:n]
    if len(chosen) < n:
        chosen += rest[: n - len(chosen)]
    return chosen


def build_prompts(tokenizer, args):
    """The repro protocol's 6 inputs: 2 ShareGPT + 2 LMSYS + 1 hotpotqa + 1 samsum."""
    data_dir = Path(args.data_dir)
    n = args.samples_per_dataset
    samples = []
    samples += select_samples(
        tokenizer,
        load_sharegpt(need_file(data_dir / args.sharegpt_file, "ShareGPT export"), max(20, n)),
        n, args.input_len)
    samples += select_samples(
        tokenizer,
        load_lmsys(need_file(data_dir / args.lmsys_file, "LMSYS export"), max(40, n)),
        n, args.input_len)
    per_task = max(1, n // 2)
    for task, fname in (("hotpotqa", args.longbench_hotpotqa_file),
                        ("samsum", args.longbench_samsum_file)):
        samples += select_samples(
            tokenizer,
            load_longbench(need_file(data_dir / fname, f"LongBench {task} export"),
                           max(10, per_task), task),
            per_task, args.input_len)
    for i, s in enumerate(samples):
        s["prompt_idx"] = i
    return samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def need_file(path, what):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[ERROR] {what} not found: {p}\n"
                         "[HINT] the CFHead cache-build job / data upload must finish first; "
                         "check the --*-file/--*-dir flags.")
    return p


def resolve_cache(cache_dir, name):
    p = Path(name)
    if not p.is_absolute():
        p = Path(cache_dir) / p
    return need_file(p, "CFHead cache")


# ---------------------------------------------------------------------------
# One trajectory: (arm x prompt), greedy, EOS disabled, shadow-scored
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_prompt(arm, prompt, model, tokenizer, rt, args, device, jsonl_path):
    ids, mask = tokenize_fixed(tokenizer, prompt["text"], args.input_len)
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.tensor([mask], dtype=torch.long, device=device)
    past = None
    gen = []
    records = []
    n_hit = n_agree = n_cand_total = 0
    buckets = {"repeat": [0, 0, 0], "novel": [0, 0, 0]}  # n, hit, agree
    miss_hit, miss_agree = [], []
    t0 = time.perf_counter()

    fh = open(jsonl_path, "w", encoding="utf-8")
    for step in range(args.steps):
        out = model(input_ids=cur, attention_mask=attn, past_key_values=past,
                    use_cache=True, output_hidden_states=True, return_dict=True)
        past = out.past_key_values
        h = out.hidden_states[-1][:, -1]                     # [1,H] head input
        # shadow ground truth: native full-head argmax on the current context
        shadow = F.linear(h.float(), rt.w_fp32)              # [1,V] fp32
        native_top1 = int(shadow.argmax(-1).item())

        if arm == "native":
            cand = rt.cf_q.candidates(h, args.k_cfhead_q)[0]
            chosen = native_top1
        elif arm in ("cfhead_q", "cfhead_m"):
            rt_cf = rt.cf_q if arm == "cfhead_q" else rt.cf_m
            k = args.k_cfhead_q if arm == "cfhead_q" else args.k_cfhead_m
            cand2 = rt_cf.candidates(h, k)                   # [1,k]
            scores = rt_cf.fine_logits(h, cand2)             # exact rescoring
            # repo tie semantics (CFHead.py masked_fill/min): lowest token id
            # among fine-max ties — plain argmax would under-report agreement
            maxl = scores.max(dim=-1, keepdim=True).values
            chosen = int(cand2.masked_fill(scores != maxl, rt.w_fp32.shape[0])[0].min().item())
            cand = cand2[0]
        elif arm == "mob_cascade":
            h_f = h.float()
            bl, pl = rt.mob(h_f)
            log_pi = F.log_softmax(pl, dim=-1)
            cand = propose_roundflip(bl, log_pi, rt.codes_all, rt.vid_to_tok)
            # decode_rerank_native(proposal="roundflip") re-runs
            # propose_roundflip internally (deterministic -> same set); kept as
            # the mainline call so the pick matches eval/p3_eval.py exactly.
            chosen = int(decode_rerank_native(
                bl, log_pi, rt.codes_all, rt.E, h_f,
                proposal="roundflip", vid_to_tok=rt.vid_to_tok)[0].item())
        else:
            raise ValueError(f"unknown arm {arm!r}")

        hit = int(bool((cand == native_top1).any().item()))
        agree = int(chosen == native_top1)
        repeat = int(chosen in gen[-args.repeat_window:])
        n_hit += hit
        n_agree += agree
        n_cand_total += int(cand.numel())
        b = "repeat" if repeat else "novel"
        buckets[b][0] += 1
        buckets[b][1] += hit
        buckets[b][2] += agree
        if not hit:
            miss_hit.append(step)
        if not agree:
            miss_agree.append(step)
        records.append({
            "arm": arm, "prompt_idx": prompt["prompt_idx"],
            "dataset": prompt["dataset"], "sample_id": prompt["sample_id"],
            "step": step, "hit": hit, "agree": agree, "repeat": repeat,
            "n_cand": int(cand.numel()), "chosen": chosen, "native_top1": native_top1,
        })

        gen.append(chosen)
        cur = torch.tensor([[chosen]], dtype=torch.long, device=device)
        attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device=device)], dim=1)

        done = step + 1
        if done % args.checkpoint_every == 0 or step == args.steps - 1:
            fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
            fh.flush()
            records.clear()
            el = time.perf_counter() - t0
            print(f"    {arm}/p{prompt['prompt_idx']} {done}/{args.steps} "
                  f"hit={n_hit / done:.4f} agree={n_agree / done:.4f} "
                  f"({1000.0 * el / done:.1f} ms/step)", flush=True)
    fh.close()

    elapsed = time.perf_counter() - t0
    del past
    gc.collect()
    torch.cuda.empty_cache()

    n = len(gen)
    return {
        "arm": arm,
        "prompt_idx": prompt["prompt_idx"],
        "dataset": prompt["dataset"],
        "sample_id": prompt["sample_id"],
        "source": prompt["source"],
        "input_tokens": len(ids),
        "input_real_tokens": int(sum(mask)),
        "steps": n,
        "hit_count": n_hit,
        "hit_rate": n_hit / n,
        "agree_count": n_agree,
        "agree_rate": n_agree / n,
        "buckets": {
            name: {"n": v[0], "hit": v[1], "agree": v[2],
                   "hit_rate": (v[1] / v[0]) if v[0] else None,
                   "agree_rate": (v[2] / v[0]) if v[0] else None}
            for name, v in buckets.items()
        },
        "miss_steps_hit": miss_hit,
        "miss_steps_agree": miss_agree,
        "mean_n_cand": n_cand_total / n,
        "generation_seconds": elapsed,
        "sec_per_step": elapsed / n,
        "generated_token_ids": gen,
    }


def aggregate_arm(arm, per_prompt, args):
    tot = sum(p["steps"] for p in per_prompt)
    hit = sum(p["hit_count"] for p in per_prompt)
    agree = sum(p["agree_count"] for p in per_prompt)
    buckets = {}
    for name in ("repeat", "novel"):
        bn = sum(p["buckets"][name]["n"] for p in per_prompt)
        bh = sum(p["buckets"][name]["hit"] for p in per_prompt)
        ba = sum(p["buckets"][name]["agree"] for p in per_prompt)
        buckets[name] = {"n": bn, "hit": bh, "agree": ba,
                         "hit_rate": (bh / bn) if bn else None,
                         "agree_rate": (ba / bn) if bn else None}
    mean_cand = sum(p["mean_n_cand"] * p["steps"] for p in per_prompt) / tot
    return {
        "arm": arm,
        "candidate_set": CANDIDATE_SOURCE[arm],
        "prompts": per_prompt,
        "overall": {
            "steps": tot,
            "hit_count": hit,
            "hit_rate": hit / tot,
            "agree_count": agree,
            "agree_rate": agree / tot,
            "buckets": buckets,
            "mean_n_cand": mean_cand,
        },
        "config": run_config(args),
    }


def run_config(args):
    return {
        "model_dir": args.model_dir,
        "backbone_dtype": args.backbone_dtype,
        "steps": args.steps,
        "seed": args.seed,
        "input_len": args.input_len,
        "repeat_window": args.repeat_window,
        "checkpoint_every": args.checkpoint_every,
        "k_cfhead_q": args.k_cfhead_q,
        "k_cfhead_m": args.k_cfhead_m,
        "cfhead_q_cache": args.cfhead_q_cache,
        "cfhead_m_cache": args.cfhead_m_cache,
        "mob_head": args.mob_head,
        "mob_k": args.mob_k,
        "mob_hidden": args.mob_hidden,
        "mob_blocks": args.mob_blocks,
        "codebook": args.codebook,
        "decoding": "greedy, batch 1, KV cache, EOS disabled; fp32 shadow native argmax",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="CFHead vs MoB long-form autoregressive head-to-head (Qwen3.5-2B)")
    p.add_argument("--model-dir", default=f"{SCRATCH}/models/Qwen3.5-2B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--backbone-dtype", default="bfloat16",
                   choices=["bfloat16", "float16"],
                   help="protocol default bf16 (matches the repro + cache build dtype); "
                        "float16 is the fast Volta option if the 24h box gets tight")
    p.add_argument("--cfhead-cache-dir", default=f"{SCRATCH}/cfhead_cache")
    p.add_argument("--cfhead-q-cache", default=CFHEAD_Q_CACHE_NAME,
                   help="quality-tier cache filename under --cfhead-cache-dir (or absolute path)")
    p.add_argument("--cfhead-m-cache", default=CFHEAD_M_CACHE_NAME,
                   help="storage-tier cache filename under --cfhead-cache-dir (or absolute path)")
    p.add_argument("--k-cfhead-q", type=int, default=256)
    p.add_argument("--k-cfhead-m", type=int, default=8192)
    p.add_argument("--mob-head", default=f"{SCRATCH}/runs/p9_qwen35_swiglu/head_latest.pt")
    p.add_argument("--mob-k", type=int, default=128)
    p.add_argument("--mob-hidden", type=int, default=7300)
    p.add_argument("--mob-blocks", type=int, default=1)
    p.add_argument("--codebook", default=f"{SCRATCH}/data/codebook_qwen35.npz")
    p.add_argument("--data-dir", default=f"{SCRATCH}/bench_data")
    p.add_argument("--sharegpt-file", default="sharegpt.json")
    p.add_argument("--lmsys-file", default="lmsys.jsonl")
    p.add_argument("--longbench-hotpotqa-file", default="longbench_hotpotqa.jsonl")
    p.add_argument("--longbench-samsum-file", default="longbench_samsum.jsonl")
    p.add_argument("--samples-per-dataset", type=int, default=2,
                   help="ShareGPT/LMSYS each; LongBench gets max(1, n//2) per task "
                        "(protocol default 2 -> 2+2+1+1 = 6 prompts)")
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--out-dir", default=f"{SCRATCH}/runs/cfhead_longform")
    p.add_argument("--steps", type=int, default=8192)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--arms", default=",".join(ALL_ARMS),
                   help="comma list, subsettable: native,cfhead_q,cfhead_m,mob_cascade")
    p.add_argument("--repeat-window", type=int, default=32)
    p.add_argument("--checkpoint-every", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fix_seed(args.seed)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"[ERROR] unknown arms {unknown}; choose from {ALL_ARMS}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype = getattr(torch, args.backbone_dtype)
    print(f"device={device} dtype={dtype} torch={torch.__version__} "
          f"cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={torch.cuda.get_device_name(0)} "
              f"vram={props.total_memory / 2 ** 20:.0f} MiB", flush=True)
    print(f"arms={arms} steps={args.steps} seed={args.seed} EOS=disabled batch=1", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype).to(device)
    model.eval()
    model.requires_grad_(False)  # frozen backbone
    print(f"model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    # resident fp32 copy of the tied lm_head weight for the shadow ground truth
    # ([V,H] fp32 ~= 2.0 GB at V=248320,H=2048; the per-step [1,V] fp32 logits
    # are ~1 MB, no vocab chunking needed at batch 1)
    w_src = model.get_output_embeddings().weight
    w_fp32 = w_src.detach().float()
    V, H = w_fp32.shape
    print(f"native head: V={V} H={H} (fp32 shadow weight resident)", flush=True)

    # ---- lazily build only the runtimes the selected arms need ----
    rt = SimpleNamespace(w_fp32=w_fp32, cf_q=None, cf_m=None, mob=None,
                         codes_all=None, vid_to_tok=None, E=None)
    if "native" in arms or "cfhead_q" in arms:
        rt.cf_q = CFHeadRT(resolve_cache(args.cfhead_cache_dir, args.cfhead_q_cache),
                           w_src, device)
        print(f"cfhead_q cache: {rt.cf_q.meta} k={args.k_cfhead_q}", flush=True)
    if "cfhead_m" in arms:
        rt.cf_m = CFHeadRT(resolve_cache(args.cfhead_cache_dir, args.cfhead_m_cache),
                           w_src, device)
        print(f"cfhead_m cache: {rt.cf_m.meta} k={args.k_cfhead_m}", flush=True)
    if "mob_cascade" in arms:
        cb = np.load(need_file(args.codebook, "MoB codebook"))
        if cb["codes"].shape[0] != V:
            raise SystemExit(f"[ERROR] codebook V={cb['codes'].shape[0]} != model V={V}: "
                             f"{args.codebook}")
        rt.codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).to(device)
        cb_m = rt.codes_all.shape[1]
        rt.vid_to_tok = torch.full((1 << cb_m,), -1, dtype=torch.long, device=device)
        rt.vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().to(device)] = \
            torch.arange(len(cb["vertex_id"]), device=device)
        rt.mob = SwiGLUHead(k=args.mob_k, hidden=args.mob_hidden,
                            blocks=args.mob_blocks, m=cb_m).to(device)
        rt.mob.load_state_dict(torch.load(need_file(args.mob_head, "MoB head")))
        rt.mob.eval()
        rt.E = model.get_input_embeddings().weight
        n_params = sum(q.numel() for q in rt.mob.parameters())
        print(f"mob head: swiglu K={args.mob_k} hidden={args.mob_hidden} "
              f"blocks={args.mob_blocks} m={cb_m} params={n_params / 1e6:.1f}M", flush=True)

    # ---- prompts ----
    prompts = build_prompts(tokenizer, args)
    for s in prompts:
        print(f"  prompt p{s['prompt_idx']}: {s['dataset']}/{s['sample_id']} "
              f"({raw_token_len(tokenizer, s['text'])} raw tokens)", flush=True)

    # ---- arm-major loop: per-arm summary is written as soon as the arm ends ----
    all_summaries = {}
    for arm in arms:
        per_prompt = []
        for prompt in prompts:
            jsonl_path = out_dir / f"longform_{arm}_p{prompt['prompt_idx']}.jsonl"
            print(f"=== arm={arm} prompt {prompt['prompt_idx'] + 1}/{len(prompts)} "
                  f"{prompt['dataset']}/{prompt['sample_id']} ===", flush=True)
            summ = run_prompt(arm, prompt, model, tokenizer, rt, args, device, jsonl_path)
            per_prompt.append(summ)
            print(f"  -> hit={summ['hit_rate']:.4f} agree={summ['agree_rate']:.4f} "
                  f"mean_cand={summ['mean_n_cand']:.1f} "
                  f"({summ['sec_per_step'] * 1000:.1f} ms/step)", flush=True)
        agg = aggregate_arm(arm, per_prompt, args)
        arm_path = out_dir / f"longform_summary_{arm}.json"
        with open(arm_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        all_summaries[arm] = agg
        ov = agg["overall"]
        print(f"=== arm={arm} DONE: hit={ov['hit_rate']:.4f} agree={ov['agree_rate']:.4f} "
              f"repeat(hit/agree)={ov['buckets']['repeat']['hit_rate']}/"
              f"{ov['buckets']['repeat']['agree_rate']} "
              f"novel(hit/agree)={ov['buckets']['novel']['hit_rate']}/"
              f"{ov['buckets']['novel']['agree_rate']} -> {arm_path}", flush=True)

    combined = {
        "arms": all_summaries,
        "prompts_meta": [
            {"prompt_idx": s["prompt_idx"], "dataset": s["dataset"],
             "sample_id": s["sample_id"], "source": s["source"],
             "preview": s["text"][:160].replace("\n", " ")}
            for s in prompts
        ],
        "config": run_config(args),
    }
    all_path = out_dir / "longform_summary_all.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"wrote {all_path}", flush=True)
    print("CFHEAD_LONGFORM_DONE", flush=True)


if __name__ == "__main__":
    main()
