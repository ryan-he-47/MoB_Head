# Benchmark-task head-to-head: Original vs CFHead (two configs) vs MoB-cascade
# on Qwen3.5-2B — the "口径 A / benchmark" arm of docs/26-cfhead-headtohead-plan.md §3.
#
# Protocol ported from the CFHead reproduction (E:\CFHead\CFHead\repro):
#   * seed-99 shuffle -> fixed sample counts (accuracy_compare.py, mmlu_topk_sweep.py)
#   * 5-shot chat prompts via tokenizer.apply_chat_template (instruct model)
#   * greedy decoding with HF KV-cache incremental decoding, EOS early stop
#     (benchmarks, unlike the longform harness — EOS is allowed)
#   * metrics: accuracy (repro answer extraction), token/LCP agreement vs the
#     Original trajectory, containment@k = fraction of Original-trajectory steps
#     whose argmax token lies in the arm's candidate set (paper top-1 hit ratio),
#     mean candidates per step.
#
# Run from the repo root:  python -m eval.cfhead_benchmarks [options]
#
# Deviations from the repro (noted per task spec):
#   1. MGSM (en/zh): this parquet export's `answer`/`equation_solution` columns
#      are 100% NaN (250/250, both languages) — gold answers come from
#      `answer_number` (int32, intact). 5-shot demos come from the MGSM
#      train_en export (`--mgsm-demo-file`, proper worked solutions ending in
#      "The answer is N") for BOTH languages = the MGSM paper's EN-CoT
#      condition; labeled in the report. (2026-09-19 fix: the original run
#      extracted gold='nan' and demoed 'nan' assistant replies — caught by the
#      all-False correct column; containment/agreement rows from that run are
#      void and MGSM was rerun.)
#   2. GSM8K/MGSM max_new = 256 (task spec; the repro used 512 for gsm8k).
#   3. EOS set = tokenizer.eos_token_id U generation_config.eos_token_id (Qwen
#      instruct emits <|im_end|>); finished steps are filled with
#      tokenizer.eos_token_id exactly like the repro's masked_fill.
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval.cfhead_rt import CFHeadRT
from mob.mob_common import SwiGLUHead, decode_rerank_native, propose_roundflip

SEED = 99
NUM_RE = r"[-+]?\d[\d,]*(?:\.\d+)?"
INT_RE = r"[-+]?\d[\d,]*"

# task -> (loader tag, sample count, max_new_tokens, answer_type)
TASKS = {
    "mgsm_en": ("mgsm", 250, 256, "mgsm"),
    "mgsm_zh": ("mgsm", 250, 256, "mgsm"),
    "mmlu": ("jsonl", 100, 4, "choice"),
    "gsm8k": ("jsonl", 128, 256, "gsm8k"),
    "arc_challenge": ("jsonl", 128, 4, "choice"),
}
TASK_FILES = {
    "mgsm_en": "mgsm_en.parquet",
    "mgsm_zh": "mgsm_zh.parquet",
    "mmlu": "mmlu.jsonl",
    "gsm8k": "gsm8k.jsonl",
    "arc_challenge": "arc_challenge.jsonl",
}
# Repro-style per-task instruction prefix (same shape as the GSM8K 5-shot demos
# in the repro repo's script/eval_common.py::build_gsm8k). The zh instruction is
# a straight translation — the official MGSM 8-shot template is unavailable.
MGSM_INSTRUCTION = {
    "en": "Solve the problem. Give the final answer after 'The answer is'.",
    "zh": "解答下面的数学题，并把最终答案写在 'The answer is' 后面。",
}
ARMS = ["original", "cfhead_q", "cfhead_m", "mob_cascade"]
CFHEAD_K = {"cfhead_q": 256, "cfhead_m": 8192}          # runtime top-k per arm
CFHEAD_CLUSTERS = {"cfhead_q": 64, "cfhead_m": 4}       # C in the cache filename
SUMMARY_FIELDS = ["task", "arm", "accuracy", "token_agr", "containment", "mean_candidates", "n"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# data loading / answer extraction (ported from script/eval_common.py)
# ---------------------------------------------------------------------------
def load_jsonl_seed99(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:n]


def load_mgsm(path: Path, lang: str, demo_path: Path) -> list[dict]:
    """Full test split (250 rows expected) in file order. This parquet export's
    `answer` column is 100% NaN -> gold = `answer_number`; 5-shot demos from the
    MGSM train_en export (worked solutions, 'The answer is N' style)."""
    import pandas as pd

    df = pd.read_parquet(path)
    ddf = pd.read_parquet(demo_path)
    demos = [(str(ddf.iloc[i]["question"]), str(ddf.iloc[i]["answer"]))
             for i in range(min(5, len(ddf)))]
    instr = MGSM_INSTRUCTION[lang]
    rows = []
    for i in range(len(df)):
        q = str(df.iloc[i]["question"])
        gold = str(int(df.iloc[i]["answer_number"]))
        messages = []
        for dq, da in demos:
            messages.append({"role": "user", "content": f"{instr}\n\n{dq}"})
            messages.append({"role": "assistant", "content": da})
        messages.append({"role": "user", "content": f"{instr}\n\n{q}"})
        rows.append({
            "task": f"mgsm_{lang}",
            "sample_id": i,
            "messages": messages,
            "target": gold,
            "answer_type": "mgsm",
        })
    if len(rows) != 250:
        log(f"WARNING: {path.name} has {len(rows)} rows (MGSM full test split is expected to be 250)")
    return rows


def extract_answer(text: str, answer_type: str) -> str:
    """Repro extraction, verbatim (choice / gsm8k / text)."""
    t = text.strip()
    if answer_type == "choice":
        m = re.search(r"\b([A-E])\b", t.upper())
        return m.group(1) if m else t[:1].upper()
    if answer_type == "gsm8k":
        m = re.search(r"####\s*(%s)" % NUM_RE, t)
        if m:
            return m.group(1).replace(",", "")
        nums = re.findall(NUM_RE, t)
        return nums[-1].replace(",", "") if nums else t
    return t


def extract_gold_mgsm(target: str) -> str:
    m = re.search(r"####\s*(%s)" % NUM_RE, target.strip())
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(INT_RE, target)
    return nums[-1].replace(",", "") if nums else target.strip()


def extract_pred_mgsm(output: str) -> str:
    nums = re.findall(INT_RE, output.strip())
    return nums[-1].replace(",", "") if nums else output.strip()


def score_row(output: str, target: str, answer_type: str) -> tuple[str, str, bool]:
    if answer_type == "mgsm":
        pred, gold = extract_pred_mgsm(output), extract_gold_mgsm(target)
        return pred, gold, bool(gold) and pred == gold
    pred = extract_answer(output, answer_type)
    gold = extract_answer(target, answer_type)
    return pred, gold, bool(gold) and pred == gold


def format_prompt(tokenizer, row: dict) -> str:
    """Repro format_prompt: the full 5-shot chat template path (messages field)."""
    messages = row.get("messages")
    if messages:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = row.get("prompt", row.get("query", ""))
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# generation (ported from repro accuracy_compare.py::generate_original_collect
# / script/chat.py::generate_with_selector, generalized to an EOS set)
# ---------------------------------------------------------------------------
class _BackboneOut:
    """Duck-typed backbone output for the no-`.model` fallback path."""

    def __init__(self, past_key_values, last_hidden_state):
        self.past_key_values = past_key_values
        self.last_hidden_state = last_hidden_state


def forward_backbone(model, **model_inputs):
    """Run only the transformer body, not the CausalLM lm_head (repro helper)."""
    backbone = getattr(model, "model", None)
    if backbone is None:
        model_inputs["output_hidden_states"] = True
        outputs = model(**model_inputs)
        return _BackboneOut(outputs.past_key_values, outputs.hidden_states[-1])
    return backbone(**model_inputs)


@torch.no_grad()
def greedy_generate(model, input_ids, attention_mask, max_new_tokens: int,
                    eos_tensor: torch.Tensor, fill_id: int, select_fn,
                    collect_hidden: bool = False):
    """Greedy decode, batch 1. select_fn: h [1,H] -> token ids [1] long.
    Returns (generated ids [1,S], per-step hidden states [S,H] on device)."""
    past = None
    generated = []
    hidden = []
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    cur_ids = input_ids
    for _ in range(max_new_tokens):
        outputs = forward_backbone(
            model,
            input_ids=cur_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        h = outputs.last_hidden_state[:, -1, :]
        if collect_hidden:
            hidden.append(h[0].detach())
        next_ids = select_fn(h).masked_fill(finished, fill_id)
        generated.append(next_ids)
        cur_ids = next_ids.unsqueeze(-1)
        finished = finished | torch.isin(next_ids, eos_tensor)
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones((attention_mask.shape[0], 1), device=attention_mask.device,
                        dtype=attention_mask.dtype)],
            dim=1,
        )
        if torch.all(finished):
            break
    h_stack = torch.stack(hidden, dim=0) if hidden else None
    return torch.stack(generated, dim=1), h_stack


def agreement_metrics(method_ids: list[int], orig_ids: list[int]) -> tuple[float, float]:
    """Repro agreement: position-wise match over aligned region; LCP / method len."""
    n = min(len(method_ids), len(orig_ids))
    if n == 0:
        return 0.0, 0.0
    matches = sum(1 for i in range(n) if method_ids[i] == orig_ids[i])
    lcp = 0
    for a, b in zip(method_ids, orig_ids):
        if a == b:
            lcp += 1
        else:
            break
    return matches / n, lcp / max(len(method_ids), 1)


# ---------------------------------------------------------------------------
# containment on the Original trajectory (paper top-1 hit ratio)
# ---------------------------------------------------------------------------
@torch.no_grad()
def cfhead_containment(rt: CFHeadRT, k: int, h_stack: torch.Tensor,
                       orig_ids: torch.Tensor, chunk: int = 32) -> tuple[int, int, float]:
    hits, steps = 0, h_stack.shape[0]
    for s in range(0, steps, chunk):
        cand = rt.candidates(h_stack[s:s + chunk], k)               # [c,k]
        hits += int((cand == orig_ids[s:s + chunk].unsqueeze(1)).any(1).sum().item())
    return hits, steps, float(k)


@torch.no_grad()
def mob_containment(head, codes_all, vid_to_tok, h_stack: torch.Tensor,
                    orig_ids: torch.Tensor) -> tuple[int, int, float]:
    # NOTE (2026-09-19, post-hoc): the head forward is batched over all steps
    # (row-independent ops -> numerically identical to per-step calls, ~3x less
    # launch overhead). propose_roundflip stays per-step (n_batch==1 API) and
    # remains the cost pig of the original pass (~10-15s per MGSM sample).
    hits, steps, sizes = 0, h_stack.shape[0], 0
    bl_all, pl_all = head(h_stack.float())
    log_pi_all = F.log_softmax(pl_all, dim=-1)
    for t in range(steps):
        cand = propose_roundflip(bl_all[t:t + 1], log_pi_all[t:t + 1], codes_all, vid_to_tok)
        hits += int((cand == orig_ids[t]).any().item())
        sizes += cand.numel()
    return hits, steps, sizes / max(steps, 1)


# ---------------------------------------------------------------------------
# arm construction
# ---------------------------------------------------------------------------
def resolve_cfhead_cache(cache_dir: Path, clusters: int, required: bool) -> Path | None:
    hits = sorted(cache_dir.glob(f"cfhead_v3_*_T8_C{clusters}_*.pt"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous CFHead cache for C={clusters} in {cache_dir}: {hits}")
    if required:
        raise FileNotFoundError(f"no CFHead cache for C={clusters} in {cache_dir}")
    return None


def build_cfhead_rt(cache_path: Path, weight: torch.Tensor, device: torch.device,
                    clusters: int) -> CFHeadRT:
    rt = CFHeadRT(cache_path, weight, device)
    meta = rt.meta
    if meta.get("tile_size") != 8 or meta.get("cluster_size") != clusters:
        raise ValueError(f"cache {cache_path.name} meta mismatch: {meta}")
    if meta.get("vocab_size") != weight.shape[0]:
        raise ValueError(f"cache {cache_path.name} vocab {meta.get('vocab_size')} != model V {weight.shape[0]}")
    return rt


def build_mob(args, model, device: torch.device):
    cb = np.load(args.codebook)
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).to(device)   # [V,CB_M]
    cb_m = codes_all.shape[1]
    vid_to_tok = torch.full((1 << cb_m,), -1, dtype=torch.long, device=device)
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().to(device)] = \
        torch.arange(len(cb["vertex_id"]), device=device)
    head = SwiGLUHead(k=args.mob_k, h=args.mob_h, hidden=args.mob_hidden,
                      m=cb_m, blocks=args.mob_blocks).to(device)
    head.load_state_dict(torch.load(args.mob_head, map_location="cpu"))
    head.eval()
    return head, codes_all, vid_to_tok


# ---------------------------------------------------------------------------
# task runner
# ---------------------------------------------------------------------------
def run_task(task: str, args, model, tokenizer, device: torch.device,
             eos_tensor: torch.Tensor, fill_id: int, cfhead_rts: dict,
             mob_ctx, e_weight: torch.Tensor, diag_q_rt, out_dir: Path) -> list[dict]:
    loader, n_samples, max_new, answer_type = TASKS[task]
    if loader == "mgsm":
        samples = load_mgsm(Path(args.bench_root) / TASK_FILES[task], task.split("_")[1],
                            Path(args.bench_root) / args.mgsm_demo_file)
    else:
        samples = load_jsonl_seed99(Path(args.bench_root) / TASK_FILES[task], n_samples)
    if args.limit is not None:
        samples = samples[: args.limit]
    log(f"=== {task}: {len(samples)} samples, max_new={max_new}, answer_type={answer_type} ===")

    lm_head = model.get_output_embeddings()

    def select_original(h):
        return lm_head(h).argmax(-1)

    # ---- Original pass (always run: it is the agreement/containment reference) ----
    t0 = time.perf_counter()
    orig_rows = []
    orig_info = []   # per-sample reference data for the arm passes
    orig_path = out_dir / f"{task}_original.jsonl"
    orig_fh = orig_path.open("w", encoding="utf-8") if "original" in args.arms_set else None
    try:
        for idx, row in enumerate(samples):
            prompt = format_prompt(tokenizer, row)
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            gen_ids, h_stack = greedy_generate(model, input_ids, attn, max_new,
                                               eos_tensor, fill_id, select_original,
                                               collect_hidden=True)
            ids = gen_ids[0].detach().cpu().tolist()
            output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            pred, gold, correct = score_row(output, str(row.get("target", "")), answer_type)

            # containment of every requested arm's candidate set on THIS trajectory
            orig_tok = gen_ids[0].detach()
            cont = {}
            for arm, (rt, k) in cfhead_rts.items():
                cont[arm] = cfhead_containment(rt, k, h_stack, orig_tok)
            if mob_ctx is not None:
                head, codes_all, vid_to_tok = mob_ctx
                cont["mob_cascade"] = mob_containment(head, codes_all, vid_to_tok, h_stack, orig_tok)
            if diag_q_rt is not None and "cfhead_q" not in cont:
                rt, k = diag_q_rt
                cont["cfhead_q"] = cfhead_containment(rt, k, h_stack, orig_tok)

            rec = {
                "task": task,
                "arm": "original",
                "sample_id": row.get("sample_id"),
                "prediction": pred,
                "gold": gold,
                "correct": correct,
                "output": output,
                "n_tokens": int(gen_ids.shape[1]),
                "token_agreement": 1.0,
                "lcp_agreement": 1.0,
                # diagnostic: containment of the original argmax in the
                # CFHead-Q (C=64, k=256) coarse candidate set
                "containment": (cont["cfhead_q"][0] / cont["cfhead_q"][1]) if "cfhead_q" in cont else None,
                "containment_scope": "cfhead_q" if "cfhead_q" in cont else None,
                "containment_hits": cont["cfhead_q"][0] if "cfhead_q" in cont else 0,
                "containment_steps": cont["cfhead_q"][1] if "cfhead_q" in cont else 0,
                "mean_candidates": cont["cfhead_q"][2] if "cfhead_q" in cont else None,
                "containment_cfhead_m": (cont["cfhead_m"][0] / cont["cfhead_m"][1]) if "cfhead_m" in cont else None,
                "containment_mob_cascade": (cont["mob_cascade"][0] / cont["mob_cascade"][1]) if "mob_cascade" in cont else None,
                "mean_candidates_mob": cont["mob_cascade"][2] if "mob_cascade" in cont else None,
            }
            if "in_fewshot_demos" in row:
                rec["in_fewshot_demos"] = row["in_fewshot_demos"]
            orig_rows.append(rec)
            if orig_fh is not None:
                orig_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            orig_info.append({"ids": ids, "output": output, "cont": cont})
            if (idx + 1) % 16 == 0 or idx == len(samples) - 1:
                log(f"  [{task}/original] {idx + 1}/{len(samples)} ({time.perf_counter() - t0:.1f}s)")
    finally:
        if orig_fh is not None:
            orig_fh.close()
    log(f"[{task}] original pass done ({time.perf_counter() - t0:.1f}s)")

    summary = []
    if "original" in args.arms_set:
        summary.append(summarize(task, "original", orig_rows))

    # ---- arm passes ----
    def select_cfhead(rt, k):
        def fn(h):
            return rt.select(h, k)[0]
        return fn

    arm_select = {}
    for arm, (rt, k) in cfhead_rts.items():
        arm_select[arm] = select_cfhead(rt, k)
    if mob_ctx is not None:
        mob_head, mob_codes, mob_vid = mob_ctx

        def select_mob(h, _head=mob_head, _codes=mob_codes, _vid=mob_vid, _e=e_weight):
            hf = h.float()
            bl, pl = _head(hf)
            log_pi = F.log_softmax(pl, dim=-1)
            return decode_rerank_native(bl, log_pi, _codes, _e, hf,
                                        proposal="roundflip", vid_to_tok=_vid)

        arm_select["mob_cascade"] = select_mob

    for arm in ARMS:
        if arm == "original" or arm not in args.arms_set or arm not in arm_select:
            continue
        t0 = time.perf_counter()
        rows = []
        with (out_dir / f"{task}_{arm}.jsonl").open("w", encoding="utf-8") as fh:
            for idx, row in enumerate(samples):
                prompt = format_prompt(tokenizer, row)
                enc = tokenizer(prompt, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attn = enc["attention_mask"].to(device)
                gen_ids, _ = greedy_generate(model, input_ids, attn, max_new,
                                             eos_tensor, fill_id, arm_select[arm])
                ids = gen_ids[0].detach().cpu().tolist()
                output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                pred, gold, correct = score_row(output, str(row.get("target", "")), answer_type)
                o = orig_info[idx]
                ta, lcp = agreement_metrics(ids, o["ids"])
                ch, cs, mc = o["cont"][arm]
                rec = {
                    "task": task,
                    "arm": arm,
                    "sample_id": row.get("sample_id"),
                    "prediction": pred,
                    "gold": gold,
                    "correct": correct,
                    "output": output,
                    "n_tokens": int(gen_ids.shape[1]),
                    "token_agreement": ta,
                    "lcp_agreement": lcp,
                    "containment": ch / max(cs, 1),
                    "containment_scope": "original_trajectory",
                    "containment_hits": ch,
                    "containment_steps": cs,
                    "mean_candidates": mc,
                    "original_output": o["output"],
                    "original_n_tokens": len(o["ids"]),
                }
                if "in_fewshot_demos" in row:
                    rec["in_fewshot_demos"] = row["in_fewshot_demos"]
                rows.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (idx + 1) % 16 == 0 or idx == len(samples) - 1:
                    log(f"  [{task}/{arm}] {idx + 1}/{len(samples)} ({time.perf_counter() - t0:.1f}s)")
        log(f"[{task}] {arm} done ({time.perf_counter() - t0:.1f}s)")
        summary.append(summarize(task, arm, rows))

    return summary


def summarize(task: str, arm: str, rows: list[dict]) -> dict:
    n = len(rows)
    acc = 100.0 * sum(1 for r in rows if r["correct"]) / n
    ta = 100.0 * sum(r["token_agreement"] for r in rows) / n
    hits = sum(int(r.get("containment_hits") or 0) for r in rows)
    steps = sum(int(r.get("containment_steps") or 0) for r in rows)
    cont = 100.0 * hits / steps if steps > 0 else None
    mc_vals = [(float(r["mean_candidates"]), int(r["containment_steps"]))
               for r in rows if r.get("mean_candidates") is not None and r.get("containment_steps")]
    mean_cand = (sum(mc * st for mc, st in mc_vals) / sum(st for _, st in mc_vals)) if mc_vals else None
    return {
        "task": task,
        "arm": arm,
        "accuracy": round(acc, 4),
        "token_agr": round(ta, 4),
        "containment": round(cont, 4) if cont is not None else "",
        "mean_candidates": round(mean_cand, 2) if mean_cand is not None else "",
        "n": n,
    }


def write_summary(out_dir: Path, rows: list[dict]) -> None:
    order = {t: i for i, t in enumerate(TASKS)}
    aorder = {a: i for i, a in enumerate(ARMS)}
    rows = sorted(rows, key=lambda r: (order.get(r["task"], 9), aorder.get(r["arm"], 9)))
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Original vs CFHead vs MoB-cascade benchmark comparison (Qwen3.5-2B)")
    p.add_argument("--model-dir", default="/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B")
    p.add_argument("--bench-root", default="/gpfs1/scratch/yihe47/qwen35/bench_data")
    p.add_argument("--mgsm-demo-file", default="mgsm_train_en.parquet")
    p.add_argument("--cfhead-cache-dir", default="/gpfs1/scratch/yihe47/qwen35/cfhead_cache")
    p.add_argument("--mob-head", default="/gpfs1/scratch/yihe47/qwen35/runs/p9_qwen35_swiglu/head_latest.pt")
    p.add_argument("--codebook", default="/gpfs1/scratch/yihe47/qwen35/data/codebook_qwen35.npz")
    p.add_argument("--out", default="/gpfs1/scratch/yihe47/qwen35/runs/cfhead_benchmarks")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--arms", default=",".join(ARMS),
                   help="Comma subset of original,cfhead_q,cfhead_m,mob_cascade. "
                        "The original pass always runs (it is the agreement/containment "
                        "reference); it is only recorded when listed.")
    p.add_argument("--cfhead-q-k", type=int, default=CFHEAD_K["cfhead_q"])
    p.add_argument("--cfhead-m-k", type=int, default=CFHEAD_K["cfhead_m"])
    p.add_argument("--mob-k", type=int, default=128)
    p.add_argument("--mob-h", type=int, default=2048)
    p.add_argument("--mob-hidden", type=int, default=7300)
    p.add_argument("--mob-blocks", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None, help="debug: cap samples per task")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.arms_set = {a.strip() for a in args.arms.split(",") if a.strip()}
    unknown = args.arms_set - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {unknown} (choose from {ARMS})")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype).to(device)
    model.eval()
    log(f"model loaded from {args.model_dir} in {time.perf_counter() - t0:.1f}s dtype={dtype}")

    weight = model.get_output_embeddings().weight
    V, H = weight.shape
    e_weight = model.get_input_embeddings().weight
    log(f"vocab={V} hidden={H} tied={model.get_input_embeddings().weight.data_ptr() == weight.data_ptr()}")

    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    gen_cfg = getattr(model, "generation_config", None)
    cfg_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if isinstance(cfg_eos, int):
        eos_ids.add(cfg_eos)
    elif cfg_eos:
        eos_ids.update(int(x) for x in cfg_eos)
    if not eos_ids:
        raise ValueError("no EOS token id found (tokenizer + generation_config both empty)")
    fill_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else sorted(eos_ids)[0]
    eos_tensor = torch.tensor(sorted(eos_ids), device=device)
    log(f"eos_ids={sorted(eos_ids)} fill_id={fill_id}")

    # ---- arm runtimes ----
    cache_dir = Path(args.cfhead_cache_dir)
    cfhead_rts: dict[str, tuple[CFHeadRT, int]] = {}
    diag_q_rt = None
    if "cfhead_q" in args.arms_set:
        q_path = resolve_cfhead_cache(cache_dir, CFHEAD_CLUSTERS["cfhead_q"], True)
        if q_path is None:
            raise FileNotFoundError(f"no CFHead cache for C=64 in {cache_dir}")
        cfhead_rts["cfhead_q"] = (
            build_cfhead_rt(q_path, weight, device, CFHEAD_CLUSTERS["cfhead_q"]),
            args.cfhead_q_k,
        )
        log(f"cfhead_q ready (k={args.cfhead_q_k}, cache={q_path.name})")
    if "cfhead_m" in args.arms_set:
        m_path = resolve_cfhead_cache(cache_dir, CFHEAD_CLUSTERS["cfhead_m"], True)
        if m_path is None:
            raise FileNotFoundError(f"no CFHead cache for C=4 in {cache_dir}")
        cfhead_rts["cfhead_m"] = (
            build_cfhead_rt(m_path, weight, device, CFHEAD_CLUSTERS["cfhead_m"]),
            args.cfhead_m_k,
        )
        log(f"cfhead_m ready (k={args.cfhead_m_k}, cache={m_path.name})")
    if "original" in args.arms_set and "cfhead_q" not in cfhead_rts:
        # diagnostic containment of the original trajectory against CFHead-Q
        q_path = resolve_cfhead_cache(cache_dir, CFHEAD_CLUSTERS["cfhead_q"], False)
        if q_path is not None:
            diag_q_rt = (build_cfhead_rt(q_path, weight, device, CFHEAD_CLUSTERS["cfhead_q"]),
                         args.cfhead_q_k)
            log("cfhead_q cache loaded for the original-arm containment diagnostic")
        else:
            log("cfhead_q cache absent -> original-arm containment diagnostic skipped")

    mob_ctx = None
    if "mob_cascade" in args.arms_set:
        mob_ctx = build_mob(args, model, device)
        log(f"mob_cascade ready (SwiGLU k={args.mob_k} hidden={args.mob_hidden} "
            f"blocks={args.mob_blocks} m={mob_ctx[1].shape[1]}, roundflip+rerank)")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown_t = set(tasks) - set(TASKS)
    if unknown_t:
        raise ValueError(f"unknown tasks: {unknown_t} (choose from {list(TASKS)})")

    summary_rows: list[dict] = []
    for task in tasks:
        summary_rows.extend(run_task(task, args, model, tokenizer, device,
                                     eos_tensor, fill_id, cfhead_rts, mob_ctx,
                                     e_weight, diag_q_rt, out_dir))
        write_summary(out_dir, summary_rows)
        log(f"summary updated -> {out_dir / 'summary.csv'}")

    log("FINAL SUMMARY (accuracy / token_agr / containment in %)")
    for r in summary_rows:
        log(f"  {r['task']:14s} {r['arm']:12s} acc={r['accuracy']:>8} tok_agr={r['token_agr']:>8} "
            f"cont={r['containment']:>8} cand={r['mean_candidates']:>8} n={r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
