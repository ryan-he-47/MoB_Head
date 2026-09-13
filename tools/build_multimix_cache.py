# Build the Qwen3.5 multilingual+code+instruct interleaved token cache.
# Mix (user decision 2026-09-14): EN 50% / ZH 30% / Code 15% / Instruct 5%,
# ~2B tokens total, document-level interleave (seeded), per-source dev caches
# held out BEFORE mixing. Emits into the scratch dir:
#   qwen35_mix_train_tokens.npy  (int32, ~2B)
#   qwen35_mix_segments.npy      (int64 [n_seg,3]: start,end,source_id) - audit
#   qwen35_dev_{en,zh,code,instruct}_tokens.npy (int32, ~2M tokens each)
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SOURCES = ["en", "zh", "code", "instruct"]
MIX = {"en": 0.50, "zh": 0.30, "code": 0.15, "instruct": 0.05}
TOTAL_BUDGET = 2_000_000_000
DEV_TOKENS = 2_000_000
CODE_LANGS = {"Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#",
              "Go", "Rust", "Shell"}


def texts_en(root):
    for p in sorted(glob.glob(os.path.join(root, "fineweb_edu_100bt", "**", "*.parquet"),
                              recursive=True)):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=2000, columns=["text"]):
            for t in batch.column("text").to_pylist():
                yield t


def texts_zh(root):
    files = sorted(glob.glob(os.path.join(root, "cci3hq", "**", "*.jsonl"), recursive=True))
    if files:
        import json as js
        for p in files:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = js.loads(line)
                    except Exception:
                        continue
                    t = d.get("text") or d.get("content")
                    if t:
                        yield t
        return
    for p in sorted(glob.glob(os.path.join(root, "fineweb2_zh", "**", "*.parquet"),
                              recursive=True)):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=2000, columns=["text"]):
            for t in batch.column("text").to_pylist():
                yield t
    import gzip
    for p in sorted(glob.glob(os.path.join(root, "telechat_ptd", "**", "*.jsonl.gz"),
                              recursive=True)):
        import json as js
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = js.loads(line)
                except Exception:
                    continue
                t = d.get("text") or d.get("content")
                if t:
                    yield t


def texts_code(root):
    for p in sorted(glob.glob(os.path.join(root, "stack_v3", "**", "*.parquet"),
                              recursive=True)):
        pf = pq.ParquetFile(p)
        cols = [c for c in ("content", "license_type", "language") if c in pf.schema.names]
        for batch in pf.iter_batches(batch_size=2000, columns=cols):
            d = batch.to_pydict()
            for i in range(len(d["content"])):
                if "license_type" in d and d["license_type"][i] not in (None, "permissive"):
                    continue
                if "language" in d and d["language"][i] not in CODE_LANGS:
                    continue
                yield d["content"][i]


def texts_instruct(root, tok):
    for p in sorted(glob.glob(os.path.join(root, "smoltalk", "**", "*.parquet"),
                              recursive=True)):
        pf = pq.ParquetFile(p)
        col = "messages" if "messages" in pf.schema.names else pf.schema.names[0]
        for batch in pf.iter_batches(batch_size=500, columns=[col]):
            for msgs in batch.column(col).to_pylist():
                try:
                    msgs = [dict(m) for m in msgs]
                    yield tok.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=False)
                except Exception:
                    continue


def tokenize_source(name, gen, tok, budget, eos, dev_tokens, log_every=20000):
    """Tokenize a document stream; returns (train_chunks list, dev array, n_tok)."""
    train, dev, n_dev, n_tok, n_doc = [], [], 0, 0, 0
    batch, bs = [], 512
    def flush(b):
        nonlocal n_dev, n_tok, n_doc
        if not b:
            return
        enc = tok(b, add_special_tokens=False)["input_ids"]
        for ids in enc:
            ids = ids + [eos]
            n_doc += 1
            if n_dev < dev_tokens:
                dev.extend(ids)
                n_dev += len(ids)
            elif n_tok < budget:
                train.append(np.asarray(ids, dtype=np.int32))
                n_tok += len(ids)
            if n_doc % log_every == 0:
                print(f"  [{name}] {n_doc} docs, train {n_tok/1e6:.0f}M tok", flush=True)
    for t in gen:
        if n_tok >= budget:
            break
        if not t or len(t) < 32:
            continue
        batch.append(t)
        if len(batch) >= bs:
            flush(batch)
            batch = []
    flush(batch)
    print(f"[{name}] DONE: {n_doc} docs, train {n_tok/1e6:.1f}M tok, dev {n_dev/1e6:.1f}M", flush=True)
    return train, np.asarray(dev, dtype=np.int32), n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/gpfs1/scratch/yihe47/qwen35")
    ap.add_argument("--model-dir", default="/gpfs1/scratch/yihe47/qwen35/models/Qwen3.5-2B")
    ap.add_argument("--total", type=int, default=TOTAL_BUDGET)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    data = os.path.join(args.root, "data")
    out = os.path.join(args.root, "cache")
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    eos = tok.eos_token_id or tok.pad_token_id

    gens = {
        "en": texts_en(data),
        "zh": texts_zh(data),
        "code": texts_code(data),
        "instruct": texts_instruct(data, tok),
    }
    per = {}
    devs = {}
    for name in SOURCES:
        budget = int(args.total * MIX[name])
        chunks, dev, n_tok = tokenize_source(name, gens[name], tok, budget, eos, DEV_TOKENS)
        per[name] = chunks
        devs[name] = dev
        np.save(os.path.join(out, f"qwen35_dev_{name}_tokens.npy"), dev)

    # document-level interleave by mix proportion (seeded round-robin)
    rng = np.random.default_rng(args.seed)
    order = []
    for name, w in MIX.items():
        order += [name] * int(w * 100)
    pos = {name: 0 for name in SOURCES}
    seg_records = []
    parts = []
    written = 0
    next_log = 200_000_000
    while True:
        rng.shuffle(order)
        progressed = False
        for name in order:
            if pos[name] < len(per[name]):
                ids = per[name][pos[name]]
                pos[name] += 1
                parts.append(ids)
                seg_records.append((written, written + len(ids), SOURCES.index(name)))
                written += len(ids)
                progressed = True
                if written >= next_log:
                    print(f"mixed {written/1e6:.0f}M tok", flush=True)
                    next_log += 200_000_000
        if not progressed:
            break
    final = np.concatenate(parts) if parts else np.asarray([], dtype=np.int32)
    np.save(os.path.join(out, "qwen35_mix_train_tokens.npy"), final)
    np.save(os.path.join(out, "qwen35_mix_segments.npy"),
            np.asarray(seg_records, dtype=np.int64))
    meta = {"mix": MIX, "tokens": int(written), "sources": SOURCES,
            "dev_tokens_per_source": DEV_TOKENS, "seed": args.seed}
    with open(os.path.join(out, "qwen35_mix_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"MIX_DONE: {written/1e6:.1f}M tokens, {len(seg_records)} segments", flush=True)


if __name__ == "__main__":
    main()
