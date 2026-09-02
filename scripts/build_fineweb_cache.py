# Build FineWeb-Edu token caches on the cluster (CPU-only; no model, no CUDA).
# Streams HuggingFaceFW/fineweb-edu, tokenizes in batches, packs with EOS.
# Produces: <out-dir>/fineweb_dev_tokens.npy (first --dev-docs docs) and
#           <out-dir>/fineweb_train_tokens.npy (docs after the dev slice).
# Set HF_HUB_OFFLINE=1 if the dataset was pre-downloaded to HF cache.
import argparse
import os
import sys

import numpy as np
from transformers import AutoTokenizer


def stream_pack(tok, eos_id, dataset, config, split, field, skip, max_docs, out_path):
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split, streaming=True)
    parts, buf = [], []
    seen = kept = 0
    n_tok = 0
    for doc in ds:
        seen += 1
        if seen <= skip:
            continue
        text = (doc[field] or "").strip()
        if text:
            buf.append(text)
        kept += 1
        if len(buf) >= 1000:
            for ids in tok(buf).input_ids:
                ids = ids + [eos_id]
                parts.append(np.asarray(ids, dtype=np.int32))
                n_tok += len(ids)
            print(f"[{out_path}] docs kept={kept}, tokens so far={n_tok:,}", flush=True)
            buf = []
        if max_docs and kept >= max_docs:
            break
    if buf:
        for ids in tok(buf).input_ids:
            ids = ids + [eos_id]
            parts.append(np.asarray(ids, dtype=np.int32))
            n_tok += len(ids)
    arr = np.concatenate(parts) if parts else np.asarray([], dtype=np.int32)
    np.save(out_path, arr)
    print(f"saved {out_path}: {len(arr):,} tokens from {kept} docs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="tokenizer location")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-docs", type=int, default=1000000)
    ap.add_argument("--dev-docs", type=int, default=2000)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--field", default="text")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    eos = tok.eos_token_id

    stream_pack(tok, eos, args.dataset, args.config, args.split, args.field,
                skip=0, max_docs=args.dev_docs,
                out_path=os.path.join(args.out_dir, "fineweb_dev_tokens.npy"))
    stream_pack(tok, eos, args.dataset, args.config, args.split, args.field,
                skip=args.dev_docs, max_docs=args.train_docs,
                out_path=os.path.join(args.out_dir, "fineweb_train_tokens.npy"))


if __name__ == "__main__":
    main()
