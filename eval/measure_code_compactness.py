# Operative validation of the codebook: are the teacher's top-40 tokens at a
# position mapped to compact code regions? Compares tree codebook vs (a) random
# codebook control and (b) random 40-token sets.
import glob

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r".\LLM-Research--Llama-3.2-1B\snapshots\master"
DATA_DIR = r".\data\babylm_data\babylm_10M"
CODEBOOK = r".\data\codebook.npz"
LINES_PER_FILE = 1000
SEQ_LEN = 512
N_CHUNKS = 24
BATCH = 4
TOPK = 40
PREFIX_DEPTH = 8  # count distinct prefixes at this depth as subtree proxy


def pw_hamming_weighted(codes, w):
    # codes: [k, M] uint8, w: [k] prob -> weighted mean pairwise hamming
    d = (codes[:, None, :] != codes[None, :, :]).sum(-1).astype(np.float32)
    ww = w[:, None] * w[None, :]
    return (d * ww).sum() / ww.sum()


def main():
    cb = np.load(CODEBOOK)
    codes_np = cb["codes"].astype(np.uint8)  # [V, 17]
    rng = np.random.default_rng(42)
    codes_rnd = codes_np[rng.permutation(len(codes_np))]

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16).cuda()
    model.eval()

    all_ids = []
    for path in sorted(glob.glob(DATA_DIR + r"\*.train")):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= LINES_PER_FILE:
                    break
                ids = tok(line.strip()).input_ids
                all_ids.extend(ids + [tok.eos_token_id])
    chunks = [
        all_ids[i : i + SEQ_LEN]
        for i in range(0, min(N_CHUNKS * SEQ_LEN, len(all_ids) - SEQ_LEN), SEQ_LEN)
    ]

    res = {"tree": [], "rnd_codes": [], "rnd_tokens": [], "prefix8": []}
    with torch.no_grad():
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            ids = torch.tensor(batch, dtype=torch.long).cuda()
            logits = model(ids).logits[:, :-1].reshape(-1, 128256)
            top = logits.topk(TOPK, dim=-1)
            top_ids = top.indices.cpu().numpy()
            top_w = torch.softmax(top.values.float(), dim=-1).cpu().numpy()
            for row_ids, row_w in zip(top_ids, top_w):
                c_tree = codes_np[row_ids]
                res["tree"].append(pw_hamming_weighted(c_tree, row_w))
                res["rnd_codes"].append(pw_hamming_weighted(codes_rnd[row_ids], row_w))
                rnd_tok = rng.integers(0, len(codes_np), TOPK)
                w_unif = np.full(TOPK, 1 / TOPK)
                res["rnd_tokens"].append(pw_hamming_weighted(codes_np[rnd_tok], w_unif))
                pref = c_tree[:, :PREFIX_DEPTH]
                res["prefix8"].append(len(np.unique(pref, axis=0)))
    for k_, v in res.items():
        print(f"{k_}: mean={np.mean(v):.3f}")
    print("(hamming means are weighted-mean-pairwise within top-40; "
          "prefix8 = distinct 8-bit prefixes per position, of max 40)")


if __name__ == "__main__":
    main()
