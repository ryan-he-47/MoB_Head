# Measure teacher top-k mass coverage on BabyLM-10M (Llama-3.2-1B, frozen, bf16).
# This decides whether the top-40 truncation in the distillation target is safe.
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"E:\Trash_things\llm_codec\LLM-Research--Llama-3.2-1B\snapshots\master"
DATA_DIR = r"E:\Trash_things\llm_codec\data\babylm_data\babylm_10M"
LINES_PER_FILE = 1000
SEQ_LEN = 512
N_CHUNKS = 48
BATCH = 4
KS = (40, 64, 128, 256, 512, 1024)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16).cuda()
    model.eval()

    # pack lines into fixed-length chunks, EOS between lines
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
    print(f"chunks: {len(chunks)} x {SEQ_LEN} tokens")

    mass = {k: [] for k in KS}
    with torch.no_grad():
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            ids = torch.tensor(batch, dtype=torch.long).cuda()
            logits = model(ids).logits[:, :-1]  # predict next token
            logits = logits.reshape(-1, logits.shape[-1])
            lse = torch.logsumexp(logits.float(), dim=-1)
            top = logits.topk(max(KS), dim=-1).values.float()
            for k in KS:
                m = torch.exp(top[:, :k] - lse.unsqueeze(-1)).sum(-1)
                mass[k].append(m.cpu())
    import numpy as np
    for k in KS:
        v = torch.cat(mass[k]).numpy()
        print(f"top-{k:>2} coverage: mean={v.mean():.4f} median={np.median(v):.4f} "
              f"p10={np.percentile(v, 10):.4f} p90={np.percentile(v, 90):.4f}")

if __name__ == "__main__":
    main()
