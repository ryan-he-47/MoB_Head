# Generation-side PPL for the decode paths (user question 2026-09-02):
# "perplexity of the rerank output" = perplexity of greedy continuations UNDER
# THE NATIVE HEAD. The cascade is a decode path, not a density over V, so a
# restricted 32-way softmax PPL would be trivially optimistic (p' = p/mass32);
# scoring generated text under the native head is the honest fluency proxy.
# Paths compared: native argmax / mixture-argmax / cascade rerank.
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob_common import H, M, MoBHead, decode_mixture_argmax, decode_rerank_native

MODEL_DIR = r"E:\Trash_things\llm_codec\LLM-Research--Llama-3.2-1B\snapshots\master"
PROMPTS = [
    "The little boy was playing with",
    "Once upon a time there was a",
    "My mother said I should",
    "The dog ran to the park and",
]
GEN = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=r"E:\Trash_things\llm_codec\runs\head_latest_wsd.pt")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--width", type=int, default=3072)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).cuda().eval()
    cb = np.load(r"E:\Trash_things\llm_codec\data\codebook.npz")
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    head = MoBHead(k=args.k, width=args.width).cuda()
    head.load_state_dict(torch.load(args.head))
    head.eval()
    E = model.get_input_embeddings().weight

    def gen(prompt, mode):
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        n_prompt = ids.shape[1]
        with torch.no_grad():
            for _ in range(GEN):
                if mode == "native":
                    nt = model(ids).logits[0, -1].argmax().item()
                else:
                    out = model.model(ids)
                    h = out.last_hidden_state[:, -1].float()
                    bl, pl = head(h)
                    lp = F.log_softmax(pl, -1)
                    if mode == "mixture":
                        nt = decode_mixture_argmax(bl, lp, codes_all)[0].item()
                    else:
                        nt = decode_rerank_native(bl, lp, codes_all, E, h)[0].item()
                ids = torch.cat([ids, torch.tensor([[nt]], device="cuda")], 1)
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        nll = F.cross_entropy(logits[:-1], ids[0, 1:], reduction="none")
        return nll[n_prompt - 1:]  # NLL of generated positions only

    per = {"native": [], "mixture": [], "rerank": []}
    print(f"{'prompt':40s} | {'native':>7s} | {'mixture':>7s} | {'rerank':>7s}")
    for p in PROMPTS:
        for mode in per:
            per[mode].append(gen(p, mode))
        print(f"{p[:40]:40s} | {torch.cat(per['native'])[-GEN:].exp().mean().item():7.2f} | "
              f"{torch.cat(per['mixture'])[-GEN:].exp().mean().item():7.2f} | "
              f"{torch.cat(per['rerank'])[-GEN:].exp().mean().item():7.2f}")
    print("\nPPL of generated text under the native head (exp of mean NLL, all positions):")
    for mode, arrs in per.items():
        nll = torch.cat(arrs).mean().item()
        print(f"  {mode:8s}: NLL {nll:.3f}  PPL {np.exp(nll):.2f}")


if __name__ == "__main__":
    main()
