# Chinese-emergence qualitative probe (user idea 2026-09-13): the best Llama-line
# MoB head (distilled on English-only FineWeb-Edu) meets Chinese prompts.
# Question: does the head extrapolate to Chinese regions of hidden space (OOD
# for its training distribution), i.e. does Chinese "emerge"? Three decode
# paths side by side per prompt: native greedy / MoB round / MoB cascade
# rerank (mixture proposal). Qualitative only — rigorous PPL/agree waits for
# Chinese dev data.
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob.mob_common import (MoBHead, mob_mix_logprob, mixture_topk,
                            decode_greedy)

PROMPTS = [
    "从前有座山，山里有座庙，",
    "中国的首都是",
    "今天天气很好，我决定",
    "人工智能的未来",
    "床前明月光，",
    "小明有三个苹果，吃了两个，还剩",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r"E:\Trash_things\llm_codec")
    ap.add_argument("--head", default=r"E:\Trash_things\llm_codec\runs\head_final_cont.pt")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--width", type=int, default=3072)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--out", default=r"E:\Trash_things\llm_codec\runs\chinese_qual.txt")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_dir = os.path.join(args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()
    E = model.get_input_embeddings().weight

    cb = np.load(os.path.join(args.data_root, "data", "codebook.npz"))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    CB_M = codes_all.shape[1]
    vid_to_tok = torch.full((1 << CB_M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().cuda()] = \
        torch.arange(len(cb["vertex_id"]), device="cuda")

    head = MoBHead(k=args.k, width=args.width, depth=3, m=CB_M).cuda()
    head.load_state_dict(torch.load(args.head))
    head.eval()

    lines = []
    def emit(s):
        print(s, flush=True)
        lines.append(s)

    with torch.no_grad():
        for pr in PROMPTS:
            ids = tok(pr, return_tensors="pt").input_ids.cuda()
            paths = {"native": [], "mob_round": [], "mob_rerank": []}
            # three independent rollouts from the same prompt
            for mode in paths:
                cur_ids = ids
                past = None
                for _ in range(args.tokens):
                    out = model(cur_ids, past_key_values=past, use_cache=True,
                                output_hidden_states=True)
                    past = out.past_key_values
                    h = out.hidden_states[-1][:, -1].float()
                    if mode == "native":
                        nxt = out.logits[:, -1].argmax(-1)
                    else:
                        bl, pl = head(h)
                        log_pi = F.log_softmax(pl, -1)
                        if mode == "mob_round":
                            nxt = decode_greedy(bl, log_pi, vid_to_tok, codes_all, m=CB_M)
                        else:  # cascade: mixture top-32 + exact E-row rescore
                            cand = mixture_topk(bl, log_pi, codes_all, 32)[0]
                            nxt = cand.gather(0, (E[cand].float() @ h[0]).argmax(0, keepdim=True))
                    paths[mode].append(int(nxt.reshape(-1)[0]))
                    cur_ids = nxt.reshape(1, 1)
            emit(f"\nprompt: {pr!r}")
            for mode, seq in paths.items():
                emit(f"  {mode:10s}: {tok.decode(seq)!r}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
