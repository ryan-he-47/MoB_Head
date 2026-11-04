# Showcase: greedy autoregressive generation with the distilled MoB head vs the
# native teacher head, on a batch of prompts. Milestone material for the user.
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from mob.mob_common import H, M, MoBHead, decode_greedy

PROMPTS = [
    "The little boy was playing with",
    "Once upon a time there was a",
    "The dog ran to the park and",
    "My mother said I should",
    "The children went to school and",
    "She looked at the baby and",
    "I have a red ball and",
    "The cat sat on the",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=r".")
    ap.add_argument("--head", default=r".\runs\p2_local_10m_e3\head_final.pt")
    ap.add_argument("--out", default=r".\runs\showcase_samples.txt")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--k", type=int, default=32, help="mixture components (must match checkpoint)")
    ap.add_argument("--width", type=int, default=2458, help="head MLP width (must match checkpoint)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_dir = os.path.join(args.data_root, "LLM-Research--Llama-3.2-1B", "snapshots", "master")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16).cuda().eval()
    head = MoBHead(k=args.k, width=args.width).cuda()
    head.load_state_dict(torch.load(args.head))
    head.eval()
    cb = np.load(os.path.join(args.data_root, "data", "codebook.npz"))
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()
    vid_to_tok = torch.full((1 << M,), -1, dtype=torch.long, device="cuda")
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long().cuda()] = \
        torch.arange(len(cb["vertex_id"]), device="cuda")

    lines = []
    with torch.no_grad():
        for prompt in PROMPTS:
            ids_mob = tok(prompt, return_tensors="pt").input_ids.cuda()
            ids_tea = ids_mob.clone()
            for _ in range(args.tokens):
                out = model(ids_mob, output_hidden_states=True)
                h = out.hidden_states[-1][:, -1].float()
                bl, pl = head(h)
                log_pi = F.log_softmax(pl, dim=-1)
                nt = decode_greedy(bl, log_pi, vid_to_tok, codes_all)[0].item()
                ids_mob = torch.cat([ids_mob, torch.tensor([[nt]], device="cuda")], 1)
                nt_t = model(ids_tea).logits[0, -1].argmax().item()
                ids_tea = torch.cat([ids_tea, torch.tensor([[nt_t]], device="cuda")], 1)
            mob_txt = tok.decode(ids_mob[0].tolist(), skip_special_tokens=True)
            tea_txt = tok.decode(ids_tea[0].tolist(), skip_special_tokens=True)
            lines.append(f"prompt: {prompt!r}")
            lines.append(f"  MoB-greedy: {mob_txt}")
            lines.append(f"  teacher   : {tea_txt}")
            lines.append("")
            print(f"done: {prompt!r}")

    text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
