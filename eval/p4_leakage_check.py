# Leakage check for the P4 joint run (docs/14): training minimizes the
# UNrenormalized -log p_mix(code), but acceptance uses the logZ-renormalized
# NLL. If joint training leaks probability mass onto the 2816 unused vertices,
# the acceptance metric stalls while train NLL keeps dropping. logZ per
# position = log mass on valid vertices (<=0; closer to 0 = less leakage).
# Usage: run twice — with and without the adapter — and compare.
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from mob.mob_common import H, M, MoBHead, mob_mix_logprob

ROOT = r"E:\Trash_things\llm_codec"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n-chunks", type=int, default=16)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model = AutoModelForCausalLM.from_pretrained(
        ROOT + r"\LLM-Research--Llama-3.2-1B\snapshots\master",
        dtype=torch.bfloat16).cuda().eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).cuda().eval()
        backbone = model.get_base_model().model
    else:
        backbone = model.model
    head = MoBHead(k=128, width=3072)
    head.load_state_dict(torch.load(args.head, map_location="cpu"))
    head = head.cuda().float().eval()
    cb = np.load(ROOT + r"\data\codebook.npz")
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32)).cuda()

    dev = np.load(ROOT + r"\data\fineweb_dev_tokens.npy")
    T = 512
    n = min(args.n_chunks, len(dev) // (T + 1))
    chunks = torch.from_numpy(dev[: n * (T + 1)].reshape(n, T + 1).copy()).long().cuda()

    nlls, logzs, raw = [], [], []
    with torch.no_grad():
        for i in range(0, n, 2):
            x = chunks[i:i + 2]
            h = backbone(x[:, :-1], use_cache=False).last_hidden_state.float().reshape(-1, H)
            bl, pl = head(h)
            log_pi = F.log_softmax(pl, dim=-1)
            tgt = x[:, 1:].reshape(-1)
            lp_true = mob_mix_logprob(bl, log_pi, codes_all[tgt].unsqueeze(1))[:, 0]
            lz = []
            for s in range(0, h.shape[0], 8):
                b = bl[s:s + 8]
                term = torch.einsum("vm,nkm->nvk", codes_all, b)
                base = F.logsigmoid(-b).sum(-1).unsqueeze(1)
                mix = torch.logsumexp(log_pi[s:s + 8].unsqueeze(1) + term + base, dim=-1)
                lz.append(torch.logsumexp(mix, dim=-1))
            logzs.append(torch.cat(lz))
            nlls.append(torch.cat(lz) - lp_true)
            raw.append(-lp_true)
    logZ = torch.cat(logzs)
    print(f"adapter={args.adapter}")
    print(f"mob_dev_nll (renormalized): {torch.cat(nlls).mean().item():.4f}")
    print(f"mean logZ (valid-vertex log-mass): {logZ.mean().item():.4f}")
    print(f"mean valid-vertex mass: {logZ.exp().mean().item():.4f}")
    print(f"raw unnormalized NLL: {torch.cat(raw).mean().item():.4f}")


if __name__ == "__main__":
    main()
