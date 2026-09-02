# Encoder-side counterpart experiment (user proposal 2026-09-01):
# MLP from 17-bit code -> word embedding, MSE-supervised, trained to overfit
# the fixed lookup mapping (128256 input-output pairs). If the plug-in PPL test
# passes, the embedding table can be replaced entirely.
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r"E:\Trash_things\llm_codec\LLM-Research--Llama-3.2-1B\snapshots\master"
CODEBOOK = r"E:\Trash_things\llm_codec\data\codebook.npz"
DEV_CACHE = r"E:\Trash_things\llm_codec\data\babylm_dev_tokens.npy"

H = 2048
M_BITS = 17
EVAL_CHUNKS = 16
SEQ_LEN = 512


class EncMLP(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = [nn.Linear(M_BITS, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, H)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=3072)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=r"E:\Trash_things\llm_codec\runs\enc_mlp.pt")
    ap.add_argument("--skip-plug-in-test", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dev = torch.device("cuda")

    cb = np.load(CODEBOOK)
    codes = torch.from_numpy(cb["codes"].astype(np.float32)).to(dev)  # [V,17]

    # embedding table straight from safetensors (no need to spin up the model yet)
    from safetensors import safe_open
    with safe_open(MODEL_DIR + r"\model.safetensors", framework="pt") as f:
        emb = f.get_tensor("model.embed_tokens.weight").float().to(dev)  # [V,2048]
    V = emb.shape[0]
    print(f"V={V}, codes {tuple(codes.shape)}, emb {tuple(emb.shape)}")

    enc = EncMLP(args.width, args.depth).to(dev)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"encoder params: {n_params:,} ({n_params*4/2**20:.1f} MiB fp32 / "
          f"{n_params*2/2**20:.1f} MiB fp16) vs table {V*H*2/2**20:.0f} MiB bf16")

    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)
    emb_var = emb.var().item()

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, V, (args.batch,), device=dev)
        pred = enc(codes[idx])
        loss = F.mse_loss(pred, emb[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                rel = (loss.item() / emb_var)
                print(f"step {step:5d}  mse={loss.item():.5f}  rel_mse={rel:.4f}  "
                      f"elapsed={time.time()-t0:.0f}s")

    # full-table eval: cosine similarity + relative error over all V
    with torch.no_grad():
        preds = []
        for s in range(0, V, 32768):
            preds.append(enc(codes[s:s + 32768]))
        pred = torch.cat(preds)
        cos = F.cosine_similarity(pred, emb, dim=-1)
        rel_err = ((pred - emb).norm(dim=-1) / emb.norm(dim=-1).clamp_min(1e-8))
    print(f"\n== full-table fidelity ==  (target: memorize the lookup)")
    print(f"cosine: mean={cos.mean():.4f} median={cos.median():.4f} "
          f"p5={np.percentile(cos.cpu().numpy(), 5):.4f} p1={np.percentile(cos.cpu().numpy(), 1):.4f} "
          f"min={cos.min():.4f}")
    print(f"relative L2 error: mean={rel_err.mean():.4f} p95={np.percentile(rel_err.cpu().numpy(), 95):.4f}")

    torch.save(enc.state_dict(), args.out)

    # ---- decisive test: plug-in PPL on babylm_dev with native lm_head ----
    if args.skip_plug_in_test:
        return
    print("\n== plug-in PPL test (backbone + native lm_head, embedding := MLP(code)) ==")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).cuda().eval()
    dev_ids = np.load(DEV_CACHE)
    T = SEQ_LEN
    n = min(EVAL_CHUNKS, len(dev_ids) // (T + 1))
    chunks = torch.from_numpy(dev_ids[: n * (T + 1)].reshape(n, T + 1).copy()).long().cuda()
    enc_bf = enc  # compute in fp32, cast to bf16 at injection

    def ppl(use_mlp):
        nlls, cnts = [], []
        with torch.no_grad():
            for i in range(0, n, 4):
                x = chunks[i:i + 4]
                if use_mlp:
                    e = enc_bf(codes[x[:, :-1]].float()).to(torch.bfloat16)
                    hs = model.model(inputs_embeds=e).last_hidden_state
                else:
                    hs = model.model(input_ids=x[:, :-1]).last_hidden_state
                logits = model.lm_head(hs).float()
                nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      x[:, 1:].reshape(-1), reduction="none")
                nlls.append(nll)
        return torch.cat(nlls).mean().item()

    ppl_native = ppl(False)
    ppl_mlp = ppl(True)
    print(f"babylm_dev NLL: native-embedding {ppl_native:.4f} (ppl {np.exp(ppl_native):.1f}) | "
          f"mlp-embedding {ppl_mlp:.4f} (ppl {np.exp(ppl_mlp):.1f}) | "
          f"ratio {np.exp(ppl_mlp - ppl_native):.3f}x")


if __name__ == "__main__":
    main()
