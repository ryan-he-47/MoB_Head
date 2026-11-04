# P1 smoke test: overfit the MoB (Mixture of factorized Bernoullis) head on a
# fixed set of positions, verifying the full distillation pipeline end to end:
#   frozen backbone -> (h, teacher top-k) -> weighted NLL loss -> metrics,
#   component-collapse alarms, and greedy decode vs teacher top-1.
# Spec: docs/01-method.md
import glob
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r".\LLM-Research--Llama-3.2-1B\snapshots\master"
DATA_DIR = r".\data\babylm_data\babylm_10M"
CODEBOOK = r".\data\codebook.npz"

H = 2048          # backbone hidden size
WIDTH = 2458      # 1.2 * H
K = 32            # mixture components
M = 17            # code bits
TOPK = 1024       # distillation target size ()
N_CHUNKS = 16     # 16 x 512 = 8k positions to overfit
SEQ_LEN = 512
BATCH = 4
STEPS = 2000
LR = 2e-3
LAMBDA_ENT = 0.01   # pi entropy regularizer weight
DAEM_T0, DAEM_FRAC = 2.0, 0.10  # pi-logit temperature 2 -> 1 over first 10% steps
INIT_C = 4.0        # bit-logit magnitude for k-means-vertex init
LOSS_CHUNK = 2048   # positions per loss sub-batch

torch.manual_seed(42)


class MoBHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(H, WIDTH), nn.SiLU(),
            nn.Linear(WIDTH, WIDTH), nn.SiLU(),
            nn.Linear(WIDTH, WIDTH), nn.SiLU(),
        )
        self.out = nn.Linear(WIDTH, K * M + K)
        nn.init.normal_(self.out.weight, std=1e-3)

    def forward(self, h):
        z = self.out(self.trunk(h))
        return z[:, : K * M].view(-1, K, M), z[:, K * M :]  # bit_logits, pi_logits


def mob_mix_logprob(bit_logits, log_pi, codes):
    """bit_logits [n,K,M], log_pi [n,K], codes [n,k,M] fp32 (0/1).
    Returns log p_mix(code) under the mixture: [n, k]."""
    term = torch.einsum("nkm,njm->nkj", codes, bit_logits)
    base = F.logsigmoid(-bit_logits).sum(-1)          # [n,K]
    comp = term + base.unsqueeze(1)                    # [n,k,K]
    return torch.logsumexp(log_pi.unsqueeze(1) + comp, dim=-1)  # [n,k]


def extract_targets():
    """Frozen backbone forward: h (fp32), teacher top-k ids + renormalized probs."""
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    all_ids = []
    for path in sorted(glob.glob(DATA_DIR + r"\*.train")):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 300:
                    break
                ids = tok(line.strip()).input_ids
                all_ids.extend(ids + [tok.eos_token_id])
    chunks = [
        all_ids[i : i + SEQ_LEN]
        for i in range(0, N_CHUNKS * SEQ_LEN, SEQ_LEN)
        if len(all_ids[i : i + SEQ_LEN]) == SEQ_LEN
    ]
    print(f"extracting targets from {len(chunks)} chunks...")
    hs, tids, tps = [], [], []
    with torch.no_grad():
        for i in range(0, len(chunks), BATCH):
            ids = torch.tensor(chunks[i : i + BATCH], dtype=torch.long).cuda()
            out = model(ids, output_hidden_states=True)
            h = out.hidden_states[-1][:, :-1].float()      # [b,T-1,2048]
            logits = out.logits[:, :-1]                    # [b,T-1,V]
            top = logits.topk(TOPK, dim=-1)
            w = torch.softmax(top.values.float(), dim=-1)  # renormalize in top-k
            hs.append(h.reshape(-1, H))
            tids.append(top.indices.reshape(-1, TOPK))
            tps.append(w.reshape(-1, TOPK))
    hs = torch.cat(hs)
    tids = torch.cat(tids)
    tps = torch.cat(tps)
    # teacher top-1 per position (for decode agreement)
    top1 = tids[:, 0].clone()
    del model
    torch.cuda.empty_cache()
    return tok, hs, tids, tps, top1


def kmeans_vertex_init(codes_all, tids, tps):
    """k-means (K clusters) over the target-vertex distribution; returns
    initial bit-logits [K, M] as +/- INIT_C at the cluster vertices."""
    flat_ids = tids.reshape(-1)
    flat_w = tps.reshape(-1)
    n_samp = 20000
    samp = torch.multinomial(flat_w, n_samp, replacement=True)
    X = codes_all[flat_ids[samp]].cuda()                    # [n_samp, 17] fp32
    # kmeans++ init
    cent = [X[torch.randint(n_samp, (1,)).item()]]
    for _ in range(K - 1):
        d = torch.cdist(X, torch.stack(cent)).min(1).values
        cent.append(X[d.multinomial(1).item()])
    cent = torch.stack(cent)
    for _ in range(30):
        assign = torch.cdist(X, cent).argmin(1)
        new = cent.clone()
        for j in range(K):
            mask = assign == j
            if mask.any():
                new[j] = X[mask].mean(0)
        if torch.allclose(new, cent, atol=1e-6):
            break
        cent = new
    bits = (cent > 0.5).float()
    return (bits * 2 - 1) * INIT_C  # vertex bit 1 -> +c, bit 0 -> -c


def main():
    cb = np.load(CODEBOOK)
    codes_all = torch.from_numpy(cb["codes"].astype(np.float32))  # [V,17] CPU
    vid_to_tok = torch.full((1 << M,), -1, dtype=torch.long)
    vid_to_tok[torch.from_numpy(cb["vertex_id"]).long()] = torch.arange(len(cb["vertex_id"]))
    vid_to_tok = vid_to_tok.cuda()

    tok, h, tids, tps, top1 = extract_targets()
    n_pos = h.shape[0]
    print(f"positions: {n_pos}, coverage weight sum per position (should be 1): "
          f"{tps.sum(-1).mean():.4f}")

    codes_gpu = codes_all.cuda()
    head = MoBHead().cuda()
    # init: bias of out layer = [vertex bit-logits, zeros for pi]
    with torch.no_grad():
        init_logits = kmeans_vertex_init(codes_gpu, tids, tps)      # [K,M]
        head.out.bias.zero_()
        head.out.bias[: K * M] = init_logits.reshape(-1)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.0)

    def loss_on(idx, temp):
        nll_parts, ent_parts = [], []
        for s in range(0, idx.numel(), LOSS_CHUNK):
            sub = idx[s : s + LOSS_CHUNK]
            bit_logits, pi_logits = head(h[sub])
            log_pi = F.log_softmax(pi_logits / temp, dim=-1)
            codes = codes_gpu[tids[sub]]
            mix = mob_mix_logprob(bit_logits, log_pi, codes)
            nll_parts.append(-(tps[sub] * mix).sum(-1))
            pi = log_pi.exp()
            ent_parts.append((pi * log_pi).sum(-1))
        nll = torch.cat(nll_parts).mean()
        ent = torch.cat(ent_parts).mean()
        return nll + LAMBDA_ENT * ent, nll.detach(), ent.detach()

    all_idx = torch.arange(n_pos, device="cuda")
    t0 = time.time()
    for step in range(STEPS):
        temp = DAEM_T0 + (1.0 - DAEM_T0) * min(1.0, step / (DAEM_FRAC * STEPS))
        loss, nll, ent = loss_on(all_idx, temp)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == STEPS - 1:
            print(f"step {step:5d}  loss={loss.item():.4f}  nll={nll.item():.4f}  "
                  f"ent_term={ent.item():.4f}  T={temp:.2f}  elapsed={time.time()-t0:.0f}s")

    # ---- final evaluation ----
    with torch.no_grad():
        pi_all, agree, ptop1 = [], [], []
        for s in range(0, n_pos, LOSS_CHUNK):
            sub = all_idx[s : s + LOSS_CHUNK]
            bit_logits, pi_logits = head(h[sub])
            log_pi = F.log_softmax(pi_logits, dim=-1)
            pi_all.append(log_pi.exp())
            kstar = log_pi.argmax(-1)
            bl = bit_logits[torch.arange(sub.numel(), device="cuda"), kstar]  # [n,17]
            vids = torch.zeros(sub.numel(), dtype=torch.long, device="cuda")
            for j in range(M):
                vids |= (bl[:, j] > 0).long() << (M - 1 - j)
            toks = vid_to_tok[vids]
            agree.append((toks == top1[sub]))
            codes_top1 = codes_gpu[top1[sub]].unsqueeze(1)        # [n,1,17]
            mix1 = mob_mix_logprob(bit_logits, log_pi, codes_top1)[:, 0]
            ptop1.append(mix1.exp())
        pi_mean = torch.cat(pi_all).mean(0)
        h_pi = -(pi_mean * pi_mean.clamp_min(1e-12).log()).sum().item()
        n_eff = float(np.exp(h_pi))
        dead = int((pi_mean < 1e-4).sum().item())
        agree = torch.cat(agree).float().mean().item()
        ptop1 = torch.cat(ptop1).mean().item()
    print(f"\n== P1 smoke result ==")
    print(f"final weighted NLL: {nll.item():.4f} nats (uniform-vertex baseline: 11.79)")
    print(f"effective components exp(H(pi_bar)): {n_eff:.1f} / {K}")
    print(f"dead components (mean pi < 1e-4): {dead}")
    print(f"decode == teacher top-1 (greedy, argmax pi + round bits): {agree:.4f}")
    print(f"mixture prob on teacher top-1 code: {ptop1:.4f}")

    # ---- sample generations: MoB greedy vs teacher greedy ----
    tok2 = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    prompts = ["The little boy was playing with", "Once upon a time there was a"]
    for prompt in prompts:
        ids_mob = tok2(prompt, return_tensors="pt").input_ids.cuda()
        ids_tea = ids_mob.clone()
        with torch.no_grad():
            for _ in range(24):
                out = model(ids_mob, output_hidden_states=True)
                hh = out.hidden_states[-1][:, -1].float()
                bl, pl = head(hh)
                kstar = pl.argmax(-1).item()
                bits = (bl[0, kstar] > 0).long()
                vid = 0
                for j in range(M):
                    vid |= bits[j].item() << (M - 1 - j)
                nt = vid_to_tok[vid].item()
                if nt < 0:
                    break
                ids_mob = torch.cat([ids_mob, torch.tensor([[nt]], device="cuda")], 1)
                out_t = model(ids_tea)
                nt_t = out_t.logits[0, -1].argmax().item()
                ids_tea = torch.cat([ids_tea, torch.tensor([[nt_t]], device="cuda")], 1)
        print(f"\nprompt: {prompt!r}")
        print(f"  MoB-greedy : {tok2.decode(ids_mob[0, ids_mob.shape[1]-24:].tolist())!r}")
        print(f"  teacher-greedy: {tok2.decode(ids_tea[0, ids_tea.shape[1]-24:].tolist())!r}")


if __name__ == "__main__":
    main()
