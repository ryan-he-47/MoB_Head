# Shared MoB (Mixture of factorized Bernoullis) head definition + stable loss math.
# Used by p2_train.py and p3_eval.py. Spec: docs/03-experiment-plan.md sections 4.2-4.3.
import torch
import torch.nn as nn
import torch.nn.functional as F

H = 2048      # Llama-3.2-1B hidden size
WIDTH = 2458  # 1.2 * H
K = 32        # mixture components
M = 17        # code bits (ceil(log2(128256)))


class MoBHead(nn.Module):
    def __init__(self, k=K, width=WIDTH):
        super().__init__()
        self.k = k
        self.trunk = nn.Sequential(
            nn.Linear(H, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
        )
        self.out = nn.Linear(width, k * M + k)
        nn.init.normal_(self.out.weight, std=1e-3)

    def forward(self, h):
        z = self.out(self.trunk(h))
        return z[:, : self.k * M].view(-1, self.k, M), z[:, self.k * M :]  # bit_logits, pi_logits


def mob_mix_logprob(bit_logits, log_pi, codes):
    """log p_mix(code) for a batch of positions.
    bit_logits [n,K,M] fp32, log_pi [n,K] fp32, codes [n,k,M] fp32 (0/1).
    Per-bit: log Bern(c;z) = c*z + logsigmoid(-z). Returns [n, k]. All fp32."""
    term = torch.einsum("nkm,njm->nkj", codes, bit_logits)
    base = F.logsigmoid(-bit_logits).sum(-1)               # [n,K]
    comp = term + base.unsqueeze(1)                        # [n,k,K]
    return torch.logsumexp(log_pi.unsqueeze(1) + comp, dim=-1)


def decode_greedy(bit_logits, log_pi, vid_to_tok, codes_all):
    """argmax-pi component + per-bit rounding -> token id. Returns [n] long.
    Falls back to the nearest valid vertex (Hamming) if rounding lands on one
    of the 2816 unused vertices (spec: docs/03 section 4.5)."""
    ks = log_pi.argmax(-1)
    bl = bit_logits[torch.arange(ks.numel(), device=bit_logits.device), ks]
    bits = (bl > 0).float()                                    # [n, M]
    vids = torch.zeros(ks.numel(), dtype=torch.long, device=bit_logits.device)
    for j in range(M):
        vids |= bits[:, j].long() << (M - 1 - j)
    toks = vid_to_tok[vids]
    bad = toks < 0
    if bad.any():
        d = (codes_all.unsqueeze(0) != bits[bad].unsqueeze(1)).sum(-1)  # [n_bad, V]
        toks[bad] = d.argmin(-1)
    return toks
