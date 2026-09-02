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


def mixture_vertex_logprob(bit_logits, log_pi, codes_all):
    """log p_mix(v) = logsumexp_k(log_pi_k + log Bern(v; theta_k)) for every
    valid vertex, in vocab-id order. bit_logits [n,K,M], log_pi [n,K],
    codes_all [V,M] (0/1). Returns [n,V] fp32.
    Memory: [n,V,K] fp32 intermediate — chunk n for batched use
    (n=1 decode: K=128 -> 66 MB; n=16 eval chunks -> ~1 GB)."""
    term = torch.einsum("vm,nkm->nvk", codes_all, bit_logits)
    base = F.logsigmoid(-bit_logits).sum(-1).unsqueeze(1)      # [n,1,K]
    return torch.logsumexp(log_pi.unsqueeze(1) + term + base, dim=-1)


def decode_mixture_argmax(bit_logits, log_pi, codes_all):
    """Mainline decode (docs/11 section 10, user decision 2026-09-02): rank all
    valid vertices by full mixture probability, take top-1 — no component pick,
    no rounding, no Hamming fallback needed. codes_all is [V,M] in vocab-id
    order, so the argmax index IS the token id. Returns [n] long.
    Batched callers must chunk over n (see mixture_vertex_logprob note)."""
    return mixture_vertex_logprob(bit_logits, log_pi, codes_all).argmax(-1)


def decode_rerank_native(bit_logits, log_pi, codes_all, E, h, topn=32):
    """Cascade decode (evaluated 2026-09-02, docs/09 section 6): MoB mixture
    proposes top-n candidate tokens, then exact native logits h @ E[cand]^T
    rescore them (reads topn x H rows of the tied embedding table, ~131 KB at
    topn=32 — no full-V matmul). Top-1 agreement ceiling = MoB top-n recall of
    native top-1. E [V,H] any dtype, h [n,H]. Returns [n] long."""
    cand = mixture_vertex_logprob(bit_logits, log_pi, codes_all).topk(topn, dim=-1).indices
    E_sub = E[cand]                                            # [n,topn,H]
    scores = torch.einsum("nh,nth->nt", h.to(E_sub.dtype), E_sub)
    return cand.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
