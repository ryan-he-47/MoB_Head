# Shared MoB (Mixture of factorized Bernoullis) head definition + stable loss math.
# Used by p2_train.py and p3_eval.py. Spec: docs/03-experiment-plan.md sections 4.2-4.3.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

H = 2048      # Llama-3.2-1B hidden size
WIDTH = 2458  # 1.2 * H
K = 32        # mixture components
M = 17        # code bits (ceil(log2(128256)))


class MoBHead(nn.Module):
    def __init__(self, k=K, width=WIDTH, h=H, depth=3, m=M):
        super().__init__()
        self.k = k
        self.m = m
        layers = [nn.Linear(h, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(width, k * m + k)
        nn.init.normal_(self.out.weight, std=1e-3)

    def forward(self, h):
        z = self.out(self.trunk(h))
        return z[:, : self.k * self.m].view(-1, self.k, self.m), z[:, self.k * self.m :]  # bit_logits, pi_logits


class LinearHead(nn.Module):
    """Bare linear head: h -> k*m+k logits, no hidden layer (probe arm for the
    'is shallower actually better?' question, docs/16 Q2). Same forward
    signature as MoBHead so all decode/loss paths work unchanged."""

    def __init__(self, k=K, h=H, m=M):
        super().__init__()
        self.k = k
        self.m = m
        self.out = nn.Linear(h, k * m + k)
        nn.init.normal_(self.out.weight, std=1e-3)

    def forward(self, h):
        z = self.out(h)
        return z[:, : self.k * self.m].view(-1, self.k, self.m), z[:, self.k * self.m :]


def ecc_parity_matrix(m, p, seed=123, device="cpu"):
    """Fixed random linear parity-check matrix [p, m] over GF(2) for the ECC
    extension (INFRASTRUCTURE ONLY, experiments frozen per user 2026-09-06).
    Each parity bit = XOR of a random ~half subset of the m data bits."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand((p, m), generator=g, device=device) < 0.5).float()


def ecc_extend(codes, pmat):
    """codes [..., M] (0/1) -> [..., M+P] by appending parity bits computed
    online from the parity matrix (deterministic given the data bits)."""
    par = (codes @ pmat.T) % 2
    return torch.cat([codes, par], dim=-1)


def mob_mix_logprob(bit_logits, log_pi, codes):
    """log p_mix(code) for a batch of positions.
    bit_logits [n,K,M] fp32, log_pi [n,K] fp32, codes [n,k,M] fp32 (0/1).
    Per-bit: log Bern(c;z) = c*z + logsigmoid(-z). Returns [n, k]. All fp32."""
    term = torch.einsum("nkm,njm->nkj", codes, bit_logits)
    base = F.logsigmoid(-bit_logits).sum(-1)               # [n,K]
    comp = term + base.unsqueeze(1)                        # [n,k,K]
    return torch.logsumexp(log_pi.unsqueeze(1) + comp, dim=-1)


def decode_greedy(bit_logits, log_pi, vid_to_tok, codes_all, m=M):
    """argmax-pi component + per-bit rounding -> token id. Returns [n] long.
    Falls back to the nearest valid vertex (Hamming) if rounding lands on an
    unused vertex (spec: docs/03 section 4.5). m = code length in bits."""
    ks = log_pi.argmax(-1)
    bl = bit_logits[torch.arange(ks.numel(), device=bit_logits.device), ks]
    bits = (bl > 0).float()                                    # [n, m]
    vids = torch.zeros(ks.numel(), dtype=torch.long, device=bit_logits.device)
    for j in range(m):
        vids |= bits[:, j].long() << (m - 1 - j)
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
    (n=1 decode: K=128 -> 66 MB; n=16 eval chunks -> ~1 GB).
    n==1 fast path routes the code-scoring through a [V,M]@[M,K] GEMM
    (~3-10x faster than the einsum on CPU/GPU, identical math)."""
    if bit_logits.shape[0] == 1:
        term = (codes_all @ bit_logits[0].T).unsqueeze(0)      # [1,V,K]
    else:
        term = torch.einsum("vm,nkm->nvk", codes_all, bit_logits)
    base = F.logsigmoid(-bit_logits).sum(-1).unsqueeze(1)      # [n,1,K]
    return torch.logsumexp(log_pi.unsqueeze(1) + term + base, dim=-1)


def mixture_topk(bit_logits, log_pi, codes_all, topn):
    """Exact top-n token ids by full mixture probability, for n_batch==1 decode.
    Bound pruning: with A[v,k] = log_pi_k + log Bern(v; theta_k),
    L(v) = max_k A[v,k] <= log p_mix(v) <= L(v) + log K = U(v); only vertices
    with L >= (topn-th largest L) - log K can reach the top-n set and get the
    exact logsumexp — result is IDENTICAL to ranking mixture_vertex_logprob.
    Constants (per-bit base + log_pi) are folded into one [V,M+1]@[M+1,K] GEMM.
    codes_all [V,M] in vocab-id order -> returned indices are token ids."""
    K = bit_logits.shape[1]
    z = bit_logits[0]                                          # [K,M]
    base = F.logsigmoid(-z).sum(-1) + log_pi[0]                # [K]
    ones = torch.ones(codes_all.shape[0], 1, dtype=codes_all.dtype, device=codes_all.device)
    A = torch.cat([codes_all, ones], 1) @ torch.cat([z, base.unsqueeze(1)], 1).T  # [V,K]
    L = A.max(-1).values                                       # [V]
    thresh = L.topk(topn).values[-1]
    keep = (L >= thresh - math.log(K)).nonzero(as_tuple=True)[0]  # exact-prune survivors
    exact = torch.logsumexp(A[keep], dim=-1)                   # exp on survivors only
    return keep[exact.topk(topn).indices]                      # [topn] token ids


def decode_mixture_argmax(bit_logits, log_pi, codes_all):
    """Mainline decode (docs/11 section 10, user decision 2026-09-02): rank all
    valid vertices by full mixture probability, take top-1 — no component pick,
    no rounding, no Hamming fallback needed. codes_all is [V,M] in vocab-id
    order, so the argmax index IS the token id. Returns [n] long.
    n==1 uses the bound-pruned exact path; batched callers must chunk over n
    (see mixture_vertex_logprob note)."""
    if bit_logits.shape[0] == 1:
        return mixture_topk(bit_logits, log_pi, codes_all, 1)
    return mixture_vertex_logprob(bit_logits, log_pi, codes_all).argmax(-1)


def decode_rerank_native(bit_logits, log_pi, codes_all, E, h, topn=32,
                         proposal="mixture", vid_to_tok=None):
    """Cascade decode (mainline-adopted 2026-09-02, docs/09 section 6.1): MoB
    proposes top-n candidate tokens, then exact native logits h @ E[cand]^T
    rescore them (reads topn x H rows of the tied embedding table, ~131 KB at
    topn=32 — no full-V matmul). proposal="mixture": exact bound-pruned ranking
    (top-1 agreement 92.2%); proposal="round": round the top-n components by pi
    to their vertices — zero GEMM, ~free (agreement 89.9%, docs/09 section 6.1;
    needs vid_to_tok, n==1 only). E [V,H] any dtype, h [n,H]. Returns [n] long."""
    if proposal == "round":
        assert bit_logits.shape[0] == 1 and vid_to_tok is not None
        m = bit_logits.shape[-1]
        ktop = log_pi[0].topk(min(topn, bit_logits.shape[1])).indices      # [n32]
        bits = (bit_logits[0][ktop] > 0).long()                            # [n32,m]
        shifts = (1 << torch.arange(m - 1, -1, -1, device=bits.device)).long()
        cand = vid_to_tok[(bits * shifts).sum(-1)]
        cand = cand[cand >= 0].unique()                                    # dedup, drop invalid
        scores = h[0].to(E.dtype) @ E[cand].T                              # [<=n32]
        return cand[scores.argmax()].view(1)
    if bit_logits.shape[0] == 1:
        cand = mixture_topk(bit_logits, log_pi, codes_all, topn).unsqueeze(0)
    else:
        cand = mixture_vertex_logprob(bit_logits, log_pi, codes_all).topk(topn, dim=-1).indices
    E_sub = E[cand]                                            # [n,topn,H]
    scores = torch.einsum("nh,nth->nt", h.to(E_sub.dtype), E_sub)
    return cand.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
