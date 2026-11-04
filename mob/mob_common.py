# Shared MoB (Mixture of factorized Bernoullis) head definition + stable loss math.
# Used by p2_train.py and p3_eval.py. Spec: docs/01-method.md
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


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class ResMLPHead(nn.Module):
    """Residual MLP head with RMS pre-normalization and a normalized readout."""

    def __init__(self, k=K, width=WIDTH, h=H, depth=5, m=M):
        super().__init__()
        self.k, self.m = k, m
        self.inp = nn.Linear(h, width)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({"norm": RMSNorm(width), "fc": nn.Linear(width, width)})
            for _ in range(depth - 1)
        ])
        self.final_norm = RMSNorm(width)
        self.out = nn.Linear(width, k * m + k)
        nn.init.normal_(self.out.weight, std=1e-3)

    def forward(self, h):
        x = self.inp(h)
        for b in self.blocks:
            x = x + b["fc"](F.silu(b["norm"](x)))
        z = self.out(self.final_norm(x))
        return z[:, : self.k * self.m].view(-1, self.k, self.m), z[:, self.k * self.m :]


class SwiGLUHead(nn.Module):
    """SwiGLU head with RMS pre-normalization, residual blocks and final norm.

    Each block computes x + W3(SiLU(x_n W1) * (x_n W2)), x_n = RMSNorm(x).
    """

    def __init__(self, k=K, h=H, hidden=7300, m=M, blocks=1):
        super().__init__()
        self.k, self.m = k, m
        self.blocks = nn.ModuleList()
        for _ in range(blocks):
            self.blocks.append(nn.ModuleDict({
                "norm": RMSNorm(h),
                "w1": nn.Linear(h, hidden, bias=False),
                "w2": nn.Linear(h, hidden, bias=False),
                "w3": nn.Linear(hidden, h, bias=False),
            }))
        self.final_norm = RMSNorm(h)
        self.out = nn.Linear(h, k * m + k)
        nn.init.normal_(self.out.weight, std=1e-3)

    def forward(self, h):
        x = h
        for b in self.blocks:
            xn = b["norm"](x)
            x = x + b["w3"](F.silu(b["w1"](xn)) * b["w2"](xn))
        z = self.out(self.final_norm(x))
        return z[:, : self.k * self.m].view(-1, self.k, self.m), z[:, self.k * self.m :]


class LinearHead(nn.Module):
    """Linear diagnostic head with the same output contract as MoBHead."""

    def __init__(self, k=K, h=H, m=M):
        super().__init__()
        self.k = k
        self.m = m
        self.out = nn.Linear(h, k * m + k)
        nn.init.normal_(self.out.weight, std=1e-3)

    def forward(self, h):
        z = self.out(h)
        return z[:, : self.k * self.m].view(-1, self.k, self.m), z[:, self.k * self.m :]


def propose_roundflip(bit_logits, log_pi, codes_all, vid_to_tok, topn=32,
                      aug_topn=32, alpha=0.15, u_max=6, per_comp_cap=8):
    """round-topn component modes PLUS flip-enumeration hedge on the top
    `aug_topn` components' uncertain bits (|sigmoid(z)-0.5| <= alpha).
    Validated 2026-09-07 (docs/01-method.md): with aug_topn=32/per_comp_cap=8
    ("diversity first, hedge second, every component a small budget") it
    beats all measured variants — recall 96.5%, hit@top8 100%, rerank 96.5%
    at mean 81 / p95 122 candidates, NO codebook GEMM. n_batch==1 only.
    (Pure adaptive-scope variants and flip-clouds-before-modes orderings
    measured and rejected: cross-component coverage must come first.)"""
    m = bit_logits.shape[-1]
    shifts = (1 << torch.arange(m - 1, -1, -1, device=bit_logits.device)).long()
    ktop = log_pi[0].topk(min(topn, bit_logits.shape[1]), -1).indices
    bits = (bit_logits[0, ktop] > 0).long()
    toks = vid_to_tok[(bits * shifts).sum(-1)]
    toks = toks[toks >= 0]
    extra = []
    for k in ktop[:aug_topn].tolist():
        p = bit_logits[0, k].sigmoid()
        unc = ((p - 0.5).abs() <= alpha).nonzero(as_tuple=True)[0]
        u = min(unc.numel(), u_max)
        unc = unc[:u]
        if u == 0:
            continue
        base = (p > 0.5).long()
        base[unc] = 0
        base_vid = (base * shifts).sum()
        combos = ((torch.arange(2 ** u, device=bit_logits.device).unsqueeze(1)
                   >> torch.arange(u - 1, -1, -1, device=bit_logits.device)) & 1).long()
        vids = base_vid + (combos * shifts[unc]).sum(-1)
        t = vid_to_tok[vids]
        extra.append(t[t >= 0][:per_comp_cap])
    if extra:
        toks = torch.unique(torch.cat([toks] + extra))
    return toks


def propose_adaptive(bit_logits, log_pi, codes_all, vid_to_tok, mode="ratio",
                     ratio=0.05, mass=0.95, alpha=0.25, u_max=12, max_cand=512,
                     floor=0):
    """Adaptive candidate generation exploiting the mixture's probabilistic
    structure (docs/01-method.md). n_batch==1 only.
    1. Dynamic component scope: mode="ratio" keeps components with
       pi_k >= pi_max * ratio (sharp -> few, flat -> many); mode="mass" keeps
       the smallest prefix covering `mass` of the pi mass. `floor` forces at
       least that many components (v2: never starve cross-component coverage).
    2. Per selected component: bits with |sigmoid(z)-0.5| <= alpha are
       "uncertain"; enumerate ALL flip combinations of those bits (capped at
       u_max per component), certain bits fixed at their rounded values.
    v2 ordering fix: candidates are emitted in PRIORITY order (components by
    pi desc, within a component by ascending flip count), dedup keeps the
    first (highest-priority) occurrence, and emission stops at max_cand —
    no arbitrary token-id-sorted truncation (the v1 recall-killer).
    Returns candidate token ids [C] (C <= max_cand, deduped, valid only)."""
    m = bit_logits.shape[-1]
    shifts = (1 << torch.arange(m - 1, -1, -1, device=bit_logits.device)).long()
    pi = log_pi[0].exp()
    order = pi.argsort(descending=True)
    if mode == "ratio":
        n_sel = max(int((pi[order] >= pi[order[0]] * ratio).sum().item()), floor)
    else:
        n_sel = max(int((pi[order].cumsum(0) < mass).sum().item()) + 1, floor)
    sel = order[:n_sel]
    seen = torch.zeros(1 << m, dtype=torch.bool, device=bit_logits.device)
    cands, total = [], 0
    for k in sel.tolist():
        p = bit_logits[0, k].sigmoid()                            # [m]
        unc = ((p - 0.5).abs() <= alpha).nonzero(as_tuple=True)[0]
        u = min(unc.numel(), u_max)
        unc = unc[:u]
        base = (p > 0.5).long()
        base[unc] = 0
        base_vid = (base * shifts).sum()
        if u == 0:
            vids = base_vid.view(1)
        else:
            combos = ((torch.arange(2 ** u, device=bit_logits.device).unsqueeze(1)
                       >> torch.arange(u - 1, -1, -1, device=bit_logits.device)) & 1).long()
            hw = combos.sum(-1).argsort()                        # fewer flips first
            vids = (base_vid + (combos[hw] * shifts[unc]).sum(-1))
        toks = vid_to_tok[vids]
        toks = toks[toks >= 0]
        new = toks[~seen[toks]]
        seen[new] = True
        take = min(new.numel(), max_cand - total)
        if take > 0:
            cands.append(new[:take])
            total += take
        if total >= max_cand:
            break
    if not cands:
        return torch.empty(0, dtype=torch.long, device=bit_logits.device)
    return torch.cat(cands)


def propose_varflip(bit_logits, log_pi, codes_all, vid_to_tok, alpha=0.15,
                    u_max=12, per_comp_cap=16, max_cand=256):
    """Variance-ranked full-scope flip proposal ():
    1) modes of ALL K components, deduped (component modes DO collide —
       converged low-pi clones share vertices; the plain vectorized round-32
       path never deduped, roundflip/adaptive do);
    2) flip enumeration in DESCENDING total-bit-variance order
       (sum_m p(1-p)) — most diffuse component first;
    3) per-component cap, live dedup, global budget max_cand.
    n_batch==1 only. Returns candidate token ids [C], C <= max_cand."""
    m = bit_logits.shape[-1]
    shifts = (1 << torch.arange(m - 1, -1, -1, device=bit_logits.device)).long()
    probs = bit_logits[0].sigmoid()                              # [K,m]
    seen = torch.zeros(1 << m, dtype=torch.bool, device=bit_logits.device)
    cands, total = [], 0
    mode_toks = vid_to_tok[((bit_logits[0] > 0).long() * shifts).sum(-1)]
    mode_toks = mode_toks[mode_toks >= 0]
    if mode_toks.numel():
        new = mode_toks[~seen[mode_toks]]
        cands.append(new)
        seen[new] = True
        total += new.numel()
    var_order = (probs * (1 - probs)).sum(-1).argsort(descending=True)
    for k in var_order.tolist():
        if total >= max_cand:
            break
        p = probs[k]
        unc = ((p - 0.5).abs() <= alpha).nonzero(as_tuple=True)[0]
        u = min(unc.numel(), u_max)
        unc = unc[:u]
        if u == 0:
            continue
        base = (p > 0.5).long()
        base[unc] = 0
        base_vid = (base * shifts).sum()
        combos = ((torch.arange(2 ** u, device=bit_logits.device).unsqueeze(1)
                   >> torch.arange(u - 1, -1, -1, device=bit_logits.device)) & 1).long()

        # variances, descending. Since var = sigmoid(z)(1-sigmoid(z)) is
        # strictly decreasing in |z|, this IS the per-component exact
        # likelihood order (flip the cheapest bits first), tighter than the
        # hamming-count order used by roundflip/adaptive.
        prio = (combos.float() * (p[unc] * (1 - p[unc]))).sum(-1)
        vids = base_vid + (combos[prio.argsort(descending=True)] * shifts[unc]).sum(-1)
        toks = vid_to_tok[vids]
        toks = toks[toks >= 0]
        new = toks[~seen[toks]]
        take = int(min(new.numel(), per_comp_cap, max_cand - total))
        if take > 0:
            cands.append(new[:take])
            seen[new[:take]] = True
            total += take
    if not cands:
        return torch.empty(0, dtype=torch.long, device=bit_logits.device)
    return torch.cat(cands)


def propose_scoreflip(bit_logits, log_pi, codes_all, vid_to_tok, alpha=0.25,
                      u_max=12, per_comp_cap=32, max_cand=256, mass=0.99):
    """'Propose liberally, rescore exactly, truncate by score' (2026-09-19):
    components covering `mass` of pi emit modes + uncertain-bit flip lattices
    (hamming-first); the pooled deduped candidates are rescored with the EXACT
    mixture logprob (logsumexp_k(log_pi_k + logBern)) and truncated to
    max_cand BY SCORE. Approximates mixture_topk without the [V,K] ranking
    GEMM — emission breadth from the flip lattices, ranking from the exact
    density. n_batch==1 only. Returns candidate token ids [<=max_cand]."""
    m = bit_logits.shape[-1]
    shifts = (1 << torch.arange(m - 1, -1, -1, device=bit_logits.device)).long()
    probs = bit_logits[0].sigmoid()
    pi = log_pi[0].exp()
    order = pi.argsort(descending=True)
    n_sel = max(int((pi[order].cumsum(0) < mass).sum().item()) + 1, 1)
    sel = order[:n_sel]
    seen = torch.zeros(1 << m, dtype=torch.bool, device=bit_logits.device)
    cands, total = [], 0
    pool_cap = 8 * max_cand
    for k in sel.tolist():
        if total >= pool_cap:
            break
        p = probs[k]
        unc = ((p - 0.5).abs() <= alpha).nonzero(as_tuple=True)[0]
        u = min(unc.numel(), u_max)
        unc = unc[:u]
        base = (p > 0.5).long()
        if u == 0:
            vids = (base * shifts).sum().view(1)
        else:
            base = base.clone()
            base[unc] = 0
            base_vid = (base * shifts).sum()
            combos = ((torch.arange(2 ** u, device=bit_logits.device).unsqueeze(1)
                       >> torch.arange(u - 1, -1, -1, device=bit_logits.device)) & 1).long()
            prio = (combos.float() * (p[unc] * (1 - p[unc]))).sum(-1)  # see varflip note
            vids = base_vid + (combos[prio.argsort(descending=True)] * shifts[unc]).sum(-1)
        toks = vid_to_tok[vids]
        toks = toks[toks >= 0]
        new = toks[~seen[toks]]
        take = int(min(new.numel(), per_comp_cap, pool_cap - total))
        if take > 0:
            cands.append(new[:take])
            seen[new[:take]] = True
            total += take
    if not cands:
        return torch.empty(0, dtype=torch.long, device=bit_logits.device)
    pooled = torch.cat(cands)
    if pooled.numel() <= max_cand:
        return pooled
    # exact mixture rescoring of the pool: [C,K] @ constants, logsumexp over K
    codes_c = codes_all[pooled]                                  # [C,M] 0/1
    z = bit_logits[0]                                            # [K,M]
    term = codes_c @ z.T                                         # [C,K]
    base_k = F.logsigmoid(-z).sum(-1)                            # [K]
    score = torch.logsumexp(log_pi[0].unsqueeze(0) + term + base_k.unsqueeze(0), dim=-1)
    return pooled[score.topk(max_cand).indices]


def ecc_parity_matrix(m, p, seed=123, device="cpu"):
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
    unused vertex (spec: docs/01-method.md). m = code length in bits."""
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
    """Exact mixture decode: rank all
    valid vertices by full mixture probability, take top-1 — no component pick,
    no rounding, no Hamming fallback needed. codes_all is [V,M] in vocab-id
    order, so the argmax index IS the token id. Returns [n] long.
    n==1 uses the bound-pruned exact path; batched callers must chunk over n
    (see mixture_vertex_logprob note)."""
    if bit_logits.shape[0] == 1:
        return mixture_topk(bit_logits, log_pi, codes_all, 1)
    return mixture_vertex_logprob(bit_logits, log_pi, codes_all).argmax(-1)


def decode_rerank_native(bit_logits, log_pi, codes_all, E, h, topn=32,
                         proposal="mixture", vid_to_tok=None, adaptive_kwargs=None):
    """Cascade decode: MoB
    proposes top-n candidate tokens, then exact native logits h @ E[cand]^T
    rescore them (reads topn x H rows of the tied embedding table, ~131 KB at
    topn=32 — no full-V matmul). proposal="mixture": exact bound-pruned ranking
    (top-1 agreement 92.2%); proposal="round": round the top-n components by pi
    to their vertices — zero GEMM, ~free (agreement 89.9%, docs/01-method.md;
    needs vid_to_tok, n==1 only). E [V,H] any dtype, h [n,H]. Returns [n] long."""
    if proposal == "roundflip":
        assert bit_logits.shape[0] == 1 and vid_to_tok is not None
        cand = propose_roundflip(bit_logits, log_pi, codes_all, vid_to_tok,
                                 **(adaptive_kwargs or {}))
        scores = h[0].to(E.dtype) @ E[cand].T
        return cand[scores.argmax()].view(1)
    if proposal == "adaptive":
        assert bit_logits.shape[0] == 1 and vid_to_tok is not None
        cand = propose_adaptive(bit_logits, log_pi, codes_all, vid_to_tok,
                                **(adaptive_kwargs or {}))
        if cand.numel() == 0:
            cand = mixture_topk(bit_logits, log_pi, codes_all, 1)
        scores = h[0].to(E.dtype) @ E[cand].T
        return cand[scores.argmax()].view(1)
    if proposal == "scoreflip":
        # budget-256-class mainline proposal (2026-09-19, docs/01-method.md): best
        # cheap approximation of mixture_topk — self-truncates on peaked
        # positions (~90-105 candidates), 256 on diffuse ones.
        assert bit_logits.shape[0] == 1 and vid_to_tok is not None
        cand = propose_scoreflip(bit_logits, log_pi, codes_all, vid_to_tok,
                                 **(adaptive_kwargs or {}))
        if cand.numel() == 0:
            cand = mixture_topk(bit_logits, log_pi, codes_all, 1)
        scores = h[0].to(E.dtype) @ E[cand].T
        return cand[scores.argmax()].view(1)
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
