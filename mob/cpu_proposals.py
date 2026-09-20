"""CPU-native (numba JIT) implementations of the MoB flip proposals.

User task 2026-09-19: the python-loop proposals (roundflip/scoreflip) are
launch-overhead-bound on CPU; numba nopython gives C-class speed with zero
compiler toolchain (llvmlite wheel) — the pragmatic equivalent of a C++ port
for this box (no MSVC).

Semantics are EXACTLY mob_common.propose_scoreflip / propose_roundflip:
  scoreflip: pi-mass(0.99) component scope; per component emit the flip
  lattice of the uncertain bits (|p-0.5| <= alpha, u <= u_max) in DESCENDING
  SUM-OF-FLIPPED-BIT-VARIANCES order (user spec 2026-09-19; NOTE: this equals
  the exact per-component likelihood order only within a fixed Hamming
  weight — the true logp cost is sum|z|; kept as specced for parity, with
  sum|z| available as a one-line upgrade);
  per-comp cap over NEW unique tokens; pool cap; exact mixture logprob
  rescore; top-max_cand by score (ties broken by emission index, matching
  torch stable-sort semantics).
  roundflip: top-32 components by pi; modes first; flips on all 32,
  cap 8/comp, HAMMING-count order (legacy semantics, unchanged).

Packed codebook (CPU-cache friendly): codes_packed uint32 [V] (18-bit vid,
bit 0 = MSB, same bit order as mob_common's shifts) + tok_of_vid int32 [2^M].
~1.3 MB total for V=248,320, M=18 (vs 18 MB fp32 codes_all).

All functions are n_batch==1; CPU tensors in, int64 torch tensor out.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import njit


def pack_codebook(codes: np.ndarray, vertex_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """codes [V,M] 0/1, vertex_id [V] int -> (codes_packed uint32 [V],
    tok_of_vid int32 [2^M])."""
    V, M = codes.shape
    shifts = (1 << np.arange(M - 1, -1, -1, dtype=np.int64)).astype(np.uint32)
    packed = (codes.astype(np.uint32) * shifts).sum(axis=1).astype(np.uint32)
    tok = np.full(1 << M, -1, dtype=np.int32)
    tok[vertex_id.astype(np.int64)] = np.arange(V, dtype=np.int32)
    return packed, tok


@njit(cache=True)
def _lattice(z_k, vid_shift, alpha, u_max, vids_out, prio_out, var_mode):
    """Flip-lattice of component z_k [M]: all 2^u assignments of the first
    u = min(n_unc, u_max) uncertain bits (index order, matching the torch
    reference). vids_out[combo] = vertex id; prio_out[combo] = priority such
    that argsort(-prio) == torch's combo order (var_mode 1: sum p(1-p) desc;
    var_mode 0: hamming count asc, i.e. prio = -count)."""
    M = z_k.shape[0]
    unc = np.empty(M, np.int32)
    unc_p = np.empty(M, np.float32)
    n_unc = 0
    for m in range(M):
        p = 1.0 / (1.0 + np.exp(-z_k[m]))
        if abs(p - 0.5) <= alpha:
            unc[n_unc] = m
            unc_p[n_unc] = p
            n_unc += 1
    u = min(n_unc, u_max)
    base = 0
    for m in range(M):
        if z_k[m] > 0:
            keep = True
            for j in range(u):
                if unc[j] == m:
                    keep = False
                    break
            if keep:
                base |= vid_shift[m]
    n = 1 << u
    for combo in range(n):
        vid = base
        prio = 0.0
        for j in range(u):
            if (combo >> (u - 1 - j)) & 1:  # torch reference bit order
                vid |= vid_shift[unc[j]]
                if var_mode == 1:
                    prio += unc_p[j] * (1.0 - unc_p[j])
                else:
                    prio -= 1.0
        vids_out[combo] = vid
        prio_out[combo] = prio
    return n


@njit(cache=True)
def _order_desc_tiebreak(prio, n):
    """argsort(-prio) with emission-index tiebreak (torch-stable semantics):
    key = -prio * 1e9 + idx, ascending."""
    keys = np.empty(n, np.float64)
    for i in range(n):
        keys[i] = -prio[i] * 1e9 + i
    return np.argsort(keys)


@njit(cache=True)
def _scoreflip(bl, log_pi, codes_packed, tok_of_vid, seen, pool_vids, pool_scores,
               vid_shift, alpha, u_max, per_comp_cap, max_cand, mass):
    K, M = bl.shape
    # pi order desc with index tiebreak, then mass scope
    pi_keys = np.empty(K, np.float64)
    for k in range(K):
        pi_keys[k] = -log_pi[k] * 1e9 + k
    order = np.argsort(pi_keys)
    tot = 0.0
    for k in range(K):
        tot += np.exp(log_pi[k])
    cum = 0.0
    n_sel = 1
    for i in range(K):
        cum += np.exp(log_pi[order[i]])
        n_sel = i + 1
        if cum >= mass * tot:
            break
    seen[:] = 0
    pool_cap = pool_vids.shape[0]
    n_pool = 0
    lat_vid = np.empty(4096, np.uint32)
    lat_prio = np.empty(4096, np.float32)
    for i in range(n_sel):
        if n_pool >= pool_cap:
            break
        k = order[i]
        n = _lattice(bl[k], vid_shift, alpha, u_max, lat_vid, lat_prio, 1)
        o = _order_desc_tiebreak(lat_prio, n)
        taken = 0
        for jj in range(n):
            vid = lat_vid[o[jj]]
            tok = tok_of_vid[vid]
            if tok < 0 or seen[vid]:
                continue
            seen[vid] = 1
            pool_vids[n_pool] = vid
            n_pool += 1
            taken += 1
            if taken >= per_comp_cap or n_pool >= pool_cap:
                break
    if n_pool == 0:
        return np.empty(0, np.int32)
    if n_pool <= max_cand:
        out = np.empty(n_pool, np.int32)
        for i in range(n_pool):
            out[i] = tok_of_vid[pool_vids[i]]
        return out
    # exact mixture rescore: score(vid) = logsumexp_k(log_pi_k + sum_m logsig(-z_km) + bits·z_k)
    base_k = np.empty(K, np.float32)
    for k in range(K):
        acc = 0.0
        for m in range(M):
            z = bl[k, m]
            if z > 0:
                acc += -np.log(1.0 + np.exp(-z)) - z  # = logsigmoid(-z) stably
            else:
                acc += -np.log(1.0 + np.exp(z))
        base_k[k] = acc + log_pi[k]
    dots = np.empty(K, np.float32)
    for i in range(n_pool):
        vid = pool_vids[i]
        for k in range(K):
            dot = 0.0
            for m in range(M):
                if (vid >> (M - 1 - m)) & 1:
                    dot += bl[k, m]
            dots[k] = base_k[k] + dot
        best = -1e30
        for k in range(K):
            if dots[k] > best:
                best = dots[k]
        ssum = 0.0
        for k in range(K):
            ssum += np.exp(dots[k] - best)
        pool_scores[i] = best + np.log(ssum)
    top = _order_desc_tiebreak(pool_scores, n_pool)[:max_cand]
    out = np.empty(top.shape[0], np.int32)
    for i in range(top.shape[0]):
        out[i] = tok_of_vid[pool_vids[top[i]]]
    return out


@njit(cache=True)
def _roundflip(bl, log_pi, tok_of_vid, seen, vids_out, vid_shift, topn,
               aug_topn, alpha, u_max, per_comp_cap):
    """EXACT propose_roundflip semantics: per-component lattice in RAW arange
    (combo-index) order, cap = first per_comp_cap VALID tokens of that
    component (dedup does NOT extend the component's budget), final global
    dedup (torch.unique over modes+flips). n_out may hold duplicate vids
    before the final dedup."""
    K, M = bl.shape
    pi_keys = np.empty(K, np.float64)
    for k in range(K):
        pi_keys[k] = -log_pi[k] * 1e9 + k
    order = np.argsort(pi_keys)
    n_out = 0
    # 1) modes of the top-topn components (valid only; dedup at the end)
    for i in range(min(topn, K)):
        z = bl[order[i]]
        mode = 0
        for m in range(M):
            if z[m] > 0:
                mode |= vid_shift[m]
        if tok_of_vid[mode] >= 0:
            vids_out[n_out] = mode
            n_out += 1
    # 2) flips on the top-aug_topn components, arange order, first-N-valid cap
    for i in range(min(aug_topn, K)):
        z = bl[order[i]]
        unc = np.empty(M, np.int32)
        n_unc = 0
        for m in range(M):
            p = 1.0 / (1.0 + np.exp(-z[m]))
            if abs(p - 0.5) <= alpha:
                unc[n_unc] = m
                n_unc += 1
        u = min(n_unc, u_max)
        if u == 0:
            continue
        base = 0
        for m in range(M):
            if z[m] > 0:
                keep = True
                for j in range(u):
                    if unc[j] == m:
                        keep = False
                        break
                if keep:
                    base |= vid_shift[m]
        n = 1 << u
        taken = 0
        for combo in range(n):
            vid = base
            for j in range(u):
                if (combo >> (u - 1 - j)) & 1:  # torch reference bit order
                    vid |= vid_shift[unc[j]]
            if tok_of_vid[vid] >= 0:
                vids_out[n_out] = vid
                n_out += 1
                taken += 1
                if taken >= per_comp_cap:
                    break
    # 3) final global dedup (torch.unique semantics: the SET is what matters)
    seen[:] = 0
    n_uniq = 0
    for i in range(n_out):
        if not seen[vids_out[i]]:
            seen[vids_out[i]] = 1
            vids_out[n_uniq] = vids_out[i]
            n_uniq += 1
    out = np.empty(n_uniq, np.int32)
    for i in range(n_uniq):
        out[i] = tok_of_vid[vids_out[i]]
    return out


class CPUProposals:
    """Wrapper holding the packed codebook + reusable work buffers."""

    def __init__(self, codebook_npz: str, per_comp_cap_sf: int = 32,
                 max_cand_sf: int = 256, pool_factor: int = 8):
        cb = np.load(codebook_npz)
        self.codes_packed, self.tok_of_vid = pack_codebook(cb["codes"], cb["vertex_id"])
        self.M = int(cb["codes"].shape[1])
        self.vid_shift = (1 << np.arange(self.M - 1, -1, -1, dtype=np.int64)).astype(np.uint32)
        self.seen = np.zeros(1 << self.M, dtype=np.uint8)
        self.per_comp_cap_sf = per_comp_cap_sf
        self.max_cand_sf = max_cand_sf
        self.pool_vids = np.empty(pool_factor * max_cand_sf, np.uint32)
        self.pool_scores = np.empty(pool_factor * max_cand_sf, np.float32)
        self.rf_vids = np.empty(4096, np.uint32)

    @staticmethod
    def _to_np(bit_logits: torch.Tensor, log_pi: torch.Tensor):
        return (bit_logits[0].detach().cpu().numpy().astype(np.float32),
                log_pi[0].detach().cpu().numpy().astype(np.float32))

    def scoreflip(self, bit_logits: torch.Tensor, log_pi: torch.Tensor,
                  alpha: float = 0.25, u_max: int = 12, mass: float = 0.99) -> torch.Tensor:
        bl, lp = self._to_np(bit_logits, log_pi)
        toks = _scoreflip(bl, lp, self.codes_packed, self.tok_of_vid, self.seen,
                          self.pool_vids, self.pool_scores, self.vid_shift,
                          alpha, u_max, self.per_comp_cap_sf, self.max_cand_sf, mass)
        return torch.from_numpy(toks.astype(np.int64))

    def roundflip(self, bit_logits: torch.Tensor, log_pi: torch.Tensor,
                  topn: int = 32, aug_topn: int = 32, alpha: float = 0.15,
                  u_max: int = 6, per_comp_cap: int = 8) -> torch.Tensor:
        bl, lp = self._to_np(bit_logits, log_pi)
        toks = _roundflip(bl, lp, self.tok_of_vid, self.seen, self.rf_vids,
                          self.vid_shift, topn, aug_topn, alpha, u_max, per_comp_cap)
        return torch.from_numpy(toks.astype(np.int64))
