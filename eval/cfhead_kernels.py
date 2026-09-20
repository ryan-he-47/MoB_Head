"""Triton fast paths for CFHead coarse/fine — vendored from
external/CFHead/kernel/cfhead_triton.py (HigherIdeal/CFHead, research license;
see that file for the originals). Only the two kernels the head-to-head
actually uses: general uint8 coarse (any C <= 255) and fine GEMV.
Optional: used by eval.cfhead_rt when `import triton` succeeds and inputs are
CUDA; otherwise cfhead_rt falls back to the chunked torch path.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _cfhead_coarse_kernel(lut_ptr, index_t_ptr, out_ptr,
                              V: tl.constexpr, N: tl.constexpr, C: tl.constexpr,
                              BLOCK_V: tl.constexpr):
        block_id = tl.program_id(0)
        batch_id = tl.program_id(1)
        offsets = block_id * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = offsets < V
        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for tile_id in tl.range(0, N):
            cluster = tl.load(index_t_ptr + tile_id * V + offsets, mask=mask, other=0).to(tl.int32)
            values = tl.load(lut_ptr + (batch_id * N + tile_id) * C + cluster,
                             mask=mask, other=0.0).to(tl.float32)
            acc += values
        tl.store(out_ptr + batch_id * V + offsets, acc, mask=mask)

    @triton.jit
    def _cfhead_fine_kernel(hidden_ptr, weight_ptr, candidate_ptr, out_ptr,
                            K: tl.constexpr, H: tl.constexpr,
                            BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr):
        candidate_block_id = tl.program_id(0)
        batch_id = tl.program_id(1)
        k_offsets = candidate_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        token_ids = tl.load(candidate_ptr + batch_id * K + k_offsets,
                            mask=k_mask, other=0).to(tl.int64)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for h_start in tl.range(0, H, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            h_vals = tl.load(hidden_ptr + batch_id * H + h_offsets,
                             mask=h_mask, other=0.0).to(tl.float32)
            w_vals = tl.load(weight_ptr + token_ids[:, None] * H + h_offsets[None, :],
                             mask=k_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w_vals * h_vals[None, :], axis=1)
        tl.store(out_ptr + batch_id * K + k_offsets, acc, mask=k_mask)


def coarse_triton(lut: torch.Tensor, index_t: torch.Tensor) -> torch.Tensor:
    """[B,N,C] lut + [N,V] uint8 index_t (both CUDA) -> [B,V] fp32 coarse."""
    batch_size, num_tiles, cluster_size = lut.shape
    vocab_size = index_t.shape[1]
    lut = lut.contiguous()
    index_t = index_t.contiguous()
    out = torch.empty((batch_size, vocab_size), device=lut.device, dtype=torch.float32)
    block_v = 256
    grid = (triton.cdiv(vocab_size, block_v), batch_size)
    _cfhead_coarse_kernel[grid](lut, index_t, out, vocab_size, num_tiles,
                                cluster_size, BLOCK_V=block_v, num_warps=8)
    return out


def fine_triton(hidden: torch.Tensor, weight: torch.Tensor,
                candidate_idx: torch.Tensor) -> torch.Tensor:
    """hidden [B,H] + weight [V,H] + candidate_idx [B,K] (CUDA) -> [B,K] fp32."""
    batch_size, hidden_size = hidden.shape
    top_k = candidate_idx.shape[1]
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    candidate_idx = candidate_idx.contiguous()
    out = torch.empty((batch_size, top_k), device=hidden.device, dtype=torch.float32)
    block_k, block_h = 32, 64
    grid = (triton.cdiv(top_k, block_k), batch_size)
    _cfhead_fine_kernel[grid](hidden, weight, candidate_idx, out, top_k,
                              hidden_size, BLOCK_K=block_k, BLOCK_H=block_h,
                              num_warps=8)
    return out
