"""CFHead runtime (evaluation-only) — shared by all head-to-head harnesses.

Math-equivalent to external/CFHead/algorithm/CFHead.py's torch fallback path
(verified: reconstructed-GEMM coarse == LUT+gather bitwise, repro report §3.6;
our cache builder matches the repo selector same-machine to 2.3e-4 / 99.9999%).

Coarse:  coarse[v] = sum_t <h_tile_t, codebook[t, index_map[v, t]]>   (fp32)
Fine:    exact GEMV of h against original weight rows of candidates (hidden dtype)

Only depends on a cache file built by tools/build_cfhead_cache.py (or the
original repo) and the model's tied [V, H] weight for the fine stage.
"""

from __future__ import annotations

from pathlib import Path

import torch

from eval.cfhead_kernels import TRITON_AVAILABLE
from eval import cfhead_kernels

COARSE_CHUNK_TILES = 32


class CFHeadRT:
    def __init__(self, cache_path: str | Path, weight: torch.Tensor, device: torch.device):
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        meta = obj["metadata"]
        self.meta = meta
        self.codebook = obj["codebook"].to(device=device, dtype=torch.float32)  # [N, C, T]
        self.index_t_u8 = obj["index_map"].t().contiguous().to(device)          # [N, V]
        self.weight = weight.to(device)                                          # [V, H] source dtype
        self.num_tiles, self.vocab_size = self.index_t_u8.shape
        self.tile = self.codebook.shape[2]
        self.device = device

    @torch.no_grad()
    def coarse(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [B, H] -> coarse scores [B, V] fp32."""
        B = hidden.shape[0]
        N, V, C = self.num_tiles, self.vocab_size, self.codebook.shape[1]
        h = hidden.reshape(B, N, self.tile).float()
        lut = torch.einsum("bnt,nct->bnc", h, self.codebook)  # [B, N, C]
        if TRITON_AVAILABLE and hidden.is_cuda:
            return cfhead_kernels.coarse_triton(lut, self.index_t_u8)
        coarse = torch.zeros(B, V, device=hidden.device, dtype=torch.float32)
        for start in range(0, N, COARSE_CHUNK_TILES):
            end = min(start + COARSE_CHUNK_TILES, N)
            idx = self.index_t_u8[start:end].to(torch.long).unsqueeze(0).expand(B, end - start, V)
            vals = torch.gather(lut[:, start:end, :], dim=2, index=idx)
            coarse += vals.sum(dim=1)
        return coarse

    @torch.no_grad()
    def candidates(self, hidden: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k candidate token ids by coarse score. [B, k]"""
        return torch.topk(self.coarse(hidden), k=min(k, self.vocab_size), dim=-1, sorted=False).indices

    @torch.no_grad()
    def fine_logits(self, hidden: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Exact logits of candidates under the original weight. [B, k]"""
        w = self.weight[cand.reshape(-1)]                      # [B*k, H]
        out = torch.bmm(w.view(cand.shape[0], cand.shape[1], -1).to(hidden.dtype),
                        hidden.unsqueeze(-1)).squeeze(-1)
        return out

    @torch.no_grad()
    def select(self, hidden: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Full coarse-fine pick (repo tie semantics: lowest token id among fine-max
        ties — matches CFHead.py's masked_fill/min). Returns (token_id [B], cand [B, k])."""
        cand = self.candidates(hidden, k)
        logits = self.fine_logits(hidden, cand)
        maxl = logits.max(dim=-1, keepdim=True).values
        token_ids = cand.masked_fill(logits != maxl, self.vocab_size).min(dim=-1).values
        return token_ids, cand
