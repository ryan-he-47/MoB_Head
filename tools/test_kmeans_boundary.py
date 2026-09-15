# Equivalence proof-by-numbers for the kmeans_vertex_init sampler rewrite
# (user challenge 2026-09-15): old = sample (position,k) PAIRS ~ w, then take
# the pair's token; new = aggregate weights per token (scatter_add over V) and
# sample tokens directly. Verify: (a) induced token distributions are EXACTLY
# equal (algebraic identity, checked numerically); (b) empirical sample
# frequencies match within Monte-Carlo noise; (c) new path survives past 2^24.
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
from train.p2_train import kmeans_vertex_init

torch.manual_seed(0)

# ---- (a) exact distribution equality, small random case ----
V, topk, npos = 5000, 1024, 4096  # 4.2M pairs < 2^24 (old path legal)
tids = torch.randint(0, V, (npos, topk)).cuda()
tps = torch.softmax(torch.randn(npos, topk), dim=-1).cuda()
w_pair = tps.reshape(-1)                       # old: categorical over pairs
pair_probs = w_pair / w_pair.sum()
induced = torch.zeros(V).cuda().scatter_add_(0, tids.reshape(-1), pair_probs)
w_v = torch.zeros(V).cuda().scatter_add_(0, tids.reshape(-1), tps.reshape(-1).float())
new_probs = w_v / w_v.sum()
print("max |induced - new| =", (induced - new_probs).abs().max().item())
assert torch.allclose(induced, new_probs, atol=1e-6)

# ---- (b) empirical frequencies, both samplers ----
def old_samp(tids, tps, n_samp, seed):
    g = torch.Generator().manual_seed(seed)
    flat_w = tps.reshape(-1)
    s = torch.multinomial(flat_w.cpu(), n_samp, replacement=True, generator=g).cuda()
    return tids.reshape(-1)[s]

def new_samp(tids, tps, n_samp, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.zeros(V).cuda().scatter_add_(0, tids.reshape(-1), tps.reshape(-1).float())
    return torch.multinomial(w.cpu(), n_samp, replacement=True, generator=g).cuda()

n_samp = 200000
f_old = torch.bincount(old_samp(tids, tps, n_samp, 7), minlength=V).float() / n_samp
f_new = torch.bincount(new_samp(tids, tps, n_samp, 7), minlength=V).float() / n_samp
true = induced
err_old = (f_old - true).abs().max().item()
err_new = (f_new - true).abs().max().item()
print(f"empirical max|freq - true|: old {err_old:.5f}, new {err_new:.5f}  (MC noise scale ~{1/(n_samp**0.5):.4f})")

# ---- (c) past the 2^24 boundary (the batch-16 crash case) ----
V2, m, K, npos2 = 248320, 18, 128, 32768  # 33.5M pairs > 2^24
codes = torch.randint(0, 2, (V2, m), dtype=torch.float32).cuda()
tids2 = torch.randint(0, V2, (npos2, topk)).cuda()
tps2 = torch.softmax(torch.randn(npos2, topk), dim=-1).cuda()
out = kmeans_vertex_init(codes, tids2, tps2, 4.0, K)
assert out.shape == (K, m)
print("init logits:", tuple(out.shape), "| BOUNDARY_OK")
