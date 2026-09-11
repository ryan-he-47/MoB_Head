# Staged environment diagnostic for the race3 mid-load deaths (2026-09-11).
# Facts to discriminate: no traceback, exit 120, death at random positions
# during safetensors per-tensor loading, node-independent, quota already freed.
# Run with -X faulthandler (in the sbatch) so a native crash (segfault/SIGBUS)
# dumps the Python stack at the exact faulting call.
import os
import resource
import sys

import torch

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

print("[1] versions: python", sys.version.split()[0], "| torch", torch.__version__,
      "| cuda_avail", torch.cuda.is_available(), flush=True)
import transformers, safetensors
print("[2] transformers", transformers.__version__, "| safetensors", safetensors.__version__, flush=True)

x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
y = (x @ x).sum().item()
print("[3] cuda matmul OK", y, "| RSS", f"{rss_gb():.1f}GB", flush=True)

from safetensors import safe_open
import glob
shards = sorted(glob.glob("models/SmolLM2-1.7B/*.safetensors"))
print("[4] shards:", [os.path.basename(s) for s in shards], flush=True)
n = 0
for sh in shards:
    with safe_open(sh, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)  # full CPU read of every tensor
            n += t.numel()
    print(f"    read {os.path.basename(sh)} done, cumulative params {n/1e9:.2f}B, RSS {rss_gb():.1f}GB", flush=True)
print("[5] raw safetensors read OK:", n / 1e9, "B params", flush=True)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("models/SmolLM2-1.7B", dtype=torch.float16)
print("[6] from_pretrained OK, RSS", f"{rss_gb():.1f}GB", flush=True)
model = model.cuda()
print("[7] .cuda() OK, RSS", f"{rss_gb():.1f}GB",
      "| params", sum(p.numel() for p in model.parameters()), flush=True)
ids = torch.arange(128, device="cuda").unsqueeze(0)
out = model(ids, output_hidden_states=True)
print("[8] forward OK: logits", tuple(out.logits.shape), "h", tuple(out.hidden_states[-1].shape), flush=True)
print("DIAG_ALL_GREEN", flush=True)
