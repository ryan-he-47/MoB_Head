# Diag v2 (2026-09-11): discriminate GPFS-mmap data-read instability vs
# transformers-5.16.1 loader breakage, and trial both candidate fixes.
# A: raw safetensors read with FORCED materialization (float sum per tensor)
#    -> really reads the data off GPFS, checksum-style.
# B: stage-in copy to node-local $TMPDIR, then from_pretrained from the copy.
# C: bypass from_pretrained: from_config + manual state_dict load + cast.
import os
import shutil
import sys

import torch

def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

print("[0] python", sys.version.split()[0], "| torch", torch.__version__, flush=True)
import transformers, safetensors
print("[0] transformers", transformers.__version__, "| safetensors", safetensors.__version__, flush=True)

SRC = "models/SmolLM2-1.7B"
shard = os.path.join(SRC, "model.safetensors")
print("[A] shard size:", os.path.getsize(shard) / 1e9, "GB", flush=True)

# ---- Stage A: forced-materialization raw read -----------------------------
from safetensors import safe_open
n, acc = 0, 0.0
with safe_open(shard, framework="pt") as f:
    keys = list(f.keys())
    for i, k in enumerate(keys):
        t = f.get_tensor(k)
        acc += float(t.float().sum())   # force full data read
        n += t.numel()
        if i % 50 == 0:
            print(f"    A: {i}/{len(keys)} tensors, cum {n/1e9:.2f}B params, RSS {rss_gb():.1f}GB", flush=True)
print(f"[A] RAW READ OK: {n/1e9:.3f}B params, checksum {acc:.4f}, RSS {rss_gb():.1f}GB", flush=True)

# ---- Stage B: stage-in to node-local disk, then from_pretrained -----------
tmp = os.environ.get("TMPDIR", "/tmp")
dst = os.path.join(tmp, f"smollm2_stagein_{os.getpid()}")
os.makedirs(dst, exist_ok=True)
for fn in os.listdir(SRC):
    if fn.endswith((".safetensors", ".json", ".model")) or fn == "tokenizer.json":
        shutil.copy2(os.path.join(SRC, fn), os.path.join(dst, fn))
print("[B] staged to", dst, "files:", sorted(os.listdir(dst)), flush=True)
from transformers import AutoModelForCausalLM, AutoConfig
model = AutoModelForCausalLM.from_pretrained(dst, dtype=torch.float16).cuda()
ids = torch.arange(64, device="cuda").unsqueeze(0)
out = model(ids, output_hidden_states=True)
print("[B] STAGE-IN LOAD + FORWARD OK, logits", tuple(out.logits.shape), flush=True)
del model
torch.cuda.empty_cache()
shutil.rmtree(dst)

# ---- Stage C: from_config + manual state_dict -----------------------------
cfg = AutoConfig.from_pretrained(SRC)
with torch.device("meta"):
    pass
model = AutoModelForCausalLM.from_config(cfg)  # CPU, random init
sd = {}
with safe_open(shard, framework="pt") as f:
    for k in f.keys():
        sd[k] = f.get_tensor(k)
missing = model.load_state_dict(sd, strict=True)  # raises on any mismatch
model = model.to(torch.float16).cuda()
out = model(ids, output_hidden_states=True)
print("[C] MANUAL LOAD + FORWARD OK, logits", tuple(out.logits.shape), flush=True)

print("DIAG2_ALL_GREEN", flush=True)
