
import os
# Qwen3.5-2B tap-point verification (BEFORE codebook/distillation, 2026-09-14):
# the new arch is a multimodal wrapper (Qwen3_5ForConditionalGeneration) with
# hybrid Gated DeltaNet/Attention + MTP. Must confirm:
#   1. the model loads under this transformers (>= 4.57 needed; cluster 5.16.1)
#   2. hidden_states[-1] of the TEXT backbone equals the lm_head input
#      (tied embeddings => logits == h @ E.T), i.e. the MoB tap point is valid
#   3. report embedding tensor names (codebook builder needs them)
# Also runs a tiny greedy generation as a sanity check.
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("MOB_SCRATCH", "."), "models/Qwen3.5-2B")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).cuda().eval()
print("class:", type(model).__name__)
cfg = model.config
print("vocab:", cfg.vocab_size, "| hidden:", cfg.hidden_size, "| layers:",
      getattr(cfg, "num_hidden_layers", "?"), "| tied:", getattr(cfg, "tie_word_embeddings", "?"))

E = model.get_input_embeddings().weight
print("embedding:", tuple(E.shape), E.dtype)
names = [n for n, _ in model.named_parameters() if "embed" in n.lower()][:6]
print("embed-ish params:", names)

ids = tok("The little boy was playing with", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    out = model(ids, output_hidden_states=True, use_cache=False)
h = out.hidden_states[-1]
logits_manual = F.linear(h.float(), E.float())
diff = (logits_manual - out.logits.float()).abs()
rel = (diff / (out.logits.float().abs() + 1e-3)).max().item()
print(f"hidden_states[-1] == lm_head input?  max|diff|={diff.max().item():.4f}  max rel={rel:.4%}")
assert rel < 0.5, "TAP POINT MISMATCH - do not distill from hidden_states[-1]!"
print("TAP_OK")

with torch.no_grad():
    gen = model.generate(ids, max_new_tokens=16, do_sample=False)
print("sample:", repr(tok.decode(gen[0][ids.shape[1]:])))
print("VERIFY_DONE")
