#!/bin/bash
#SBATCH --job-name=setup_llmcodec
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/setup_%j.out

# One-shot cluster setup for llm_codec B+A run:
#   1. install missing packages into cose2 (torch 2.5.1+cu121 already present, V100-safe)
#   2. download Llama-3.2-1B from ModelScope
#   3. build FineWeb-Edu token caches (dev 2000 docs + train 1M docs)
set -euxo pipefail
PY=/home/yihe47/miniconda3/envs/cose2/bin/python
cd /gpfs1/home/yihe47/llm_codec

echo "=== sanity ==="
$PY --version
curl -sI https://pypi.org -o /dev/null -w 'pypi:%{http_code}\n'
curl -sI https://huggingface.co -o /dev/null -w 'hf:%{http_code}\n'
$PY -c "import torch; print('torch', torch.__version__)"

echo "=== pip installs ==="
$PY -m pip install --no-cache-dir transformers==5.16.1 datasets==5.0.1 safetensors modelscope
# guard: pip must not have upgraded torch to a wheel without sm_70 (V100) support
$PY - <<'EOF'
import torch
print("torch after installs:", torch.__version__)
if not torch.__version__.startswith("2.5.1"):
    raise SystemExit("TORCH_UPGRADED_BY_PIP - needs manual fix (install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126)")
import transformers, datasets
print("transformers", transformers.__version__, "datasets", datasets.__version__)
EOF

echo "=== model weights (ModelScope) ==="
$PY - <<'EOF'
from modelscope import snapshot_download
p = snapshot_download('LLM-Research/Llama-3.2-1B', local_dir='LLM-Research--Llama-3.2-1B/snapshots/master')
print("downloaded to", p)
EOF
ls -la LLM-Research--Llama-3.2-1B/snapshots/master/config.json LLM-Research--Llama-3.2-1B/snapshots/master/model.safetensors

echo "=== fine-tuning config check ==="
$PY - <<'EOF'
from transformers import AutoConfig, AutoTokenizer
c = AutoConfig.from_pretrained('LLM-Research--Llama-3.2-1B/snapshots/master')
print("vocab", c.vocab_size, "hidden", c.hidden_size, "tied", c.tie_word_embeddings)
t = AutoTokenizer.from_pretrained('LLM-Research--Llama-3.2-1B/snapshots/master')
print("tokenizer ok, eos", t.eos_token_id)
EOF

echo "=== build fineweb token caches ==="
$PY scripts/build_fineweb_cache.py \
  --model-dir LLM-Research--Llama-3.2-1B/snapshots/master \
  --out-dir data --train-docs 1000000 --dev-docs 2000

echo "SETUP_DONE"
