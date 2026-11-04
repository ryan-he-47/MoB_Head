#!/bin/bash
#SBATCH --exclude=gpu-v100s-06
#SBATCH --job-name=smollm2_prep
#SBATCH --partition=tiny
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm_smollm2_prep_%j.out

# SmolLM2-1.7B prep for the Frankenstein-hypothesis probe (docs/01-method.md): download
# weights, build the 16-bit semantic-tree codebook on CPU, retokenize
# FineWeb-Edu with the SmolLM2 tokenizer (300k docs ~= 300M tokens, enough for
# the 50k-step probe).
set -euxo pipefail
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=${PYTHON:-python}
MDIR=models/SmolLM2-1.7B

$PY - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("HuggingFaceTB/SmolLM2-1.7B", local_dir="models/SmolLM2-1.7B")
print("weights at", p)
EOF

$PY -m tools.build_codebook \
  --model-dir $MDIR --device cpu \
  --out-npz data/codebook_smollm2.npz \
  --out-meta data/codebook_smollm2_meta.json

$PY -m tools.build_fineweb_cache \
  --model-dir $MDIR \
  --out-dir data/smollm2_tmp \
  --train-docs 300000 --dev-docs 2000
mv data/smollm2_tmp/fineweb_train_tokens.npy data/fineweb_smollm2_tokens.npy
mv data/smollm2_tmp/fineweb_dev_tokens.npy data/fineweb_smollm2_dev_tokens.npy
rmdir data/smollm2_tmp

echo SMOLLM2_PREP_DONE
