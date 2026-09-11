#!/bin/bash
#SBATCH --job-name=diag_env2
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output=/gpfs1/home/yihe47/llm_codec/logs/diag_env2_%j.out

# Diag v2: A) forced-materialization raw read (checksum), B) stage-in to
# $TMPDIR + from_pretrained, C) from_config + manual state_dict. Any of B/C
# green = the race loader fix.
set -euxo pipefail
cd /gpfs1/home/yihe47/llm_codec
PY=/home/yihe47/miniconda3/envs/cose2/bin/python
df -h ${TMPDIR:-/tmp}
$PY -X faulthandler tools/diag_env2.py
echo DIAG2_DONE
