#!/bin/bash
#SBATCH --job-name=diag_env
#SBATCH --partition=gpu_v100s
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-v100s-06
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/slurm_diag_env_%j.out

# Staged env diagnostic (race3 mid-load deaths). -X faulthandler: a native
# crash dumps the Python stack at the faulting call instead of dying silently.
set -euxo pipefail
cd "${MOB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY=${PYTHON:-python}
$PY -X faulthandler -m tools.diag_env
echo DIAG_DONE
