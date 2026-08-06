#!/bin/bash
# One-time environment setup on MSI Agate.
#
# Run this ONCE from a login node (or an interactive GPU session) to build the
# conda env the SLURM jobs activate. It is not a SLURM script.
#
#   bash env/setup_agate.sh
#
# VERIFY module names against your cluster: `module avail cuda`, `module avail conda`.
# Names below are the common MSI Lmod names as of writing; MSI updates them.
set -euo pipefail

module purge
module load conda            # MSI provides a system conda; else `module load python`
module load cuda/12.1.1      # Agate 12.x (from `module avail cuda`)

ENV_NAME="${ENV_NAME:-tgv}"
ENV_DIR="${HOME}/.conda/envs/${ENV_NAME}"

if [ ! -d "${ENV_DIR}" ]; then
    conda create -y -n "${ENV_NAME}" python=3.11
fi
source activate "${ENV_NAME}"

# cupy-cuda12x matches CUDA 12.x. nvmath-python[cu12] is only needed for the
# --backend nvmath path; the default cupy backend does not require it.
pip install --upgrade pip
pip install "cupy-cuda12x" numpy matplotlib
pip install "nvmath-python[cu12]"     # optional: for --backend nvmath

echo
echo "env '${ENV_NAME}' ready. Quick GPU check (needs a GPU node):"
echo "  python -c 'import cupy; print(cupy.cuda.runtime.getDeviceProperties(0)[\"name\"])'"
