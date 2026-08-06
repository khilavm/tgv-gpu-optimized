#!/bin/bash
#SBATCH --job-name=tgv-fusion
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fusion_%j.out
#SBATCH --error=logs/fusion_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --account=YOUR_ACCOUNT
#
# The fusion lever: C2C plain (nvmath) vs C2C callback-fused (nvmath-fused) on the
# SAME A100, at matched precision and N — so the ratio is attributable to the
# LTO-IR callbacks alone. Fusion is C2C-only by design (see solver.py), so this
# runs only the c2c transform. Both fp32 and fp64 at N=128,256 to show the gain
# growing with N; add 256->... if memory allows (full C2C fp64 at N=256 is ~48 GB
# GB/step traffic but state fits; N=512 c2c does NOT fit on 40 GB).
set -euo pipefail
mkdir -p logs results

module purge
module load conda
module load cuda/12.1.1
source activate "${ENV_NAME:-tgv}"
cd "${SLURM_SUBMIT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv

BW="${GPU_BW:-1.55e12}"      # A100-SXM4-40GB

python benchmark.py \
    --transforms c2c \
    --precisions fp32 fp64 \
    --backends nvmath nvmath-fused \
    --grids 128 256 \
    --reps 5 --n-time 200 \
    --gpu-bandwidth "${BW}" \
    --out results/bench_fusion.json

echo "done. analyze locally with:"
echo "  python analyze.py bench results/bench_fusion.json"
echo "The 'fusion (nvmath-fused vs nvmath)' rows are the paper's fusion speedup."
