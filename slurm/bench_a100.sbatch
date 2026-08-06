#!/bin/bash
#SBATCH --job-name=tgv-bench
#SBATCH --partition=a100-4          # VERIFY: `sinfo -s` — Agate A100 partition name
#SBATCH --gres=gpu:a100:1           # one A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00             # whole lever matrix is minutes; 1h is slack
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL
# #SBATCH --account=YOUR_ACCOUNT      # UNCOMMENT and set your MSI allocation/group
#
# The lever matrix on one A100. A100 (SXM 80GB) peak HBM ~2.0 TB/s; the 40GB
# PCIe part is ~1.55 TB/s — set --gpu-bandwidth to match the card you land on
# (check `nvidia-smi` in the log). fp64 runs at 1:2 on A100, so the fp32-vs-fp64
# number here is the *honest* ~2x, not the workstation-Blackwell 3.13x.
set -euo pipefail
mkdir -p logs results

module purge
module load conda
module load cuda/12.1.1             # matches env/setup_agate.sh
source activate "${ENV_NAME:-tgv}"

cd "${SLURM_SUBMIT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# A100-80GB SXM: 2.0e12.  A100-40GB PCIe: 1.55e12.  Adjust to the card in the log.
BW="${GPU_BW:-1.55e12}"     # A100-SXM4-40GB (HBM2 ~1.55 TB/s)

# Full lever matrix at N=128,256 (both fit fp64). Add 512 only for fp32 (fp64 512
# is ~64 GB of state+gradients and may OOM on 40GB — run it separately if so).
python benchmark.py \
    --transforms r2c c2c \
    --precisions fp32 fp64 \
    --backends cupy \
    --grids 128 256 \
    --reps 5 --n-time 200 \
    --gpu-bandwidth "${BW}" \
    --out results/bench_a100_N128_256.json

# N=512 fp32 only (headline throughput point)
python benchmark.py \
    --transforms r2c c2c --precisions fp32 --backends cupy --grids 512 \
    --reps 5 --n-time 100 \
    --gpu-bandwidth "${BW}" \
    --out results/bench_a100_N512_fp32.json

echo "done. analyze locally with:"
echo "  python analyze.py bench results/bench_a100_N128_256.json --roofline roofline_a100.png"
