#!/bin/bash
#SBATCH --job-name=tgv-phys
#SBATCH --partition=a100-4          # VERIFY: `sinfo -s`
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00             # N=128/256 fp32+fp64 to the peak; N=512 fp32
#SBATCH --output=logs/phys_%j.out
#SBATCH --error=logs/phys_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL
# #SBATCH --account=YOUR_ACCOUNT      # UNCOMMENT and set your MSI allocation/group
#
# The fp32-attribution experiment: run fp32 AND fp64 to the dissipation peak at
# matched N. This isolates precision error from resolution error — the thing the
# notebooks assert but never measure. Also gives the N-convergence of eps_max.
set -euo pipefail
mkdir -p logs results

module purge
module load conda
module load cuda/12.1.1             # matches env/setup_agate.sh
source activate "${ENV_NAME:-tgv}"
cd "${SLURM_SUBMIT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Only run to just past the peak (~t=10) to save time; the peak is near t=9.
TEND="${TEND:-10.5}"

for N in 128 256; do
  for PREC in fp32 fp64; do
    python physics.py --N "${N}" --precision "${PREC}" --transform r2c \
        --T "${TEND}" --out "results/phys_r2c_${PREC}_N${N}.json"
  done
done

# N=512: fp32 to the peak (with figures — this is the money shot). fp64 512 is
# heavy; run it as a separate longer job if you want the 512 precision point too.
python physics.py --N 512 --precision fp32 --transform r2c \
    --T "${TEND}" --figures --out "results/phys_r2c_fp32_N512.json"

echo "done. analyze locally with:"
echo "  python analyze.py physics results/phys_*.json"
