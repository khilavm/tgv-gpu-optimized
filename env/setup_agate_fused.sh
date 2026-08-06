#!/bin/bash
# Add the cuFFT LTO-IR callback (device-API) stack to the existing `tgv` env.
#
# Run ONCE, after env/setup_agate.sh, from a login node or interactive session:
#   bash env/setup_agate_fused.sh
#
# Callback fusion is Linux + NVIDIA only and needs the device-API extra
# `nvmath-python[cu12-dx]`, which pulls numba-cuda + nvjitlink + nvrtc wheels for
# LTO-IR compilation. These pip wheels supply their own CUDA components, so they
# can be newer than the loaded `cuda/12.1.1` module — that's the intended setup.
# The one-off `smoke_test.py --backends nvmath-fused` is the real test of whether
# the stack compiles on this node; if compile_prolog fails, see the note below.
set -euo pipefail

module purge
module load conda
module load cuda/12.1.1
source activate "${ENV_NAME:-tgv}"

pip install --upgrade "nvmath-python[cu12-dx]"

echo
echo "installed device-API stack. Now validate the callbacks on a GPU node:"
echo "  srun -p a100-4 --gres=gpu:a100:1 --time=00:20:00 --account=YOUR_ACCOUNT --pty bash"
echo "  module load conda cuda/12.1.1 && source activate tgv"
echo "  python smoke_test.py --backends nvmath nvmath-fused --N 48"
echo
echo "Expect the c2c-*-nvmath-fused rows to print 'fused-vs-plain <1e-3' and OK."
echo "(r2c-*-nvmath-fused rows correctly SKIP: fusion is C2C-only by design.)"
echo
echo "If compile_prolog raises about CUDA/LTO/compute_capability: the documented"
echo "cuFFT workaround is compile_prolog(..., compute_capability=\"50\") — tell me"
echo "the exact error and I'll patch tgv/solver.py."
