# MSI Agate workflow — controlled TGV benchmark + physics campaign

This directory turns the scattered notebook results (`../1-ports`, `../2-optimized`,
`../3-production`) into a **controlled** measurement suitable for a performance/software
paper. The notebooks confound hardware with algorithm (the headline "3×" mixes H200+R2C
against Blackwell+C2C) and assert the fp32 accuracy cost without isolating it. This
workflow fixes both by running every lever on **one GPU, at matched N, one lever at a time**.

It targets **MSI Agate** (A100 GPUs, SLURM, Lmod), which is free with your UMN affiliation.
If Agate doesn't work out, the same scripts run on NSF ACCESS (Delta/DeltaAI) with only the
module names and partition changed.

## What this answers that the notebooks don't

1. **Deconfounded lever speedups.** `benchmark.py` times `{c2c,r2c} × {fp32,fp64} × N`
   on the same card. The fp32-vs-fp64 number on A100 (FP64 at 1:2) should land near the
   honest **~2×**, not the workstation-Blackwell **3.13×** (that card throttles FP64 to
   ~1/64, inflating the ratio). Reporting both, with the reason, is a paper result.
2. **fp32 attribution.** `physics.py` runs fp32 *and* fp64 to the dissipation peak at
   matched N. If ε_max barely moves between them, the notebooks' "~2% is fp32" claim is
   wrong and it's resolution/temporal error — a finding either way.
3. **Resolution adequacy.** Every physics run reports `k_max·η` at the peak — the standard
   "is it actually resolved" metric the notebooks never compute.
4. **Achieved bandwidth.** The bytes-moved model (`tgv.solver.bytes_per_step`) plus
   measured ms/step gives a predicted-vs-achieved fraction of peak HBM. Confirm it with
   `ncu` (below) for the real number.

## Layout

```
4-msi-agate/
├── tgv/solver.py        core solver: one class, levers = transform/precision/backend/N
├── smoke_test.py        seconds-long wiring check — RUN FIRST on a GPU node
├── benchmark.py         lever-matrix timing -> results/bench*.json
├── physics.py           integrate to the eps peak -> results/phys*.json (+figs)
├── analyze.py           JSON -> tables + roofline (runs anywhere, no GPU)
├── env/setup_agate.sh   one-time conda env build (run on Agate, not a job)
├── slurm/bench_a100.sbatch
├── slurm/physics_a100.sbatch
├── requirements.txt
└── results/  logs/      created on first run
```

## Quickstart on Agate

```bash
# 0. one-time: build the env (login node or interactive session)
bash env/setup_agate.sh          # VERIFY module names inside first (see below)

# 1. grab an interactive GPU for a 30-second sanity check
srun -p a100-4 --gres=gpu:a100:1 --time=00:15:00 --pty bash
module load conda cuda/12.4 && source activate tgv
python smoke_test.py             # must print ALL PASS before you queue anything
exit

# 2. submit the batch jobs (edit --account and VERIFY partition first)
sbatch slurm/bench_a100.sbatch
sbatch slurm/physics_a100.sbatch

# 3. pull results/*.json back to your laptop and analyze (no GPU needed)
python analyze.py bench   results/bench_a100_N128_256.json --roofline roofline.png
python analyze.py physics results/phys_*.json
```

## ⚠️ Things you MUST verify before submitting (I can't check MSI from here)

These are the spots most likely to differ from what I guessed. Grep the scripts for
`VERIFY`:

- **Partition name** — `#SBATCH --partition=a100-4`. Run `sinfo -s` on Agate; the A100
  partition may be named differently. MSI docs: the Agate "Partitions" page.
- **Account/allocation** — uncomment `#SBATCH --account=YOUR_GROUP` with your MSI group.
  Jobs won't queue without a valid allocation.
- **CUDA module version** — `module load cuda/12.4`. Run `module avail cuda` and pick a
  12.x that exists; keep `setup_agate.sh` and the sbatch files in sync (cupy-cuda12x
  needs a 12.x runtime).
- **A100 memory / bandwidth** — check `nvidia-smi` in the job log. If you land on a
  40GB A100 (PCIe, ~1.55 TB/s) instead of 80GB SXM (~2.0 TB/s), set `GPU_BW` accordingly:
  `sbatch --export=ALL,GPU_BW=1.55e12 slurm/bench_a100.sbatch`. N=512 fp64 will OOM on
  40GB — the scripts already keep 512 to fp32 for that reason.

## Compute budget (why this is cheap)

Timing needs ~200 steps, not a full run — the whole lever matrix is **minutes**. Physics
runs to the peak (~t=10, not t=20) are ~1 min (N=128) to ~20 min (N=512 fp32). Total is
well under the earlier ~6 GPU-hour estimate, and on Agate it costs an allocation, not money.

## Getting the real bandwidth number (for the roofline)

The `%BW` column from `analyze.py` uses the analytic traffic model. For the number a
reviewer will actually trust, profile one config with Nsight Compute:

```bash
ncu --set full --section MemoryWorkloadAnalysis \
    --launch-count 40 --launch-skip 200 \
    python benchmark.py --single r2c fp32 cupy 256 --n-time 60 --reps 1 \
    -o ncu_r2c_fp32_N256
```

Read "DRAM Throughput" per kernel against the card's peak. Expect the elementwise kernels
pinned near peak (memory bound) and the FFTs somewhat below — that's the roofline story.

## The fusion lever (`backend="nvmath-fused"`)

Fusion (cuFFT LTO-IR callbacks folding the dealias mask, the `ik` derivative factors, and
`1/N³` into the FFT load/store) is implemented as a fourth backend, ported from
`../2-optimized`. It is **C2C only** by design — the notebooks deliberately leave R2C
un-fused (R2C's ~2× dominates fusion's ~1.2×, and the C2R callback plumbing is
unvalidated), so `SolverConfig(transform="r2c", backend="nvmath-fused")` raises rather than
producing an unchecked result.

**Requirements:** the device-API extra `nvmath-python[cu12-dx]` (numba-cuda + LTO), Linux +
NVIDIA only. The pip wheels supply their own CUDA components, so they can be newer than the
loaded `cuda/12.1.1` module.

```bash
# 1. one-time: add the callback stack to the tgv env
bash env/setup_agate_fused.sh

# 2. validate the callbacks on a GPU node BEFORE the timed job — this is the gate
srun -p a100-4 --gres=gpu:a100:1 --time=00:20:00 --account=YOUR_ACCOUNT --pty bash
module load conda cuda/12.1.1 && source activate tgv
python smoke_test.py --backends nvmath nvmath-fused --N 48
#   c2c-*-nvmath-fused rows must print 'fused-vs-plain <1e-3' and OK
#   r2c-*-nvmath-fused rows correctly SKIP (C2C-only by design)
exit

# 3. the measurement: plain vs fused on the same card
sbatch slurm/fusion_a100.sbatch
python analyze.py bench results/bench_fusion.json   # 'fusion (...)' rows = the speedup
```

The correctness gate (`Solver.validate_fused`, run automatically by `smoke_test.py`) diffs
one fused RHS against the plain-CuPy RHS on a random state — this proves the callback wiring
(masks, `ik`, `1/N³`) before any timing is trusted. The interesting paper result is not the
~1.22× itself but the **ceiling analysis**: cuFFT exposes one prolog + one epilog per
transform, so the Leray-projection and Heun-combination passes cannot be fused; with the
elementwise passes at ~40% of per-step traffic, the fusable fraction caps the speedup near
1.75×, and the measured 1.22× locates the remaining headroom in the un-fusable projection
arithmetic.

**If `compile_prolog` fails on Agate** (CUDA/LTO/compute-capability error): the documented
cuFFT workaround is `compile_prolog(..., compute_capability="50")`. Send me the exact error
and I'll patch `tgv/solver._build_transforms_fused`.

## Method (matches the production notebooks exactly)

Fourier pseudospectral · 2/3 dealiasing · Leray projection · RK2 (Heun) ·
`dt = 0.01·(64/N)` (advective CFL) · Re = 1600 · domain `[0, 2π]³`. The r2c path carries
Hermitian Parseval weights so half-spectrum diagnostic sums equal the full-spectrum ones
(checked by `E0 = 1/8` in `validate()`).
