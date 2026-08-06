# GPU-FFT levers for a spectral Taylor–Green DNS

A small, reproducible benchmark that **deconfounds** the performance levers of a Fourier
pseudospectral direct numerical simulation (DNS) of the Taylor–Green vortex (TGV) on a
single NVIDIA GPU. It isolates, one at a time and at matched problem size, the three
optimisations that are usually reported tangled together:

- **precision** — fp32 (`complex64`) vs fp64 (`complex128`);
- **transform** — real-to-complex (R2C, half spectrum) vs complex-to-complex (C2C);
- **kernel fusion** — folding the dealias mask, the `ik` spectral-derivative factors, and
  the `1/N³` normalisation into the cuFFT transform via LTO-IR **callbacks**
  (prolog/epilog) as opposed to separate elementwise kernels.

This repository is the artifact for the accompanying paper (in preparation):

> K. Majmudar, *A study of optimization levers in a GPU
pseudospectral Taylor–Green vortex solver*

## What is measured

1. **Deconfounded lever speedups** — `benchmark.py` times `{c2c, r2c} × {fp32, fp64} × N` on
   the same card. On a datacenter A100 (FP64 at 1:2) the fp32-vs-fp64 ratio lands near **~2×**, not the ~3× seen on consumer/workstation Blackwell parts that throttle
   FP64 to ~1/64.
2. **fp32 accuracy attribution** — `physics.py` integrates fp32 and fp64 to the
   dissipation peak at matched N.
3. **Resolution adequacy** — every physics run reports `k_max·η` at the peak, the standard
   spectral resolution metric.
4. **Achieved bandwidth** — an analytic bytes-moved model (`tgv.solver.bytes_per_step`) plus
   measured ms/step gives the fraction of peak HBM the solver reaches, framing the results
   on a roofline.
5. **Fusion ceiling** — cuFFT exposes one prolog + one epilog per transform, so the
   Leray-projection and Heun-combination passes cannot be fused. With the fusable
   elementwise passes at a bounded fraction of per-step traffic, the achievable fusion
   speedup is capped. The measured gain locates the remaining headroom in the un-fusable
   projection arithmetic.

## Method

Fourier pseudospectral · 2/3 dealiasing · Leray projection · RK2 (Heun) ·
`dt = 0.01·(64/N)` (advective CFL) · Re = 1600 · domain `[0, 2π]³`. 

## Layout

```
.
├── tgv/solver.py            core solver: one class. levers = transform/precision/backend/N
├── tgv/__init__.py
├── smoke_test.py            seconds-long wiring + correctness check — run first on a GPU
├── benchmark.py             lever-matrix timing            -> results/bench*.json
├── physics.py               integrate to the eps peak      -> results/phys*.json (+ figures)
├── analyze.py               JSON -> tables + roofline (runs anywhere, no GPU needed)
├── env/setup_agate.sh       one-time conda env build (CuPy + nvmath backends)
├── env/setup_agate_fused.sh adds the cuFFT LTO-IR callback (device-API) stack
├── slurm/bench_a100.sbatch
├── slurm/physics_a100.sbatch
├── slurm/fusion_a100.sbatch
├── requirements.txt
└── results/                 the JSON + figures behind the paper's tables
```

## Requirements

- Linux + NVIDIA GPU (an A100-class card reproduces the paper's numbers).
- Python 3.11, `cupy-cuda12x`, `numpy`, `matplotlib` — see `requirements.txt`.
- `nvmath-python[cu12]` for the `nvmath` backend; **`nvmath-python[cu12-dx]`** (numba-cuda +
  nvJitLink for LTO-IR) for the fused backend. Fusion requires cuFFT ≥ 11.3 / CUDA ≥ 12.6U2
  and is Linux + NVIDIA only.

`analyze.py` needs no GPU and runs on any machine, so the `results/*.json` can be
turned back into the paper's tables and roofline directly.

## Quickstart

The `env/` and `slurm/` scripts are written for a SLURM + Lmod cluster (developed on MSI
Agate). On a different cluster, edit the partition name, the CUDA module version, and the
account. Set `--gpu-bandwidth` to the peak HBM of the
card you use (A100-SXM4-40GB ≈ 1.55 TB/s; 80GB SXM ≈ 2.0 TB/s), checked via `nvidia-smi`.

```bash
# 0. one-time: build the env (login node or interactive session)
bash env/setup_agate.sh
bash env/setup_agate_fused.sh          # only if you want the fused backend

# 1. sanity + correctness check on a GPU node (must print ALL PASS)
python smoke_test.py

# 2. the three experiments (SLURM)
sbatch slurm/bench_a100.sbatch         # lever matrix
sbatch slurm/physics_a100.sbatch       # fp32-vs-fp64 to the dissipation peak
sbatch slurm/fusion_a100.sbatch        # plain vs callback-fused C2C

# 3. turn the JSON into tables + roofline (no GPU needed)
python analyze.py bench    results/bench_a100_N128_256.json --roofline roofline.png
python analyze.py physics  results/phys_*.json
python analyze.py bench    results/bench_fusion.json
```

To run a single configuration directly (no SLURM):

```bash
python benchmark.py --single r2c fp32 cupy 256 --n-time 60 --reps 3
python physics.py   --N 256 --precision fp32 --transform r2c --T 10.5 --out phys.json
```

## The fusion backend (`backend="nvmath-fused"`)

Fusion folds the dealias mask, the `ik` derivative factors, and `1/N³` into the cuFFT
load/store callbacks (compiled to LTO-IR), replacing the separate elementwise kernels. It is
**C2C only**: R2C's ~2× already exceeds fusion's gain and the C2R callback path is not
validated, so requesting `transform="r2c"` with this backend raises an error instead of
running an unverified path. Correctness is enforced by `Solver.validate_fused` (invoked by
`smoke_test.py`), which compares between a fused RHS against the plain-CuPy RHS on a random
state before any timing is trusted.

If `compile_prolog` fails with a CUDA/LTO/compute-capability error, the documented cuFFT
workaround is `compile_prolog(..., compute_capability="50")`.

## Reproducing the paper's numbers without a GPU

```bash
python analyze.py bench   results/bench_a100_N128_256.json results/bench_a100_N512_fp32.json
python analyze.py physics results/phys_r2c_fp32_N128.json results/phys_r2c_fp64_N128.json \
                          results/phys_r2c_fp32_N256.json results/phys_r2c_fp64_N256.json \
                          results/phys_r2c_fp32_N512.json
python analyze.py bench   results/bench_fusion.json
```

## Citing

If you use this code, please cite the paper above. A `CITATION.cff` and archival DOI
(Zenodo) will be added on release.

## License

MIT — see [LICENSE](LICENSE).
