#!/usr/bin/env python3
"""Lever-matrix benchmark: ms/step for each (transform, precision, backend, N).

This is the paper's core table. It times ~200 steps (not a physics run) so the
whole matrix finishes in minutes. Every config runs on the *same* GPU, so the
timing difference between two rows is attributable to the one lever that changed
— the confound in the notebooks (H200+R2C vs Blackwell+C2C) is removed.

Usage
-----
    # Default matrix: {c2c,r2c} x {fp32,fp64} x N in {128,256,256}
    python benchmark.py --gpu-bandwidth 2.0e12 --out results/bench.json

    # One explicit config
    python benchmark.py --single r2c fp32 cupy 512 --gpu-bandwidth 4.8e12

    # Custom matrix
    python benchmark.py --transforms r2c c2c --precisions fp32 fp64 \
        --backends cupy --grids 128 256 512 --reps 5

Output is a JSON file the analysis script (analyze.py) turns into a table + roofline.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tgv.solver import Solver, SolverConfig, bytes_per_step  # noqa: E402

import cupy as cp  # noqa: E402


def device_info() -> dict:
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "name": props["name"].decode(),
        "total_mem_GB": total / 1024 ** 3,
        "free_mem_GB": free / 1024 ** 3,
        "cupy": cp.__version__,
    }


def time_config(cfg: SolverConfig, warmup: int, n_time: int, reps: int) -> dict:
    """Return timing + validation for one config, or an error record.

    The whole body is guarded: a config that OOMs (e.g. c2c fp32 at N=512 on a
    40 GB card — the full complex spectrum doesn't fit) is recorded as skipped so
    the rest of the matrix still runs and the JSON is still written. That an OOM
    happened is itself data: it documents where R2C's memory halving is *required*,
    not just faster.
    """
    solver = state = None
    try:
        solver = Solver(cfg)
        val = solver.validate()
        state = solver.state0

        # Warm up: forces plan creation / JIT / autotune out of the timed region.
        for _ in range(warmup):
            state = solver.step(state)
        cp.cuda.Stream.null.synchronize()

        per_step_ms = []
        for _ in range(reps):
            state = solver.state0  # reset so every rep sees identical work
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_time):
                state = solver.step(state)
            cp.cuda.Stream.null.synchronize()
            per_step_ms.append(1e3 * (time.perf_counter() - t0) / n_time)

        per_step_ms = np.array(per_step_ms)
        rec = {
            "config": cfg.label(),
            "ok": True,
            "transform": cfg.transform,
            "precision": cfg.precision,
            "backend": cfg.backend,
            "N": cfg.N,
            "Re": cfg.Re,
            "dt": cfg.dt,
            "warmup": warmup,
            "n_time": n_time,
            "reps": reps,
            "ms_per_step_mean": float(per_step_ms.mean()),
            "ms_per_step_std": float(per_step_ms.std()),
            "ms_per_step_min": float(per_step_ms.min()),
            "validation": val,
            "traffic_model": bytes_per_step(cfg),
        }
    except Exception as exc:  # OOM, nvmath missing, unsupported combo, etc.
        rec = {"config": cfg.label(), "ok": False, "error": repr(exc),
               "N": cfg.N, "transform": cfg.transform,
               "precision": cfg.precision, "backend": cfg.backend}
    finally:
        # Always free GPU memory between configs, success or failure, so one OOM
        # doesn't poison the pool for the configs that follow.
        del solver, state
        cp.get_default_memory_pool().free_all_blocks()
    return rec


def add_bandwidth(rec: dict, bandwidth: float | None) -> None:
    """Attach predicted step time and achieved-fraction if bandwidth is known."""
    if not rec.get("ok") or bandwidth is None:
        return
    total_bytes = rec["traffic_model"]["total_traffic_GB"] * 1e9
    ideal_ms = 1e3 * total_bytes / bandwidth
    rec["ideal_ms_per_step"] = ideal_ms
    rec["achieved_bandwidth_fraction"] = ideal_ms / rec["ms_per_step_mean"]


def build_matrix(args) -> list[SolverConfig]:
    if args.single:
        transform, precision, backend, N = args.single
        return [SolverConfig(N=int(N), transform=transform,
                             precision=precision, backend=backend)]
    cfgs = []
    for N in args.grids:
        for transform in args.transforms:
            for precision in args.precisions:
                for backend in args.backends:
                    cfgs.append(SolverConfig(
                        N=int(N), transform=transform,
                        precision=precision, backend=backend))
    return cfgs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transforms", nargs="+", default=["r2c", "c2c"])
    p.add_argument("--precisions", nargs="+", default=["fp32", "fp64"])
    p.add_argument("--backends", nargs="+", default=["cupy"])
    p.add_argument("--grids", nargs="+", default=["128", "256"])
    p.add_argument("--single", nargs=4, metavar=("TRANSFORM", "PRECISION", "BACKEND", "N"),
                   help="run one config instead of the matrix")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--n-time", type=int, default=200,
                   help="steps to time per rep (not a physics run)")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--gpu-bandwidth", type=float, default=None,
                   help="peak HBM bandwidth in bytes/s, e.g. 2.0e12 for A100-80GB, "
                        "4.8e12 for H200. If set, reports achieved fraction.")
    p.add_argument("--out", default="results/bench.json")
    args = p.parse_args()

    cfgs = build_matrix(args)
    dev = device_info()
    print(f"# device: {dev['name']}  ({dev['total_mem_GB']:.0f} GB, cupy {dev['cupy']})")
    print(f"# {len(cfgs)} configs, warmup={args.warmup}, n_time={args.n_time}, "
          f"reps={args.reps}\n")

    records = []
    for cfg in cfgs:
        rec = time_config(cfg, args.warmup, args.n_time, args.reps)
        add_bandwidth(rec, args.gpu_bandwidth)
        records.append(rec)
        if rec["ok"]:
            frac = rec.get("achieved_bandwidth_fraction")
            frac_s = f"  {100 * frac:4.1f}% BW" if frac is not None else ""
            print(f"{cfg.label():28s}  {rec['ms_per_step_mean']:8.2f} "
                  f"+/- {rec['ms_per_step_std']:.2f} ms/step"
                  f"  (E0_err {rec['validation']['E0_err']:.1e}){frac_s}")
        else:
            print(f"{cfg.label():28s}  SKIPPED: {rec['error']}")

    out = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "device": dev,
            "gpu_bandwidth_bytes_per_s": args.gpu_bandwidth,
            "argv": sys.argv,
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
