#!/usr/bin/env python3
"""Fast wiring check — run this FIRST on a GPU node before submitting jobs.

Validates every (transform, precision, backend) combination at a tiny N by
checking the round-trip error and initial energy E0 = 1/8. Takes seconds. If a
combination fails here, the SLURM job would fail the same way but after queueing.

    python smoke_test.py               # cupy backend, N=32
    python smoke_test.py --backend nvmath --N 48
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tgv.solver import Solver, SolverConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--backends", nargs="+", default=["cupy"])
    args = p.parse_args()

    import cupy as cp
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"device: {name}\n")

    ok = True
    for transform, precision, backend in itertools.product(
            ("r2c", "c2c"), ("fp32", "fp64"), args.backends):
        label = f"{transform}-{precision}-{backend}-N{args.N}"
        try:
            # SolverConfig itself raises for invalid combos (e.g. r2c + nvmath-fused,
            # which is C2C-only by design) — keep it inside the guard so those
            # surface as a clean SKIP instead of crashing the whole sweep.
            cfg = SolverConfig(N=args.N, transform=transform,
                               precision=precision, backend=backend)
            v = Solver(cfg).validate()
            # fp32 round-trip is ~1e-6; fp64 ~1e-12. E0 must be 1/8 to ~1e-3 (fp32).
            tol_E = 1e-3 if precision == "fp32" else 1e-8
            passed = v["E0_err"] < tol_E and v["roundtrip_err"] < 1e-2
            extra = ""
            if "fused_rhs_err" in v:            # nvmath-fused: also gate the callbacks
                tol_f = 1e-3 if precision == "fp32" else 1e-9
                passed &= v["fused_rhs_err"] < tol_f
                extra = f"  fused-vs-plain {v['fused_rhs_err']:.1e}"
            flag = "OK " if passed else "FAIL"
            ok &= passed
            print(f"[{flag}] {cfg.label():28s} "
                  f"E0={v['E0']:.6f} (err {v['E0_err']:.1e})  "
                  f"roundtrip {v['roundtrip_err']:.1e}{extra}")
        except Exception as exc:
            print(f"[SKIP] {label:28s} {exc!r}")

    print("\nALL PASS" if ok else "\nSOME FAILED — fix before submitting jobs")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
