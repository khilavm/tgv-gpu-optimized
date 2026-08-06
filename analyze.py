#!/usr/bin/env python3
"""Turn benchmark/physics JSON into the paper's tables and the roofline figure.

Runs anywhere — no GPU, no cupy. Iterate on this locally against JSON pulled back
from Agate.

Usage
-----
    python analyze.py bench results/bench.json                 # lever table + speedups
    python analyze.py bench results/bench.json --roofline roofline.png
    python analyze.py physics results/phys_*.json              # eps_max convergence table
"""

from __future__ import annotations

import argparse
import glob
import json
import sys


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Benchmark analysis
# --------------------------------------------------------------------------- #
def bench_table(data: dict) -> None:
    recs = [r for r in data["records"] if r.get("ok")]
    dev = data["meta"]["device"]["name"]
    print(f"\nDevice: {dev}\n")
    hdr = f"{'config':28s} {'ms/step':>10s} {'std':>7s} {'GB/step':>9s} {'%BW':>6s}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(recs, key=lambda x: (x["N"], x["transform"], x["precision"])):
        frac = r.get("achieved_bandwidth_fraction")
        frac_s = f"{100 * frac:5.1f}" if frac is not None else "  -  "
        gb = r["traffic_model"]["total_traffic_GB"]
        print(f"{r['config']:28s} {r['ms_per_step_mean']:10.2f} "
              f"{r['ms_per_step_std']:7.2f} {gb:9.2f} {frac_s:>6s}")

    _speedups(recs)


def _lookup(recs, transform, precision, backend, N):
    for r in recs:
        if (r["transform"] == transform and r["precision"] == precision
                and r["backend"] == backend and r["N"] == N):
            return r["ms_per_step_mean"]
    return None


def _speedups(recs) -> None:
    """Isolated lever speedups — each holds everything else fixed."""
    Ns = sorted({r["N"] for r in recs})
    backends = sorted({r["backend"] for r in recs})
    print("\nIsolated lever speedups (same GPU, same N, one lever changed):")

    for b in backends:
        for N in Ns:
            # fp32 vs fp64 at fixed transform
            for t in ("r2c", "c2c"):
                fp64 = _lookup(recs, t, "fp64", b, N)
                fp32 = _lookup(recs, t, "fp32", b, N)
                if fp64 and fp32:
                    print(f"  [{b} N={N} {t}] fp32 vs fp64: "
                          f"{fp64 / fp32:.2f}x  ({fp64:.1f} -> {fp32:.1f} ms)")
            # r2c vs c2c at fixed precision
            for pr in ("fp32", "fp64"):
                c2c = _lookup(recs, "c2c", pr, b, N)
                r2c = _lookup(recs, "r2c", pr, b, N)
                if c2c and r2c:
                    print(f"  [{b} N={N} {pr}] r2c vs c2c: "
                          f"{c2c / r2c:.2f}x  ({c2c:.1f} -> {r2c:.1f} ms)")

    # fusion: nvmath-fused vs nvmath (plain), C2C only, at fixed precision + N
    for N in Ns:
        for pr in ("fp32", "fp64"):
            plain = _lookup(recs, "c2c", pr, "nvmath", N)
            fused = _lookup(recs, "c2c", pr, "nvmath-fused", N)
            if plain and fused:
                print(f"  [c2c N={N} {pr}] fusion (nvmath-fused vs nvmath): "
                      f"{plain / fused:.2f}x  ({plain:.1f} -> {fused:.1f} ms)")


def roofline(data: dict, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [r for r in data["records"] if r.get("ok")]
    bw = data["meta"].get("gpu_bandwidth_bytes_per_s")
    if bw is None:
        print("no --gpu-bandwidth was recorded; roofline needs it. skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in recs:
        gb = r["traffic_model"]["total_traffic_GB"]
        ms = r["ms_per_step_mean"]
        achieved = gb / (ms / 1e3) / 1e3  # TB/s achieved on the traffic model
        # arithmetic intensity: use the model's flop-free proxy (traffic only);
        # plot achieved effective bandwidth vs N as the headline instead.
        ax.scatter(r["N"], achieved, label=f"{r['transform']}-{r['precision']}")
    ax.axhline(bw / 1e12, ls="--", color="k", lw=1, label="peak HBM")
    ax.set(xlabel="N", ylabel="effective TB/s (traffic model)",
           title=f"Achieved bandwidth vs peak — {data['meta']['device']['name']}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


# --------------------------------------------------------------------------- #
# Physics analysis
# --------------------------------------------------------------------------- #
def physics_table(paths: list[str]) -> None:
    recs = [_load(p)["record"] for p in paths]
    print(f"\n{'config':28s} {'eps_max':>10s} {'t_peak':>7s} "
          f"{'relerr%':>8s} {'kmax*eta':>9s} {'ms/step':>9s}")
    print("-" * 76)
    for r in sorted(recs, key=lambda x: (x["N"], x["precision"])):
        print(f"{r['config']:28s} {r['eps_max']:10.4e} {r['t_peak']:7.2f} "
              f"{r['eps_max_rel_error'] * 100:8.2f} "
              f"{r['resolution_at_peak']['kmax_eta']:9.2f} {r['ms_per_step']:9.1f}")

    # fp32-attribution: compare fp32 vs fp64 at matched N
    print("\nfp32 attribution (eps_max difference at matched N, transform):")
    for N in sorted({r["N"] for r in recs}):
        for t in sorted({r["transform"] for r in recs}):
            f32 = next((r for r in recs if r["N"] == N and r["precision"] == "fp32"
                        and r["transform"] == t), None)
            f64 = next((r for r in recs if r["N"] == N and r["precision"] == "fp64"
                        and r["transform"] == t), None)
            if f32 and f64:
                d = (f32["eps_max"] - f64["eps_max"]) / f64["eps_max"] * 100
                print(f"  N={N} {t}: fp32 {f32['eps_max']:.4e} vs "
                      f"fp64 {f64['eps_max']:.4e}  -> fp32 is {d:+.2f}% "
                      f"(this is the precision contribution, isolated from resolution)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bench", help="lever table + speedups from a bench JSON")
    pb.add_argument("json")
    pb.add_argument("--roofline", metavar="PNG", default=None)

    pp = sub.add_parser("physics", help="eps_max table from physics JSON(s)")
    pp.add_argument("json", nargs="+", help="one or more physics JSON files (globs ok)")

    args = p.parse_args()

    if args.cmd == "bench":
        data = _load(args.json)
        bench_table(data)
        if args.roofline:
            roofline(data, args.roofline)
    elif args.cmd == "physics":
        paths = []
        for g in args.json:
            paths.extend(sorted(glob.glob(g)) or [g])
        physics_table(paths)


if __name__ == "__main__":
    main()
