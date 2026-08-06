#!/usr/bin/env python3
"""Physics run: integrate to the dissipation peak and record eps_max(t).

This is the run that settles the fp32-attribution question. The notebooks assert
"eps_max = 1.286e-2 is ~2% high because of fp32", but never isolate fp32 from
resolution or temporal error. Run this at matched N for fp32 AND fp64, and at
N in {128,256,512}, and the source of the 2% becomes a measurement instead of a claim.

Reference: Brachet et al. 1983 give eps_max ~ 1.26e-2 near t=9 at Re=1600.

Usage
-----
    # fp32 at N=256 to the peak
    python physics.py --N 256 --precision fp32 --transform r2c --T 12 \
        --out results/phys_r2c_fp32_N256.json

    # the fp64 companion (the comparison the notebooks never made)
    python physics.py --N 256 --precision fp64 --transform r2c --T 12 \
        --out results/phys_r2c_fp64_N256.json

Records the full E(t), eps(t) history plus the peak (eps_max, t_peak) and the
k_max*eta resolution metric at the peak. Optionally writes vorticity/spectrum figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tgv.solver import Solver, SolverConfig  # noqa: E402

import cupy as cp  # noqa: E402


def resolution_metric(solver: Solver, eps: float) -> dict:
    """k_max * eta at the current dissipation rate — the standard adequacy check.

    eta = (nu^3 / eps)^(1/4) is the Kolmogorov length; k_max = N/3 after dealias.
    A well-resolved spectral DNS wants k_max*eta >= ~1.  This is the number a
    reviewer asks for before believing "well-resolved".
    """
    nu = solver.nu
    eta = (nu ** 3 / eps) ** 0.25 if eps > 0 else float("inf")
    k_max = solver.N // 3
    return {"eta": eta, "k_max": k_max, "kmax_eta": k_max * eta}


def energy_spectrum(solver: Solver, state) -> dict:
    """Shell-averaged E(k) at the given state."""
    uh, vh, wh = state
    norm = float(solver.N) ** 6
    amp2 = cp.abs(uh) ** 2 + cp.abs(vh) ** 2 + cp.abs(wh) ** 2
    if solver.herm is not None:
        amp2 = solver.herm * amp2
    energy_k = 0.5 * amp2 / norm
    kmax = solver.N // 2
    shells = cp.asnumpy(cp.clip(cp.round(cp.sqrt(solver.K2)).astype(cp.int32).ravel(), 0, kmax))
    vals = cp.asnumpy(energy_k.ravel())
    spec = np.zeros(kmax + 1)
    np.add.at(spec, shells, vals)
    return {"k": list(range(kmax + 1)), "E_k": spec.tolist()}


def run(cfg: SolverConfig, make_figures: bool, out_path: str) -> dict:
    solver = Solver(cfg)
    val = solver.validate()
    print(f"# {cfg.label()}: E0_err={val['E0_err']:.2e}  "
          f"roundtrip={val['roundtrip_err']:.2e}")

    state = solver.state0
    times, energies, dissipations = [0.0], [], []
    E0, eps0 = solver.diagnostics(state)
    energies.append(E0)
    dissipations.append(eps0)
    peak = {"eps": -1.0, "t": 0.0, "state": None}

    # warm up plans before timing
    _ = solver.step(state)
    cp.cuda.Stream.null.synchronize()

    t_wall = time.perf_counter()
    for step in range(1, cfg.n_steps + 1):
        state = solver.step(state)
        if step % cfg.log_freq == 0:
            t = step * cfg.dt
            E, eps = solver.diagnostics(state)
            times.append(t)
            energies.append(E)
            dissipations.append(eps)
            if eps > peak["eps"]:
                peak = {"eps": eps, "t": t, "state": tuple(s.copy() for s in state)}
            if step % (cfg.log_freq * 10) == 0:
                print(f"  t={t:5.2f}  E={E:.5f}  eps={eps:.4e}")
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t_wall

    peak_state = peak["state"] if peak["state"] is not None else state
    res = resolution_metric(solver, peak["eps"])
    spec = energy_spectrum(solver, peak_state)

    rec = {
        "config": cfg.label(),
        "transform": cfg.transform, "precision": cfg.precision,
        "backend": cfg.backend, "N": cfg.N, "Re": cfg.Re, "dt": cfg.dt,
        "T": cfg.T, "n_steps": cfg.n_steps,
        "validation": val,
        "eps_max": peak["eps"], "t_peak": peak["t"],
        "eps_max_reference_brachet1983": 1.26e-2,
        "eps_max_rel_error": abs(peak["eps"] - 1.26e-2) / 1.26e-2,
        "resolution_at_peak": res,
        "wall_time_s": elapsed,
        "ms_per_step": 1e3 * elapsed / cfg.n_steps,
        "history": {"t": times, "E": energies, "eps": dissipations},
        "spectrum_at_peak": spec,
    }

    print(f"  eps_max = {peak['eps']:.4e} at t={peak['t']:.2f}  "
          f"(rel err vs Brachet {rec['eps_max_rel_error'] * 100:.1f}%)")
    print(f"  k_max*eta at peak = {res['kmax_eta']:.2f}  "
          f"({'resolved' if res['kmax_eta'] >= 1 else 'UNDER-RESOLVED'})")

    if make_figures:
        _figures(solver, peak_state, peak["t"], cfg, rec, out_path)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
                   "record": rec}, f, indent=2)
    print(f"wrote {out_path}")
    return rec


def _figures(solver, peak_state, t_peak, cfg, rec, out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = os.path.splitext(out_path)[0]

    # E(t), eps(t)
    h = rec["history"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["t"], h["E"]); ax[0].set(title="Kinetic energy E(t)", xlabel="t")
    ax[1].plot(h["t"], h["eps"], color="C1")
    ax[1].axhline(1.26e-2, ls="--", lw=0.8, color="gray", label="Brachet 1983")
    ax[1].set(title="Dissipation eps(t)", xlabel="t"); ax[1].legend()
    fig.suptitle(cfg.label()); fig.tight_layout()
    fig.savefig(f"{base}_diag.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # omega_z at z=0, peak
    uh, vh, wh = peak_state
    dl = solver.dealias
    omega_z = cp.asnumpy(solver._inv(1j * solver.KX * vh * dl)
                         - solver._inv(1j * solver.KY * uh * dl))
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    vmax = np.percentile(np.abs(omega_z), 99.5)
    ax.imshow(omega_z[:, :, 0], origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
              cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.set(title=f"omega_z(z=0) t={t_peak:.1f}  {cfg.label()}")
    fig.tight_layout(); fig.savefig(f"{base}_wz.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # |omega| max-projection along z, peak — the 3-D structure the z=0 symmetry
    # plane cannot show. Collapses every vortex sheet in the volume into one image;
    # this is the turbulence "money shot" (cf. tgv_wmag_N512.png in the notebooks).
    def d(f, k):
        return solver._inv(1j * k * f * dl)          # d/d(dir) of a spectral field
    KX, KY, KZ = solver.KX, solver.KY, solver.KZ
    wx = d(wh, KY) - d(vh, KZ)
    wy = d(uh, KZ) - d(wh, KX)
    wz = d(vh, KX) - d(uh, KY)
    wmag = cp.sqrt(wx ** 2 + wy ** 2 + wz ** 2)
    proj = cp.asnumpy(wmag.max(axis=2))
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    ax.imshow(proj, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi],
              cmap="inferno", vmax=np.percentile(proj, 99.5), interpolation="bilinear")
    ax.set(title=f"|omega| max-proj  t={t_peak:.1f}  {cfg.label()}")
    fig.tight_layout(); fig.savefig(f"{base}_wmag.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {base}_diag.png, {base}_wz.png, {base}_wmag.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--Re", type=float, default=1600.0)
    p.add_argument("--transform", default="r2c", choices=["c2c", "r2c"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "fp64"])
    p.add_argument("--backend", default="cupy", choices=["cupy", "nvmath"])
    p.add_argument("--T", type=float, default=12.0)
    p.add_argument("--dt", type=float, default=None,
                   help="override the CFL time step 0.01*(64/N); use e.g. half the "
                        "default to check whether the residual eps_max error is "
                        "temporal (RK2) rather than resolution or precision")
    p.add_argument("--figures", action="store_true", help="write diagnostic PNGs")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = SolverConfig(N=args.N, Re=args.Re, transform=args.transform,
                       precision=args.precision, backend=args.backend,
                       T=args.T, dt=args.dt)
    out = args.out or f"results/phys_{cfg.label()}.json"
    run(cfg, args.figures, out)


if __name__ == "__main__":
    main()
