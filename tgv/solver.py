"""Config-driven Taylor-Green vortex pseudospectral solver.

One class, four independent levers:

    transform : "c2c" | "r2c"        (full complex vs Hermitian half-spectrum)
    precision : "fp32" | "fp64"      (complex64/float32 vs complex128/float64)
    backend   : "cupy" | "nvmath"    (cupy.fft plans vs reused nvmath-python plans)
    N         : grid points per dim

Everything else (Re, dt-CFL rule, dealias, Leray projection, RK2/Heun, diagnostics)
matches the production notebooks so results are directly comparable.

The point of this module is *controlled* measurement: change exactly one lever,
hold the GPU and N fixed, and the timing / accuracy difference is attributable to
that lever alone — unlike the notebooks, where H200+R2C vs Blackwell+C2C confounds
hardware with algorithm.

Fusion (cuFFT LTO-IR callbacks) is the fourth backend, `nvmath-fused` (C2C only),
implemented in _build_transforms_fused / _rhs_fused and gated by validate_fused().
See README.md -> "The fusion lever".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    import cupy as cp
except ImportError as exc:  # pragma: no cover - only meaningful on a GPU node
    raise ImportError(
        "cupy is required. On Agate: `module load cuda` then "
        "`pip install cupy-cuda12x` (see env/setup_agate.sh)."
    ) from exc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SolverConfig:
    N: int = 256
    Re: float = 1600.0
    transform: str = "r2c"          # "c2c" or "r2c"
    precision: str = "fp32"         # "fp32" or "fp64"
    backend: str = "cupy"           # "cupy" or "nvmath"
    T: float = 12.0                 # end time for a physics run
    dt: Optional[float] = None      # None -> CFL rule 0.01*(64/N)
    log_freq: int = 50

    def __post_init__(self) -> None:
        if self.transform not in ("c2c", "r2c"):
            raise ValueError(f"transform must be c2c|r2c, got {self.transform!r}")
        if self.precision not in ("fp32", "fp64"):
            raise ValueError(f"precision must be fp32|fp64, got {self.precision!r}")
        if self.backend not in ("cupy", "nvmath", "nvmath-fused"):
            raise ValueError(
                f"backend must be cupy|nvmath|nvmath-fused, got {self.backend!r}")
        if self.backend == "nvmath-fused" and self.transform != "c2c":
            raise ValueError(
                "nvmath-fused is C2C only: the notebooks deliberately leave R2C "
                "un-fused (R2C's ~2x dominates fusion's ~1.2x, and the C2R callback "
                "dtype plumbing is unvalidated). Use backend='cupy'/'nvmath' for r2c.")
        if self.dt is None:
            self.dt = 0.01 * (64.0 / self.N)   # advective CFL, calibrated at N=64

    @property
    def nu(self) -> float:
        return 1.0 / self.Re

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))

    def label(self) -> str:
        return f"{self.transform}-{self.precision}-{self.backend}-N{self.N}"


# --------------------------------------------------------------------------- #
# LTO-IR FFT callbacks (module level so numba-cuda can compile them).
#
# prolog: scale each spectral coefficient before an inverse transform.
# epilog: scale each coefficient produced by a forward transform.
# For a single (non-batched) (N,N,N) transform the linear `offset` never reaches
# N**3, so `factor[offset]` needs no modulo — the batched `offset % NNN` form in
# the notebooks is only required for the Re-sweep, which this module doesn't do.
# --------------------------------------------------------------------------- #
def _load_scaled(data_in, offset, factor, unused):
    return data_in[offset] * factor[offset]


def _store_scaled(data_out, offset, data, factor, unused):
    data_out[offset] = data * factor[offset]


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class Solver:
    """Single-Re TGV pseudospectral solver, configured by SolverConfig."""

    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg
        self.N = cfg.N
        self.nu = cfg.nu
        self.dt = cfg.dt

        if cfg.precision == "fp32":
            self.REAL, self.COMPLEX = cp.float32, cp.complex64
        else:
            self.REAL, self.COMPLEX = cp.float64, cp.complex128

        self._fused = False              # set True by the nvmath-fused branch
        self._build_grid()
        self._build_transforms()
        self._build_initial_condition()

    # ---- grid, wavenumbers, dealias, Hermitian weights -------------------- #
    def _build_grid(self) -> None:
        N = self.N
        L = 2.0 * np.pi
        x = cp.linspace(0.0, L, N, endpoint=False).astype(self.REAL)
        self.X, self.Y, self.Z = cp.meshgrid(x, x, x, indexing="ij")

        kf = cp.fft.fftfreq(N, d=1.0 / N).astype(self.REAL)  # 0,1,..,N/2-1,-N/2,..,-1
        kd = N // 3                                           # 2/3 dealias cutoff

        if self.cfg.transform == "c2c":
            KX, KY, KZ = cp.meshgrid(kf, kf, kf, indexing="ij")
            self.spec_shape = (N, N, N)
            self.herm = None                                 # every mode weight 1
        else:  # r2c
            kr = cp.fft.rfftfreq(N, d=1.0 / N).astype(self.REAL)  # 0,1,..,N/2
            M = N // 2 + 1
            KX, KY, KZ = cp.meshgrid(kf, kf, kr, indexing="ij")
            self.spec_shape = (N, N, M)
            # Parseval multiplicity for the stored half-spectrum: 1 on the k_z=0
            # plane and (even N) the Nyquist plane, else 2.
            herm = cp.full((N, N, M), 2.0, dtype=cp.float64)
            herm[:, :, 0] = 1.0
            if N % 2 == 0:
                herm[:, :, -1] = 1.0
            self.herm = herm

        self.KX, self.KY, self.KZ = KX, KY, KZ
        self.K2 = KX ** 2 + KY ** 2 + KZ ** 2
        self.K2s = cp.where(self.K2 == 0, 1.0, self.K2)  # safe divide for Leray
        self.dealias = (
            (cp.abs(KX) <= kd) & (cp.abs(KY) <= kd) & (cp.abs(KZ) <= kd)
        ).astype(self.REAL)
        self._inv_norm = 1.0 / (N ** 3)

    # ---- FFT wiring: cupy plans or reused nvmath plans -------------------- #
    def _build_transforms(self) -> None:
        N = self.N
        if self.cfg.backend == "cupy":
            if self.cfg.transform == "c2c":
                self._fwd = lambda a: cp.fft.fftn(a)
                self._inv = lambda a: cp.fft.ifftn(a).real
            else:
                self._fwd = lambda a: cp.fft.rfftn(a)
                self._inv = lambda a: cp.fft.irfftn(a, s=(N, N, N))
            return

        if self.cfg.backend == "nvmath-fused":
            self._build_transforms_fused()
            return

        # nvmath backend: create the plans once, rebind operands each call.
        import nvmath.fft as nvfft

        FWD = nvfft.FFTDirection.FORWARD
        INV = nvfft.FFTDirection.INVERSE
        M = N // 2 + 1

        if self.cfg.transform == "c2c":
            _p_fwd = nvfft.FFT(
                cp.empty((N, N, N), self.COMPLEX), axes=(0, 1, 2),
                options={"result_layout": "natural"},
            )
            _p_fwd.plan()
            _p_inv = nvfft.FFT(
                cp.empty((N, N, N), self.COMPLEX), axes=(0, 1, 2),
                options={"result_layout": "natural"},
            )
            _p_inv.plan()

            def _fwd(a):
                _p_fwd.reset_operand(cp.ascontiguousarray(a, dtype=self.COMPLEX))
                return _p_fwd.execute(direction=FWD)

            def _inv(a):
                _p_inv.reset_operand(cp.ascontiguousarray(a, dtype=self.COMPLEX))
                return (_p_inv.execute(direction=INV) * self._inv_norm).real
        else:  # r2c
            _p_r2c = nvfft.FFT(
                cp.empty((N, N, N), self.REAL), axes=(0, 1, 2),
                options={"fft_type": "R2C", "result_layout": "natural"},
            )
            _p_r2c.plan()
            _p_c2r = nvfft.FFT(
                cp.empty((N, N, M), self.COMPLEX), axes=(0, 1, 2),
                options={"fft_type": "C2R", "last_axis_parity": "even",
                         "result_layout": "natural"},
            )
            _p_c2r.plan()

            def _fwd(a):
                _p_r2c.reset_operand(cp.ascontiguousarray(a, dtype=self.REAL))
                return _p_r2c.execute(direction=FWD)

            def _inv(a):
                _p_c2r.reset_operand(cp.ascontiguousarray(a, dtype=self.COMPLEX))
                return _p_c2r.execute(direction=INV) * self._inv_norm

        self._fwd, self._inv = _fwd, _inv

    def _build_transforms_fused(self) -> None:
        """C2C callback-fused path (LTO-IR prolog/epilog via nvmath-python).

        Requires the device-API extra `nvmath-python[cu12-dx]` (numba-cuda + LTO)
        and a recent CUDA stack; Linux + NVIDIA only. Plain C2C plans handle the
        IC, diagnostics, and validation; the fused plans are used only inside the
        RHS. The dealias mask, the ik derivative factors, and the 1/N**3
        normalization are folded into the cuFFT load (inverse) / store (forward),
        eliminating ~15 elementwise CuPy kernels per step.
        """
        import nvmath
        import nvmath.fft as nvfft

        N, C = self.N, self.COMPLEX
        FWD = nvfft.FFTDirection.FORWARD
        INV = nvfft.FFTDirection.INVERSE
        NAT = {"result_layout": "natural"}

        # Plain C2C plans (no callbacks) — IC / diagnostics / validation reference.
        _pf = nvfft.FFT(cp.empty((N, N, N), C), axes=(0, 1, 2), options=NAT); _pf.plan()
        _pi = nvfft.FFT(cp.empty((N, N, N), C), axes=(0, 1, 2), options=NAT); _pi.plan()

        def _fwd(a):
            _pf.reset_operand(cp.ascontiguousarray(a, dtype=C))
            return _pf.execute(direction=FWD)

        def _inv(a):
            _pi.reset_operand(cp.ascontiguousarray(a, dtype=C))
            return (_pi.execute(direction=INV) * self._inv_norm).real

        self._fwd, self._inv = _fwd, _inv

        # Compile the two callbacks to LTO-IR (dtype must match the operand).
        cbd = "complex64" if self.cfg.precision == "fp32" else "complex128"
        with cp.cuda.Device():
            PROLOG = nvmath.fft.compile_prolog(_load_scaled, cbd, cbd)
            EPILOG = nvmath.fft.compile_epilog(_store_scaled, cbd, cbd)

        # Per-mode factor arrays (user_info). Must stay alive: plans hold their ptr.
        def _Cf(arr):
            return cp.ascontiguousarray(arr.astype(C))

        NNN = N ** 3
        self._factors = {
            "deal_fwd": _Cf(self.dealias),                        # fwd epilog: mask
            "deal_inv": _Cf(self.dealias / NNN),                  # inv prolog: mask + 1/N^3
            "ddx": _Cf(1j * self.KX * self.dealias / NNN),        # inv prolog: i*kx*mask + 1/N^3
            "ddy": _Cf(1j * self.KY * self.dealias / NNN),
            "ddz": _Cf(1j * self.KZ * self.dealias / NNN),
        }

        def _mk(prolog=None, epilog=None):
            p = nvfft.FFT(cp.empty((N, N, N), C), axes=(0, 1, 2), options=NAT)
            kw = {}
            if prolog is not None:
                kw["prolog"] = {"ltoir": PROLOG, "data": prolog.data.ptr}
            if epilog is not None:
                kw["epilog"] = {"ltoir": EPILOG, "data": epilog.data.ptr}
            p.plan(**kw)
            return p

        self._P_fwd_deal = _mk(epilog=self._factors["deal_fwd"])
        self._P_inv_deal = _mk(prolog=self._factors["deal_inv"])
        self._P_inv_ddx = _mk(prolog=self._factors["ddx"])
        self._P_inv_ddy = _mk(prolog=self._factors["ddy"])
        self._P_inv_ddz = _mk(prolog=self._factors["ddz"])
        self._FWD, self._INV = FWD, INV
        self._fused = True

    # ---- initial condition ------------------------------------------------ #
    def _build_initial_condition(self) -> None:
        u0 = cp.sin(self.X) * cp.cos(self.Y) * cp.cos(self.Z)
        v0 = -cp.cos(self.X) * cp.sin(self.Y) * cp.cos(self.Z)
        w0 = cp.zeros_like(self.X)
        self.state0 = (self._fwd(u0.astype(self.REAL)),
                       self._fwd(v0.astype(self.REAL)),
                       self._fwd(w0.astype(self.REAL)))
        self.KE0 = 0.5 * float(cp.mean(u0 ** 2 + v0 ** 2 + w0 ** 2))

    # ---- right-hand side -------------------------------------------------- #
    def rhs(self, state):
        return self._rhs_fused(state) if self._fused else self._rhs_generic(state)

    def _rhs_generic(self, state):
        uh, vh, wh = state
        dl = self.dealias
        uhd, vhd, whd = uh * dl, vh * dl, wh * dl

        u = self._inv(uhd); v = self._inv(vhd); w = self._inv(whd)

        KX, KY, KZ = self.KX, self.KY, self.KZ
        dudx = self._inv(1j * KX * uhd); dudy = self._inv(1j * KY * uhd); dudz = self._inv(1j * KZ * uhd)
        dvdx = self._inv(1j * KX * vhd); dvdy = self._inv(1j * KY * vhd); dvdz = self._inv(1j * KZ * vhd)
        dwdx = self._inv(1j * KX * whd); dwdy = self._inv(1j * KY * whd); dwdz = self._inv(1j * KZ * whd)

        Nuh = self._fwd(-(u * dudx + v * dudy + w * dudz)) * dl
        Nvh = self._fwd(-(u * dvdx + v * dvdy + w * dvdz)) * dl
        Nwh = self._fwd(-(u * dwdx + v * dwdy + w * dwdz)) * dl

        kdotN = KX * Nuh + KY * Nvh + KZ * Nwh
        Nuh = Nuh - kdotN / self.K2s * KX
        Nvh = Nvh - kdotN / self.K2s * KY
        Nwh = Nwh - kdotN / self.K2s * KZ

        nu, K2 = self.nu, self.K2
        return (Nuh - nu * K2 * uh, Nvh - nu * K2 * vh, Nwh - nu * K2 * wh)

    def _rhs_fused(self, state):
        """C2C RHS with the mask / ik / 1/N**3 folded into the cuFFT callbacks.

        The fused inverse plans already apply (mask, ik, 1/N**3) in a prolog, and
        the fused forward plan applies the mask in an epilog, so this routine does
        NO CuPy dealias / derivative multiplies — only the un-fusable spectral
        algebra (Leray projection + viscous term) remains. Safety copies decouple
        each result from the reused plan's output buffer so several outputs of one
        plan can be live at once (forced ON — the auto-probe mis-detects reuse on
        this nvmath build).
        """
        uh, vh, wh = state
        C, FWD, INV = self.COMPLEX, self._FWD, self._INV

        def inv(plan, a):
            plan.reset_operand(cp.ascontiguousarray(a, dtype=C))
            return cp.ascontiguousarray(plan.execute(direction=INV).real)

        def fwd(plan, a):
            plan.reset_operand(cp.ascontiguousarray(a, dtype=C))
            return plan.execute(direction=FWD).copy()

        Pd, Px, Py, Pz = (self._P_inv_deal, self._P_inv_ddx,
                          self._P_inv_ddy, self._P_inv_ddz)
        u = inv(Pd, uh); v = inv(Pd, vh); w = inv(Pd, wh)
        dudx = inv(Px, uh); dudy = inv(Py, uh); dudz = inv(Pz, uh)
        dvdx = inv(Px, vh); dvdy = inv(Py, vh); dvdz = inv(Pz, vh)
        dwdx = inv(Px, wh); dwdy = inv(Py, wh); dwdz = inv(Pz, wh)

        Pf = self._P_fwd_deal
        Nuh = fwd(Pf, -(u * dudx + v * dudy + w * dudz))
        Nvh = fwd(Pf, -(u * dvdx + v * dvdy + w * dvdz))
        Nwh = fwd(Pf, -(u * dwdx + v * dwdy + w * dwdz))

        KX, KY, KZ = self.KX, self.KY, self.KZ
        kdotN = KX * Nuh + KY * Nvh + KZ * Nwh
        Nuh = Nuh - kdotN / self.K2s * KX
        Nvh = Nvh - kdotN / self.K2s * KY
        Nwh = Nwh - kdotN / self.K2s * KZ

        nu, K2 = self.nu, self.K2
        return (Nuh - nu * K2 * uh, Nvh - nu * K2 * vh, Nwh - nu * K2 * wh)

    def step(self, state):
        """One RK2 (Heun) step."""
        dt = self.dt
        k1 = self.rhs(state)
        pred = tuple(s + dt * f for s, f in zip(state, k1))
        k2 = self.rhs(pred)
        return tuple(s + 0.5 * dt * (f1 + f2) for s, f1, f2 in zip(state, k1, k2))

    # ---- diagnostics ------------------------------------------------------ #
    def diagnostics(self, state):
        uh, vh, wh = state
        norm = float(self.N) ** 6
        amp2 = cp.abs(uh) ** 2 + cp.abs(vh) ** 2 + cp.abs(wh) ** 2
        if self.herm is not None:
            amp2 = self.herm * amp2
        E = 0.5 * float(cp.sum(amp2, dtype=cp.float64)) / norm
        eps = self.nu * float(cp.sum(self.K2 * amp2, dtype=cp.float64)) / norm
        return E, eps

    # ---- validation checks (cheap, run once) ------------------------------ #
    def validate(self) -> dict:
        """Round-trip error and initial-diagnostics check.

        A correct r2c normalization + Hermitian weighting must reproduce E0 = 1/8.
        This is both a physics check and a wiring check.
        """
        rng = cp.random.random((self.N, self.N, self.N)).astype(self.REAL)
        roundtrip = float(cp.max(cp.abs(self._inv(self._fwd(rng)) - rng)))
        E0, eps0 = self.diagnostics(self.state0)
        out = {
            "roundtrip_err": roundtrip,
            "E0": E0,
            "E0_exact": 0.125,
            "E0_err": abs(E0 - 0.125),
            "eps0": eps0,
            "KE0_physical": self.KE0,
        }
        if self._fused:
            out["fused_rhs_err"] = self.validate_fused()
        return out

    def validate_fused(self) -> float:
        """Max |fused RHS - plain RHS| on a random state — the fusion gate.

        Proves the callback wiring (mask, ik factors, 1/N**3) reproduces the plain
        CuPy path. fp32 agrees to ~1e-3 relative, fp64 to ~1e-9. Only meaningful
        for the nvmath-fused backend; the plain reference reuses this solver's
        callback-free plans, so it is self-contained.
        """
        if not self._fused:
            return 0.0
        s = tuple((cp.random.random((self.N, self.N, self.N))
                   + 1j * cp.random.random((self.N, self.N, self.N))).astype(self.COMPLEX)
                  for _ in range(3))
        rf, rp = self._rhs_fused(s), self._rhs_generic(s)
        return max(float(cp.max(cp.abs(rf[c] - rp[c]))) for c in range(3))


# --------------------------------------------------------------------------- #
# Bytes-moved model (for the roofline argument)
# --------------------------------------------------------------------------- #
def bytes_per_step(cfg: SolverConfig) -> dict:
    """Approximate DRAM traffic per RK2 step, for the arithmetic-intensity argument.

    This is a *back-of-envelope* model, not a measurement — use ncu (see README)
    for the real achieved-bandwidth number. It counts the dominant transfers:
    30 three-dimensional FFTs per step (12 inverse + 3 forward per RHS, two RHS),
    each moving the spectral array a small number of passes, plus the elementwise
    passes. Reported so the driver can print a predicted-vs-achieved column.
    """
    N = cfg.N
    bytes_real = 4 if cfg.precision == "fp32" else 8
    bytes_cplx = 2 * bytes_real

    if cfg.transform == "c2c":
        spec_elems = N ** 3
    else:
        spec_elems = N * N * (N // 2 + 1)
    spec_bytes = spec_elems * bytes_cplx
    phys_bytes = (N ** 3) * bytes_real

    n_fft = 30                       # transforms per RK2 step
    fft_passes = 3                   # ~read+intermediate+write per cuFFT 3D transform
    fft_traffic = n_fft * fft_passes * spec_bytes

    # elementwise: ~3 dealias + 9 ik-derivative + 3 post-dealias + ~7 projection,
    # each a read+write of the spectral array, times two RHS.
    n_ew = (3 + 9 + 3 + 7) * 2
    ew_traffic = n_ew * 2 * spec_bytes

    total = fft_traffic + ew_traffic
    return {
        "spec_bytes": spec_bytes,
        "phys_bytes": phys_bytes,
        "fft_traffic_GB": fft_traffic / 1e9,
        "elementwise_traffic_GB": ew_traffic / 1e9,
        "total_traffic_GB": total / 1e9,
        "elementwise_fraction": ew_traffic / total,
    }
