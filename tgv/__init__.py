"""Taylor-Green vortex pseudospectral DNS — MSI Agate workflow.

Config-driven solver extracted from the notebooks in ../1-ports, ../2-optimized,
and ../3-production so the optimization campaign can be measured under controlled
conditions (one GPU, one N, one lever changed at a time) on MSI's Agate cluster.

The physics is identical to 3-production/runs/nvmath_CuPy-NVIDIA_H200.ipynb:
Fourier pseudospectral, 2/3 dealiasing, Leray projection, RK2 (Heun),
dt = 0.01 * (64/N), Re = 1600, domain [0, 2*pi]^3.
"""

from .solver import Solver, SolverConfig

__all__ = ["Solver", "SolverConfig"]
