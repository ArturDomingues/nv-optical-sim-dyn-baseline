"""Shared NV state vectors, spin operators and module-level constants for the model family.

Every per-model module (`magaletti`, `nv7`, `duarte`, `doherty`, `control`) pulls these in via
``from ._base import *``. Physical constants live in ``constants.toml`` and are re-exported here.
"""
# ruff: noqa: F403

import numpy as np
import qutip as qt

from ..constants import *  # noqa: F403  (re-export the centralized constants)
from ..constants import DIM, B_MAGNITUDE, B_PHI, B_THETA, D_GS, MU_E
from ..utils import B0, construct_spin_matrices

# States
# NV
EXCITED = (
    qt.basis(DIM, 4),
    qt.basis(DIM, 3),
    qt.basis(DIM, 5),
)  # |+1>_es, |0>_es, |-1>_es
ISC = qt.basis(DIM, 6)
GROUND = (
    qt.basis(DIM, 1),
    qt.basis(DIM, 0),
    qt.basis(DIM, 2),
)  # |+1>_gs, |0>_gs, |-1>_gs
# ^{15} N
NIT = qt.basis(2, 0), qt.basis(2, 1)

N1, N2, N3 = (
    GROUND[1] * GROUND[1].dag(),  # type: ignore
    GROUND[2] * GROUND[2].dag(),  # type: ignore
    GROUND[0] * GROUND[0].dag(),  # type: ignore
)
N4, N5, N6 = (
    EXCITED[1] * EXCITED[1].dag(),  # type: ignore
    EXCITED[2] * EXCITED[2].dag(),  # type: ignore
    EXCITED[0] * EXCITED[0].dag(),  # type: ignore
)
N7, NC = (
    ISC * ISC.dag(),  # type: ignore
    GROUND[2] * GROUND[1].dag(),  # type: ignore
)

# Operators
# MV
SX_GS, SY_GS, SZ_GS = construct_spin_matrices(GROUND)
SX_ES, SY_ES, SZ_ES = construct_spin_matrices(EXCITED)
SM_GS, SP_GS = SX_GS - 1j * SY_GS, SX_GS + 1j * SY_GS
SM_ES, SP_ES = SX_ES - 1j * SY_ES, SX_ES + 1j * SY_ES

S_GS = np.array([SX_GS, SY_GS, SZ_GS])
S_ES = np.array([SX_ES, SY_ES, SZ_ES])
ID_NV = qt.qeye(DIM)

# ^{15} N
SX_N, SY_N, SZ_N = qt.sigmax() * 0.5, qt.sigmay() * 0.5, qt.sigmaz() * 0.5
SM_N, SP_N = SX_N - 1j * SY_N, SX_N + 1j * SY_N

S_N = np.array([SX_N, SY_N, SZ_N])
ID_N15 = qt.qeye(2)

# Static Magnetic Field (G) — default orientation from constants.toml
B = B0(B_MAGNITUDE, B_PHI, B_THETA)  # (G)

# Driving frequency / phase (no-HF + hyperfine Models 2 & 3)
OMEGA = D_GS - MU_E * B[2]
PHI = 0.0

# Hilbert-space indices of the ground / excited spin triplets (for subspace read-out).
GS_INDICES = [0, 1, 2]   # |+1>_gs, |0>_gs, |-1>_gs occupy basis indices 1, 0, 2 -> {0,1,2}
ES_INDICES = [3, 4, 5]
