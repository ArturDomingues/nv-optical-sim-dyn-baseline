"""Centralized physical constants for the NV-center models.

Values live in ``constants.toml`` (readable, sectioned, commented) and are loaded
once here, then exposed as module-level names with the same identifiers used
throughout :mod:`core.models`. Derived quantities (dephasing/decoherence rates) are
computed from the loaded ``T1``/``T2`` times.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_PATH = Path(__file__).parent / "constants.toml"
with _PATH.open("rb") as _f:
    _C = tomllib.load(_f)

# ---- Hilbert space ---------------------------------------------------------
DIM: int = _C["hilbert"]["dim"]

# ---- Strain (MHz) ----------------------------------------------------------
E_GS: float = _C["strain"]["ground"]
E_ES: float = _C["strain"]["excited"]

# ---- Zero-field splitting (MHz) -------------------------------------------
D_GS: float = _C["zero_field_splitting"]["ground"]
D_ES: float = _C["zero_field_splitting"]["excited"]

# ---- Hyperfine (MHz) -------------------------------------------------------
# Models 1 & 3 (isotropic): (parallel, perpendicular)
A_GS: tuple[float, float] = tuple(_C["hyperfine"]["isotropic"]["a_ground"])
A_ES: tuple[float, float] = tuple(_C["hyperfine"]["isotropic"]["a_excited"])
# Model 2 (anisotropic)
A_PAR: tuple[float, float] = tuple(_C["hyperfine"]["anisotropic"]["a_parallel"])
A_PERP: tuple[float, float] = tuple(_C["hyperfine"]["anisotropic"]["a_perp"])
A_ANI: tuple[float, float] = tuple(_C["hyperfine"]["anisotropic"]["a_aniso"])
A_PERP_PRIME: tuple[float, float] = tuple(_C["hyperfine"]["anisotropic"]["a_perp_prime"])
PHI_H: float = _C["hyperfine"]["anisotropic"]["phi_h"]

# ---- Magnetic moments (MHz/G) ---------------------------------------------
MU_E: float = _C["magnetic_moment"]["electron"]
MU_N: float = _C["magnetic_moment"]["nitrogen15"]

# ---- Relaxation / coherence times (microseconds) --------------------------
T1_GS: float = _C["t1"]["ground"]
T1_ES: float = _C["t1"]["excited"]
T1_N: float = _C["t1"]["nitrogen15"]
T2_GS: float = _C["t2"]["ground"]
T2_ES: float = _C["t2"]["excited"]
T2_N: float = _C["t2"]["nitrogen15"]

# ---- Dephasing / decoherence rates (MHz) — derived ------------------------
GAMMA_GS: tuple[float, float] = (1.0 / T1_GS, 1.0 / T2_GS)
GAMMA_ES: tuple[float, float] = (1.0 / T1_ES, 1.0 / T2_ES)
GAMMA_N: tuple[float, float] = (1.0 / T1_N, 1.0 / T2_N)

# ---- Drive (MHz) -----------------------------------------------------------
W_P: float = _C["drive"]["pump_rate"]
OM_R: float = _C["drive"]["rabi"]
OMEGA_MG: float = _C["drive"]["mw_drive"]

# ---- Transition rates ------------------------------------------------------
K_IND: int = _C["rates"]["selected_index"]
K_S: list[list[float]] = [list(row) for row in _C["rates"]["sets"]]

# ---- Default static magnetic field (Gauss, rad) ---------------------------
B_MAGNITUDE: float = _C["magnetic_field"]["magnitude"]
B_PHI: float = _C["magnetic_field"]["phi"]
B_THETA: float = _C["magnetic_field"]["theta"]

__all__ = [
    "DIM",
    "E_GS", "E_ES",
    "D_GS", "D_ES",
    "A_GS", "A_ES", "A_PAR", "A_PERP", "A_ANI", "A_PERP_PRIME", "PHI_H",
    "MU_E", "MU_N",
    "T1_GS", "T1_ES", "T1_N", "T2_GS", "T2_ES", "T2_N",
    "GAMMA_GS", "GAMMA_ES", "GAMMA_N",
    "W_P", "OM_R", "OMEGA_MG",
    "K_IND", "K_S",
    "B_MAGNITUDE", "B_PHI", "B_THETA",
]
