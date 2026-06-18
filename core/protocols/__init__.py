"""NV measurement protocols: CW-ODMR, pulsed ODMR, Ramsey and spin echo.

Ported from the masters notebooks (`NV_sims_qutip_proto.ipynb`), with the terse names made
descriptive. The model functions (`dynamics_no`, `dynamics_doh_hf`, `dynamics_dua_hf`) are
called with the keyword API they expect: ``model(duration, state, b=, om_r=, om=, w_p=, ti=,
mode=, progress_bar=)`` — those kwargs are kept as-is.

UNITS (consistent throughout, matching core/constants.toml):
  - frequencies (MW drive ``om``, Rabi ``om_r``, pump ``w_p``, ZFS, hyperfine): MHz
  - durations and sweep times (``duration``, ``t_prepare``, ``t_wait``, ``t_readout``,
    ``free_times``, ``echo_times``, ``readtime``): microseconds (us)
  - magnetic field magnitude (``field``): Gauss (G); orientation angles: radians (via B0)
Pass values on this scale — e.g. a 1 us wait is ``1.0``, not ``1e-6``.

Split into ``_defaults`` + ``odmr`` + ``coherence`` + ``readout``; the full public API is
re-exported here so ``from core import protocols as P`` keeps working unchanged.
"""

# Constants re-exported for back-compat (were importable from core.protocols before the split).
from ..models import OM_R, OMEGA, PHI, W_P, A_GS, B, B0, D_GS, MU_E  # noqa: F401
from ._defaults import (
    default_drive,
    default_phase,
    default_pump_rate,
    default_rabi,
    odmr_field,
    odmr_pump_factor,
    odmr_pump_rate,
    odmr_rabi,
    ramsey_pump_factor,
    ramsey_pump_rate,
)
from .coherence import ramsey, ramsey_sweep, spin_echo, spin_echo_sweep
from .odmr import (
    cw_odmr,
    cw_odmr_serial,
    odmr_frequency_sweep,
    pulsed_odmr,
    pulsed_odmr_serial,
)
from .readout import readout_contrast

__all__ = [
    "cw_odmr",
    "cw_odmr_serial",
    "default_drive",
    "default_phase",
    "default_pump_rate",
    "default_rabi",
    "odmr_field",
    "odmr_frequency_sweep",
    "odmr_pump_factor",
    "odmr_pump_rate",
    "odmr_rabi",
    "pulsed_odmr",
    "pulsed_odmr_serial",
    "ramsey",
    "ramsey_pump_factor",
    "ramsey_pump_rate",
    "ramsey_sweep",
    "readout_contrast",
    "spin_echo",
    "spin_echo_sweep",
]
