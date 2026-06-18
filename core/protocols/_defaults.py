"""Protocol defaults (descriptive names; original terse names in comments)."""

from __future__ import annotations

from ..models import OM_R, OMEGA, PHI, W_P

odmr_field = 150.0          # b_odmr  — static field for ODMR (G)
odmr_pump_factor = 1.0      # n_p_odmr
ramsey_pump_factor = 1.5    # n_p_ramsey
odmr_pump_rate = odmr_pump_factor * W_P     # w_p_odmr
ramsey_pump_rate = ramsey_pump_factor * W_P  # w_p_ramsey
odmr_rabi = 15.7            # om_r_odmr (MHz)

default_pump_rate = W_P     # W_p
default_rabi = OM_R         # Om_r
default_drive = OMEGA       # omega
default_phase = PHI         # phi
