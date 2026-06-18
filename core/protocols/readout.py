"""Readout contrast: the bright/dark reference signals (masters ``contrast()``)."""

from __future__ import annotations

import numpy as np
import qutip as qt
import scipy as scp

from ..models import B0, D_GS, MU_E


def readout_contrast(model, init_state, pl_operator, field, pump_prepare, pump_readout, rabi,
                     readtime=3.0):
    """Reference signals ``s0`` (m_s=0, bright) and ``s1`` (m_s=±1, dark) for normalizing
    Ramsey/ODMR. Integrates the PL operator over a readout window (Simpson) as in the masters
    ``contrast()``. Returns ``(s0, s1, s0_integrated, s1_integrated)``.
    """
    drive = D_GS - MU_E * field
    window = int(5000 * readtime / 10)  # masters: wind = int(5000*readtime/10)

    _, res = model(10.0, init_state, b=B0(field, 0.0, 0.0), w_p=pump_prepare, om_r=rabi,
                   om=drive, ti=0.0, mode="Laser", progress_bar="OFF")
    state, t = res.states[-1], res.times[-1]
    _, res = model(1.0, state, b=B0(field, 0.0, 0.0), w_p=pump_prepare, om_r=rabi,
                   om=drive, ti=t, mode="Free", progress_bar="OFF")
    state, t = res.states[-1], res.times[-1]
    _, res = model(10.0, state, b=B0(field, 0.0, 0.0), w_p=pump_readout, om_r=rabi,
                   om=drive, ti=t, mode="Laser", progress_bar="OFF")
    s0 = np.sum(qt.expect(pl_operator, res.states[:window]))
    s0_int = scp.integrate.simpson(qt.expect(pl_operator, res.states[:window]), res.times[:window])

    # Apply a pi pulse (-> m_s=±1, dark) and read out again.
    _, res = model(np.pi / rabi, state, b=B0(field, 0.0, 0.0), w_p=pump_prepare, om_r=rabi,
                   om=drive, ti=t, mode="MW", progress_bar="OFF")
    state, t = res.states[-1], res.times[-1]
    _, res = model(10.0, state, b=B0(field, 0.0, 0.0), w_p=pump_readout, om_r=rabi,
                   om=drive, ti=t, mode="Laser", progress_bar="OFF")
    s1 = np.sum(qt.expect(pl_operator, res.states[:window]))
    s1_int = scp.integrate.simpson(qt.expect(pl_operator, res.states[:window]), res.times[:window])
    return s0, s1, s0_int, s1_int
