"""Models 4 & 5: the NV as a controlled open system (laser/phonon bath as a resource).

Model 4 brings the open-system *control* framework of Kallush, Dann & Kosloff
(Sci. Adv. 8, eadd0828, 2022) to the NV center. Instead of treating dissipation as a
fixed nuisance, the bath becomes a control resource: the drive deforms H_S(t), and with
it the jump operators and their (detailed-balance) rates -- control-dependent
dissipation. The heavy lifting lives in core.open_control; here we build NV-flavored
ControlProblem instances and a thermal collapse-operator helper for the mesolve models.
"""
# ruff: noqa: F405

import numpy as np
import qutip as qt

from .. import open_control as _oc
from ._base import *  # noqa: F403


def thermal_cops(rate: float, gap: float, temperature: float, lowering, raising=None):
    """Detailed-balance thermal collapse operators for one transition.

    Models the "laser as a controlled heat bath": a downward jump ``lowering`` at rate
    ``rate`` and an upward jump at ``rate * exp(-gap / temperature)`` (Boltzmann factor).
    ``raising`` defaults to ``lowering.dag()``. ``rate`` is the control knob (e.g. laser
    intensity); ``temperature`` and ``gap`` are in MHz (``k_B = hbar = 1``).
    """
    if raising is None:
        raising = lowering.dag()
    up = 0.0 if temperature <= 0.0 else rate * np.exp(-gap / temperature)
    cops = [np.sqrt(rate) * lowering]
    if up > 0.0:
        cops.append(np.sqrt(up) * raising)
    return cops


def model4_ground_triplet(
    zfs: float = 1.0,
    zeeman: float = 0.25,
    temperature: float = 0.7,
    bath_coupling: float = 0.05,
    t_final: float = 8.0,
    n_times: int = 80,
    n_freqs: int = 4,
) -> "_oc.ControlProblem":
    """NV ground-state spin-1 triplet as a controlled open system (Model 4 demonstrator).

    Bare drift ``H0 = zfs * Sz^2 + zeeman * Sz`` (zero-field splitting + a small Zeeman
    term that lifts the ``m_s = +/-1`` degeneracy, so the three levels are non-degenerate).
    The microwave drive enters as the control ``V = Sx``; the bath couples through ``S = Sx``
    (transverse phonon noise). The dissipator is a **detailed-balance super-Ohmic bath**
    ``J(w) = c w^2`` (KMS) at ``temperature``, so the instantaneous Gibbs state of ``H_S(t)`` is the
    fixed point (thermodynamic consistency). Energies are in units of the ZFS (``zfs = 1``).

    Thermodynamic read-outs along a trajectory: entropy via ``core.open_control.vn_entropy``; the
    heat current is ``Q = Tr[H * D(rho)]``. Returns a :class:`core.open_control.ControlProblem`
    ready for ``simulate`` / ``optimize_crab`` (heating and cooling). QuTiP-native (``Qobj``).
    """
    jx = qt.jmat(1, "x")
    jz = qt.jmat(1, "z")
    h0 = zfs * jz**2 + zeeman * jz
    rho0 = _oc._thermal_state(h0, temperature)
    tlist = np.linspace(0.0, t_final, n_times)
    freqs = 2.0 * np.pi * np.arange(1, n_freqs + 1) / t_final
    return _oc.ControlProblem(
        h0=h0,
        control=jx,
        coupling=jx,
        rho0=rho0,
        temperature=temperature,
        bath_coupling=bath_coupling,
        tlist=tlist,
        freqs=freqs,
    )


def model5_full_nv(
    optical_gap: float = 3.0,
    zeeman: float = 0.2,
    laser_temp: float = 1.5,
    bath_coupling: float = 0.05,
    es_zfs_ratio: float | None = None,
    t_final: float = 10.0,
    n_times: int = 90,
    n_freqs: int = 4,
) -> "_oc.ControlProblem":
    """Full 7-level NV as a controlled open system (Model 5).

    Drift (units of `D_GS`): `Sz_gs^2 + (D_ES/D_GS) Sz_es^2 + optical_gap * P_es + zeeman Sz_gs`.
    Control = MW on the ground spin (`Sx_gs`). Bath = "laser" coupling the GS and ES
    manifolds, `S = sum_i |ES_i><GS_i| + h.c.`, at effective temperature `laser_temp`.

    NOTE (Medina reconciliation): this PoC abstracts the laser as an **effective thermal bath** so
    it fits the control-dependent-dissipation framework. Physically the laser is an incoherent
    **work** source (not heat at a fixed T) and the phonons are the thermal bath — that
    physically-faithful optical cycle is **Model 7** (`model7_full_optical_cycle`,
    [[nv-optical-pump-thermodynamics-medina]]). Read out the ground-spin entropy via
    `core.open_control.subspace_entropy(rho, GS_INDICES)`. QuTiP-native (``Qobj``).
    """
    sx_gs = SX_GS
    sz_gs = SZ_GS
    sz_es = SZ_ES
    p_es = N4 + N5 + N6
    ratio = (D_ES / D_GS) if es_zfs_ratio is None else es_zfs_ratio

    h0 = (
        sz_gs**2
        + ratio * sz_es**2
        + optical_gap * p_es
        + zeeman * sz_gs
    )
    s_opt = (
        EXCITED[0] * GROUND[0].dag()
        + EXCITED[1] * GROUND[1].dag()
        + EXCITED[2] * GROUND[2].dag()
    )
    coupling = s_opt + s_opt.dag()

    rho0 = _oc._thermal_state(h0, laser_temp)
    tlist = np.linspace(0.0, t_final, n_times)
    freqs = 2.0 * np.pi * np.arange(1, n_freqs + 1) / t_final
    return _oc.ControlProblem(
        h0=h0,
        control=sx_gs,
        coupling=coupling,
        rho0=rho0,
        temperature=laser_temp,
        bath_coupling=bath_coupling,
        tlist=tlist,
        freqs=freqs,
    )
