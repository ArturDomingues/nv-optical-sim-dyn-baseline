"""Models 6 & 7: the NV optical cycle as QuTiP Lindbladians (Medina et al., PRA 113, 052228, 2026).

The laser is an incoherent **work** source (pump dissipator), not a heat bath; the decay channels
dissipate **heat**. We expose the full thermodynamic bookkeeping the paper derives.

- **Model 6** ``model6_optical_cycle`` — reduced 6-level cycle (GS triplet + ES triplet): laser
  pump ``L_{1..3}`` + radiative decay ``L_{4..6}`` (+ optional spin-nonconserving).
- **Model 7** ``model7_full_optical_cycle`` — full 8-level cycle (+ ISC singlet ``|7>,|8>``), all
  17 Medina jump operators, with optional **Ernst** temperature-dependent ISC rate.

Basis order: ``|0>,|1>,|2>`` ground triplet (m_s = +1,0,-1), ``|3>,|4>,|5>`` excited triplet,
``|6>`` ISC ground singlet, ``|7>`` ISC excited singlet. Rates are in MHz (times in us); the
optical gap ``Delta_EG`` enters only the energy/thermodynamic bookkeeping (incoherent regime).

Reference: [[nv-optical-pump-thermodynamics-medina]], [[nv-es-temperature-ernst]].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import qutip as qt

__all__ = [
    "OpticalCycle",
    "model6_optical_cycle",
    "model7_full_optical_cycle",
    "run_optical_cycle",
    "ernst_isc_rate",
    "GAMMA_RAD", "KAPPA_EI", "KAPPA_I", "KAPPA_IG", "GAMMA_NC", "OPTICAL_GAP",
]

# ---- Medina Table I rates (MHz) -------------------------------------------- #
GAMMA_RAD = 77.0      # radiative E->G (spin-conserving); fluorescence rate
KAPPA_EI = 15.0       # ISC E -> singlet |8>
KAPPA_I = 1000.0      # singlet |8> -> |7> (fast, ~1 GHz)
KAPPA_IG = 0.25       # singlet |7> -> GS
GAMMA_NC = 0.25       # spin-nonconserving radiative decay
# Optical gap Delta_EG (MHz) ~ 1.945 eV ~ 470 THz; only sets the energy scale for W, Q.
OPTICAL_GAP = 4.70e8
D_GS_MHZ = 2870.0     # ground ZFS
D_ES_MHZ = 1410.0     # excited ZFS (Medina)


def ernst_isc_rate(temperature: float | None, base: float = KAPPA_EI,
                   delta_e: float = 16.0, phonon: float = 10.0) -> float:
    """Phenomenological temperature-dependent ISC rate (Ernst): a constant ``base`` plus a
    phonon-activated Bose term ``phonon / (exp(delta_e / T) - 1)`` (MHz; ``delta_e`` in MHz-energy
    units). ``temperature=None`` (or <= 0) returns the constant ``base`` (Medina baseline)."""
    if temperature is None or temperature <= 0.0:
        return base
    x = delta_e / temperature
    if x > 700.0:
        return base
    return base + phonon / np.expm1(x)


@dataclass
class OpticalCycle:
    """An NV optical-cycle model: Hamiltonian, Lindblad jump operators and thermodynamic helpers."""

    dim: int
    hamiltonian: qt.Qobj                       # H (diagonal; sets energies for W/Q)
    pump_ops: list[qt.Qobj]                    # laser pump jumps WITHOUT the sqrt(Gamma_p) factor
    decay_ops: list[qt.Qobj]                   # all heat-dissipating jumps (fixed)
    p_ground: qt.Qobj                          # projector onto the ground triplet
    p_excited: qt.Qobj                         # projector onto the excited triplet
    optical_gap: float = OPTICAL_GAP
    label: str = "optical-cycle"
    ground_indices: list[int] = field(default_factory=lambda: [0, 1, 2])

    def collapse_ops(self, gamma_p: float) -> list[qt.Qobj]:
        """Full c_ops for laser power ``gamma_p`` (>=0): scaled pump jumps + fixed decay jumps."""
        return [np.sqrt(gamma_p) * op for op in self.pump_ops] + list(self.decay_ops)

    def work_rate(self, rho: qt.Qobj, gamma_p: float) -> float:
        """Laser work current ``W = Gamma_p * gamma * Delta_EG * P_G`` (Medina Eq. 11)."""
        p_g = float(qt.expect(self.p_ground, rho))
        return gamma_p * GAMMA_RAD * self.optical_gap * p_g

    def heat_rate(self, rho: qt.Qobj) -> float:
        """Spin-conserving heat current ``Q_c = -Delta_EG * gamma * P_E`` (∝ fluorescence)."""
        p_e = float(qt.expect(self.p_excited, rho))
        return -self.optical_gap * GAMMA_RAD * p_e

    def fluorescence(self, rho: qt.Qobj) -> float:
        """Photoluminescence ∝ excited-state population (``gamma * P_E``)."""
        return GAMMA_RAD * float(qt.expect(self.p_excited, rho))


def _ket(dim: int, i: int) -> qt.Qobj:
    return qt.basis(dim, i)


def _transition(dim: int, to: int, frm: int) -> qt.Qobj:
    return _ket(dim, to) * _ket(dim, frm).dag()


def _ground_projector(dim: int) -> qt.Qobj:
    """Projector onto the ground triplet (indices 0,1,2)."""
    diag = np.zeros(dim)
    diag[:3] = 1.0
    return qt.Qobj(np.diag(diag))


def _excited_projector(dim: int) -> qt.Qobj:
    """Projector onto the excited triplet (indices 3,4,5)."""
    diag = np.zeros(dim)
    diag[3:6] = 1.0
    return qt.Qobj(np.diag(diag))


def model6_optical_cycle(spin_nonconserving: bool = True,
                         optical_gap: float = OPTICAL_GAP) -> OpticalCycle:
    """Reduced 6-level NV optical cycle: GS triplet (0,1,2) + ES triplet (3,4,5).

    Pump ``L_{1..3} = |E_i><G_i|`` (scaled by ``sqrt(Gamma_p)`` at run time), radiative decay
    ``L_{4..6} = sqrt(gamma)|G_i><E_i|`` (+ optional spin-nonconserving ``sqrt(gamma_nc)``)."""
    dim = 6
    # diagonal energies: ground ~ 0, excited shifted by the optical gap (ZFS is negligible here)
    energies = [0.0, 0.0, 0.0, optical_gap, optical_gap, optical_gap]
    h = qt.Qobj(np.diag(energies))
    pump = [_transition(dim, 3 + i, i) for i in range(3)]                 # G_i -> E_i
    decay = [np.sqrt(GAMMA_RAD) * _transition(dim, i, 3 + i) for i in range(3)]  # E_i -> G_i
    if spin_nonconserving:
        for i in range(3):
            for jdest in range(3):
                if jdest != i:
                    decay.append(np.sqrt(GAMMA_NC) * _transition(dim, jdest, 3 + i))
    p_g = _ground_projector(dim)
    p_e = _excited_projector(dim)
    return OpticalCycle(dim=dim, hamiltonian=h, pump_ops=pump, decay_ops=decay,
                        p_ground=p_g, p_excited=p_e, optical_gap=optical_gap,
                        label="M6 optical cycle (6-level)")


def model7_full_optical_cycle(temperature: float | None = None,
                              optical_gap: float = OPTICAL_GAP) -> OpticalCycle:
    """Full 8-level NV optical cycle (Medina): GS triplet (0,1,2), ES triplet (3,4,5), ISC singlet
    ground ``|6>`` and excited ``|7>``. All 17 jump operators; ``temperature`` (Ernst) raises the
    ISC E->singlet rate via :func:`ernst_isc_rate` (``None`` = Medina constant baseline)."""
    dim = 8
    g_singlet, e_singlet = 6, 7
    energies = [0.0, 0.0, 0.0, optical_gap, optical_gap, optical_gap,
                0.5 * optical_gap, 0.5 * optical_gap]  # singlet placed between G and E manifolds
    h = qt.Qobj(np.diag(energies))

    kappa_ei = ernst_isc_rate(temperature)
    pump = [_transition(dim, 3 + i, i) for i in range(3)]                 # G_i -> E_i (work)
    decay: list[qt.Qobj] = []
    # radiative E_i -> G_i (spin-conserving, fluorescence)
    decay += [np.sqrt(GAMMA_RAD) * _transition(dim, i, 3 + i) for i in range(3)]
    # ISC E_i -> excited singlet |7>
    decay += [np.sqrt(kappa_ei) * _transition(dim, e_singlet, 3 + i) for i in range(3)]
    # fast singlet |7> -> |6>
    decay.append(np.sqrt(KAPPA_I) * _transition(dim, g_singlet, e_singlet))
    # singlet |6> -> ground G_i
    decay += [np.sqrt(KAPPA_IG) * _transition(dim, i, g_singlet) for i in range(3)]
    # spin-nonconserving radiative E_i -> G_j (j != i)
    for i in range(3):
        for jdest in range(3):
            if jdest != i:
                decay.append(np.sqrt(GAMMA_NC) * _transition(dim, jdest, 3 + i))

    p_g = _ground_projector(dim)
    p_e = _excited_projector(dim)
    return OpticalCycle(dim=dim, hamiltonian=h, pump_ops=pump, decay_ops=decay,
                        p_ground=p_g, p_excited=p_e, optical_gap=optical_gap,
                        label="M7 full optical cycle (8-level)")


def run_optical_cycle(model: OpticalCycle, rho0: qt.Qobj, tlist: np.ndarray,
                      gamma_p: float | Callable[[float], float] = 1.13) -> dict:
    """Evolve an :class:`OpticalCycle` under laser power ``gamma_p`` (constant or callable of time)
    via ``qt.mesolve``; return populations + thermodynamic trajectories.

    For a time-dependent ``gamma_p(t)`` the pump jumps use the QuTiP string/`Coefficient` form via
    a function coefficient on ``sqrt(gamma_p(t))``.
    """
    if callable(gamma_p):
        # time-dependent collapse operators: [op, coeff(t)] with coeff = sqrt(gamma_p(t))
        def _coeff(t, args=None):
            return float(np.sqrt(max(gamma_p(t), 0.0)))
        c_ops = [[op, _coeff] for op in model.pump_ops] + list(model.decay_ops)
        gp_for_thermo = np.array([gamma_p(t) for t in tlist])
    else:
        c_ops = model.collapse_ops(gamma_p)
        gp_for_thermo = np.full(len(tlist), float(gamma_p))

    result = qt.mesolve(model.hamiltonian, rho0, tlist, c_ops, e_ops=[])
    states = result.states
    p_g = np.array([float(qt.expect(model.p_ground, s)) for s in states])
    p_e = np.array([float(qt.expect(model.p_excited, s)) for s in states])
    work = gp_for_thermo * GAMMA_RAD * model.optical_gap * p_g
    heat = -model.optical_gap * GAMMA_RAD * p_e
    entropy = np.array([float(qt.entropy_vn(s)) for s in states])
    return {
        "times": np.asarray(tlist), "states": states, "p_ground": p_g, "p_excited": p_e,
        "work": work, "heat": heat, "fluorescence": GAMMA_RAD * p_e, "entropy": entropy,
        "label": model.label,
    }
