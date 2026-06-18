"""Fast tests for the control-dependent-dissipation framework (core.open_control)."""

from __future__ import annotations

import numpy as np
import pytest

import qutip as qt

from core.models import ernst_isc_rate, model6_optical_cycle, model7_full_optical_cycle
from core.open_control import (
    ControlProblem,
    build_n_level_system,
    build_qubit,
    crab_envelope,
    gamma_ohmic,
    simulate,
    vn_entropy,
    _thermal_state,
)


def _random_density_matrix(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    rho = a @ a.conj().T
    return rho / np.trace(rho).real


@pytest.mark.parametrize("dim", [2, 3, 7])
def test_vn_entropy_bounds(dim: int) -> None:
    s = vn_entropy(_random_density_matrix(dim))
    assert -1e-9 <= s <= np.log(dim) + 1e-9


def test_vn_entropy_pure_and_mixed() -> None:
    pure = np.zeros((3, 3), dtype=complex)
    pure[0, 0] = 1.0
    assert vn_entropy(pure) == pytest.approx(0.0, abs=1e-9)
    assert vn_entropy(np.eye(3) / 3) == pytest.approx(np.log(3), abs=1e-9)


def test_crab_envelope_zero_coeffs_and_shape() -> None:
    t = np.linspace(0.0, 5.0, 17)
    freqs = np.array([1.0, 2.0])
    out = crab_envelope(t, np.zeros(2), freqs, tau=5.0, sigma=1.25)
    assert out.shape == t.shape
    assert np.allclose(out, 0.0)
    # scalar input returns a float
    assert isinstance(crab_envelope(1.0, np.array([1.0]), freqs[:1], 5.0, 1.25), float)


@pytest.mark.parametrize("omega,temp", [(1.0, 0.7), (2.5, 0.5), (0.3, 1.2)])
def test_gamma_detailed_balance(omega: float, temp: float) -> None:
    up = gamma_ohmic(-omega, temp, 0.1)   # absorption
    down = gamma_ohmic(omega, temp, 0.1)  # emission
    assert down / up == pytest.approx(np.exp(omega / temp), rel=1e-6)


def test_gamma_zero_at_zero_frequency() -> None:
    assert gamma_ohmic(0.0, 0.7, 0.1) == 0.0


def test_thermal_state_is_fixed_point() -> None:
    """Zero control field: a thermal state stays (≈) invariant under the bath."""
    p = build_qubit(n_times=40, t_final=5.0, temperature=0.5, bath_coupling=0.1)
    _, _, ent = simulate(p, np.zeros(len(p.freqs)))
    assert abs(ent[-1] - ent[0]) < 1e-3


def test_control_deforms_dissipation() -> None:
    """A nonzero field changes the final entropy — control-dependent dissipation."""
    p = build_qubit(n_times=40, t_final=5.0, temperature=0.5, bath_coupling=0.1)
    _, _, ent_free = simulate(p, np.zeros(len(p.freqs)))
    _, _, ent_drv = simulate(p, np.array([2.0, -1.5, 1.0, 0.5]))
    assert abs(ent_drv[-1] - ent_free[-1]) > 1e-2


def test_offthermal_relaxes_toward_gibbs() -> None:
    """From a pure state with no control, entropy rises toward the Gibbs value."""
    base = build_qubit(n_times=60, t_final=12.0, temperature=0.6, bath_coupling=0.15)
    pure = np.zeros((2, 2), dtype=complex)
    pure[0, 0] = 1.0
    p = ControlProblem(
        h0=base.h0, control=base.control, coupling=base.coupling, rho0=pure,
        temperature=base.temperature, bath_coupling=base.bath_coupling,
        tlist=base.tlist, freqs=base.freqs,
    )
    _, _, ent = simulate(p, np.zeros(len(p.freqs)))
    s_thermal = vn_entropy(_thermal_state(base.h0, base.temperature))
    assert ent[-1] > ent[0]                      # entropy increased
    assert abs(ent[-1] - s_thermal) < abs(ent[0] - s_thermal)  # moved toward Gibbs


# --- Kallush n-level system + QuTiP-native pieces --------------------------------------- #
@pytest.mark.parametrize("j,dim", [(0.5, 2), (1.0, 3), (1.5, 4)])
def test_build_n_level_dims(j: float, dim: int) -> None:
    p = build_n_level_system(j=j, n_times=20, t_final=4.0, n_freqs=4)
    assert isinstance(p.h0, qt.Qobj) and p.h0.shape == (dim, dim)
    assert p.h0.isherm
    rho0 = p.rho0 if isinstance(p.rho0, qt.Qobj) else qt.Qobj(p.rho0)
    assert abs(rho0.tr() - 1.0) < 1e-9


def test_brmesolve_path_runs() -> None:
    """The QuTiP Bloch-Redfield cross-check runs and gives a sane entropy near the secular one
    for the zero (free) field."""
    p = build_qubit(n_times=40, t_final=5.0, temperature=0.5, bath_coupling=0.1)
    _, _, ent_br = simulate(p, np.zeros(len(p.freqs)), mode="brmesolve")
    _, _, ent_sec = simulate(p, np.zeros(len(p.freqs)), mode="secular")
    assert np.all(ent_br >= -1e-9) and np.all(ent_br <= np.log(2) + 1e-6)
    assert abs(ent_br[-1] - ent_sec[-1]) < 0.1


# --- Medina optical-cycle models M6 / M7 ------------------------------------------------- #
def test_optical_cycle_models_build() -> None:
    m6, m7 = model6_optical_cycle(), model7_full_optical_cycle()
    assert m6.dim == 6 and m7.dim == 8
    assert m6.hamiltonian.isherm and m7.hamiltonian.isherm
    assert len(m6.collapse_ops(1.0)) >= 6           # 3 pump + 3 radiative (+ spin-nc)
    assert len(m7.collapse_ops(1.0)) >= 13          # the 17-jump cycle
    # work/heat helpers return finite numbers on the mixed ground state
    rho = qt.Qobj(np.diag([1 / 3, 1 / 3, 1 / 3] + [0.0] * (m7.dim - 3)))
    assert np.isfinite(m7.work_rate(rho, 1.13)) and np.isfinite(m7.heat_rate(rho))


def test_ernst_isc_rate_increases_with_temperature() -> None:
    base = ernst_isc_rate(None)
    assert ernst_isc_rate(10.0) >= base
    assert ernst_isc_rate(100.0) >= ernst_isc_rate(10.0)
