"""Compute functions: each runs a protocol sweep and returns a plain dict of arrays."""

from __future__ import annotations

import time
from functools import partial

import numpy as np
import qutip as qt
from tqdm.auto import tqdm

import core.plot_funcs as pf
from core import protocols as P
from core.models import B0, D_ES, ID_N15, K_IND, K_S, N1, N2, N4, N5, N6, NC, dynamics_no
from core.utils import superposition_label

from ._infra import (
    MODELS,
    STATIC_HAMILTONIANS,
    SUBSPACE_INDICES,
    _energy_basis_labels,
    initial_state,
    measurement_operators,
)


def compute_energy_levels(fields: np.ndarray, model: str = "no-HF", theta: float = 0.0,
                          phi: float = 0.0, table_b: list[float] | None = None) -> dict:
    """GS/ES eigen-energies vs magnetic field for the chosen model, two-pass.

    Pass 1: GS and ES eigen-energies on the dense `fields` grid. Pass 2: eigen*states* at a
    few `table_b` (G) values, labeled as superpositions. Also returns zoom windows at the
    GS/ES level anti-crossings (GSLAC ~ D_GS/mu_e, ESLAC ~ D_ES/mu_e). no-HF = 7-level (3+3
    sub-manifolds); Doherty/Duarte = 14-level (6+6, NV(7) x N15(2)). `theta`/`phi` set the
    field orientation (radians) via B0.
    """
    has_hf = MODELS[model][1]
    idx = SUBSPACE_INDICES[has_hf]
    h_func = STATIC_HAMILTONIANS[model]
    gs_labels, es_labels = _energy_basis_labels(has_hf)

    # pass 1: eigen-energies
    energies_gs, energies_es = [], []
    for field in tqdm(fields, desc=f"energy levels [{model}]"):
        h = h_func(B0(field, phi, theta)).full()
        energies_gs.append(np.linalg.eigvalsh(h[np.ix_(idx["GS"], idx["GS"])]).real)
        energies_es.append(np.linalg.eigvalsh(h[np.ix_(idx["ES"], idx["ES"])]).real)

    # pass 2: eigen-states -> superposition labels at the table-row fields
    if table_b is None:
        table_b = [0.0, 100.0, 350.0, 510.0, round(P.D_GS / P.MU_E, 1), 1200.0]
    state_table = []
    for b in table_b:
        h = h_func(B0(b, phi, theta)).full()
        h_gs = qt.Qobj(h[np.ix_(idx["GS"], idx["GS"])])
        h_es = qt.Qobj(h[np.ix_(idx["ES"], idx["ES"])])
        _, evg = h_gs.eigenstates()
        _, eve = h_es.eigenstates()
        state_table.append({
            "B": b,
            "GS": [superposition_label(v, gs_labels) for v in evg],
            "ES": [superposition_label(v, es_labels) for v in eve],
        })

    # zoom windows (x_center, x_half, y_half) about the anti-crossings; y centered on 0 MHz
    # (where m_s=0 and m_s=-1 meet). ES window sits on the ESLAC / ISC region.
    if has_hf:
        zoom_gs, zoom_es = (P.D_GS / P.MU_E, 15.0, 10.0), (496.0, 60.0, 50.0)
    else:
        zoom_gs, zoom_es = (P.D_GS / P.MU_E, 15.0, 10.0), (D_ES / P.MU_E, 15.0, 10.0)
    return {"fields": np.asarray(fields), "model": model,
            "energies_gs": np.asarray(energies_gs), "energies_es": np.asarray(energies_es),
            "zoom_gs": zoom_gs, "zoom_es": zoom_es,
            "table_b": list(table_b), "state_table": state_table}


def compute_populations(field: float = P.odmr_field, t_prepare: float = 10.0,    # field G; times us
                        t_wait: float = 1.0, t_readout: float = 6.0,
                        mw_time: float | None = 5.0, model_label: str = "no-HF",
                        k_index: int = K_IND) -> dict:
    """Population dynamics over a Laser -> Free -> MW -> Laser readout sequence.

    ``mw_time`` (us) is the MW-on duration (masters default 5.0); ``mw_time=None`` uses a
    single pi-pulse ``pi/om_r``. ``model_label`` selects no-HF (7-level) or Doherty/Duarte
    (14-level). ``k_index`` selects the K_S transition-rate set.
    """
    model, has_hf = MODELS[model_label]
    state = initial_state(has_hf)
    exp_ops = measurement_operators(has_hf)
    drive = P.D_GS - P.MU_E * field
    rabi = P.odmr_rabi
    field_vec = B0(field, 0.0, 0.0)
    mw_duration = (np.pi / rabi) if mw_time is None else mw_time
    segments = [
        (t_prepare, "Laser", P.odmr_pump_rate),
        (t_wait, "Free", 0.0),
        (mw_duration, "MW", 0.0),
        (t_readout, "Laser", P.odmr_pump_rate),
    ]
    results, tis, tfs = [], [], []
    ti = 0.0
    for duration, mode, pump in segments:
        tis.append(ti)
        tf, result = model(duration, state, b=field_vec, om_r=rabi, om=drive, w_p=pump,
                           ti=ti, mode=mode, k_index=k_index, progress_bar="OFF")
        results.append(result)
        state = result.states[-1]
        tfs.append(tf)
        ti = tf
    populations = np.real(np.array(
        [np.concatenate([qt.expect(op, res.states) for res in results]) for op in exp_ops]
    ))
    times = np.concatenate([res.times for res in results])
    return {"populations": populations, "times": times,
            "tis": np.asarray(tis), "tfs": np.asarray(tfs)}


def compute_populations_rates(model_label: str = "no-HF", k_indices: list[int] | None = None,
                              t_prepare: float = 10.0, t_wait: float = 2.0,
                              mw_time: float = 5.0, t_readout: float = 5.0) -> dict:
    """Readout PL trace (sum of excited populations) over a Laser->Free->MW->Laser sequence,
    one curve per K_S transition-rate set (masters 'Comparison of Transition Rates')."""
    if k_indices is None:
        k_indices = list(range(len(K_S)))
    curves, times = {}, None
    for k in tqdm(k_indices, desc=f"K_S populations [{model_label}]"):
        d = compute_populations(model_label=model_label, k_index=k, t_prepare=t_prepare,
                                t_wait=t_wait, mw_time=mw_time, t_readout=t_readout)
        pops = d["populations"]
        curves[f"K_S[{k}]"] = pops[3] + pops[4] + pops[5]
        times = d["times"]
    return {"times": times, "curves": curves, "model": model_label}


def compute_populations_ks_comparison(model_label: str = "no-HF",
                                      k_indices: list[int] | None = None,
                                      highlight_k: int = K_IND, t_prepare: float = 10.0,
                                      t_wait: float = 2.0, mw_time: float = 5.0,
                                      t_readout: float = 5.0) -> dict:
    """Full population dynamics (n1..n7, n_c) over a Laser->Free->MW->Laser sequence, one set per
    K_S transition-rate set (masters 'Comparison of Transition Rates', plots.ipynb Cell 96). The
    plot overlays all sets and highlights ``highlight_k``."""
    if k_indices is None:
        k_indices = list(range(len(K_S)))
    per_k, times = {}, None
    for k in tqdm(k_indices, desc=f"K_S populations [{model_label}]"):
        d = compute_populations(model_label=model_label, k_index=k, t_prepare=t_prepare,
                                t_wait=t_wait, mw_time=mw_time, t_readout=t_readout)
        per_k[k] = d["populations"]
        times = d["times"]
    return {"times": times, "per_k": per_k, "highlight_k": highlight_k, "model": model_label}


def compute_photoluminescence(powers: np.ndarray, field: float = P.odmr_field,
                              laser_time: float = 10.0) -> dict:
    """Steady-state PL (excited populations) vs laser pump rate, for the no-HF model."""
    model = dynamics_no
    exp_ops = measurement_operators(False)
    pl_operator = exp_ops[3] + exp_ops[4] + exp_ops[5]
    drive = P.D_GS - P.MU_E * field
    field_vec = B0(field, 0.0, 0.0)
    pl = []
    for pump in tqdm(powers, desc="PL vs power"):
        _, result = model(laser_time, initial_state(False), b=field_vec, om_r=P.odmr_rabi,
                          om=drive, w_p=pump, ti=0.0, mode="Laser", progress_bar="OFF")
        pl.append(float(np.real(qt.expect(pl_operator, result.states[-1]))))
    return {"powers": np.asarray(powers), "pl": np.asarray([pl]), "labels": np.asarray(["no-HF"])}


def compute_pl_traces(w_p_values: np.ndarray, init_label: str = "ms0", field: float = 100.0,
                      laser_time: float = 10.0, weighting: str = "magaletti") -> dict:
    """PL time-traces (excited-state population vs time) under continuous laser, one curve per
    pump rate ``W_p``. ``weighting='hirose'`` scales by the radiative rate (rate-weighted PL).
    ``init_label`` in {ms0, ms-1, super}. Reproduces the masters Magaletti/Hirose PL plots."""
    model = dynamics_no
    exp_ops = measurement_operators(False)
    pl_operator = exp_ops[3] + exp_ops[4] + exp_ops[5]
    drive = P.D_GS - P.MU_E * field
    field_vec = B0(field, 0.0, 0.0)
    inits = {"ms0": N1, "ms-1": N2,
             "super": ((N1 + N2) / 2)}  # ms0, ms-1, even GS mix
    init_state = inits.get(init_label, N1)
    rad_rate = K_S[K_IND][0]  # k41 radiative rate (MHz) for the Hirose weighting
    curves, times = {}, None
    for w in tqdm(w_p_values, desc=f"PL traces [{weighting}]"):
        _, result = model(laser_time, init_state, b=field_vec, om_r=P.odmr_rabi, om=drive,
                          w_p=w, ti=0.0, mode="Laser", progress_bar="OFF")
        pl = np.real(qt.expect(pl_operator, result.states))
        if weighting == "hirose":
            pl = pl * rad_rate
        curves[f"W_p={w:.2f}"] = pl
        times = result.times
    return {"times": times, "curves": curves, "weighting": weighting, "field": field,
            "init": init_label}


def compute_pl_preparations(model_label: str = "no-HF", field: float = 100.0,
                            rabi: float | None = None, laser_time: float = 10.0,
                            free_time: float = 1.0) -> dict:
    """Readout PL for three spin preparations |0>, (|0>+|1>)/sqrt(2), |1>, two ways:
    ``pure`` = laser-only readout of the prepared GS state; ``prepared`` = the readout laser of a
    full Laser->Free->MW(dt_mw)->Laser sequence (dt_mw in {0, pi/2, pi}/rabi). Reproduces the
    masters pure-vs-prepared PL plots (plots.ipynb Cells 99-101 / 103-105; dissertation Figs 19/20).
    """
    model, has_hf = MODELS[model_label]
    rabi = P.odmr_rabi if rabi is None else rabi
    exp_ops = measurement_operators(has_hf)
    pl_operator = exp_ops[3] + exp_ops[4] + exp_ops[5]
    drive = P.D_GS - P.MU_E * field
    field_vec = B0(field, 0.0, 0.0)
    pump = P.odmr_pump_rate

    labels = [r"$|0\rangle$", r"$\frac{|0\rangle+|1\rangle}{\sqrt{2}}$", r"$|1\rangle$"]
    gs_states = [N1, (N1 + N2 + NC + NC.dag()) / 2, N2]  # ms0, superposition, ms-1
    if has_hf:
        gs_states = [(s & ID_N15) / 2 for s in gs_states]
    dt_mws = [0.0, 0.5 * np.pi / rabi, np.pi / rabi]

    pure, prepared, times = {}, {}, None
    for lbl, pstate, dt_mw in zip(labels, gs_states, dt_mws):
        # pure: laser-only readout of the prepared pure state
        _, res = model(laser_time, pstate, b=field_vec, om_r=rabi, om=drive, w_p=pump,
                       ti=0.0, mode="Laser", progress_bar="OFF")
        pure[lbl] = np.real(qt.expect(pl_operator, res.states))
        times = res.times - res.times[0]
        # prepared: full sequence, keep the readout-laser segment
        state = initial_state(has_hf)
        ti = 0.0
        for dt, mode in [(laser_time, "Laser"), (free_time, "Free"),
                         (dt_mw, "MW"), (laser_time, "Laser")]:
            tf, res = model(dt, state, b=field_vec, om_r=rabi, om=drive, w_p=pump,
                            ti=ti, mode=mode, progress_bar="OFF")
            state = res.states[-1]
            ti = tf
        prepared[lbl] = np.real(qt.expect(pl_operator, res.states))  # last (readout) segment
    return {"times": np.asarray(times), "labels": labels, "pure": pure, "prepared": prepared,
            "field": field, "rabi": rabi, "model": model_label}


def compute_odmr(model_labels: list[str], field: float = P.odmr_field, n_points: int = 10,
                 duration: float = 10.0, pulsed: bool = False, rabi: float | None = None) -> dict:
    """CW (or pulsed) ODMR PL vs frequency for the given models. ``rabi`` overrides the Rabi
    frequency (e.g. pi/T2, 2pi/T2 variants); defaults to the ODMR Rabi."""
    om_r = P.odmr_rabi if rabi is None else rabi
    mw_frequencies = P.odmr_frequency_sweep(field, n_points=n_points)
    field_vec = B0(field, 0.0, 0.0)
    spectra = {}
    for label in model_labels:
        model, has_hf = MODELS[label]
        exp_ops = measurement_operators(has_hf)
        state = initial_state(has_hf)
        if pulsed:
            _, pl_signal, *_ = P.pulsed_odmr(10.0, 1.0, 6.0, mw_frequencies, model, state,
                                             exp_ops, b=field_vec, om_r=om_r)
        else:
            _, pl_signal, *_ = P.cw_odmr(duration, mw_frequencies, model, state, exp_ops,
                                         b=field_vec, om_r=om_r)
        spectra[label] = np.real(np.asarray(pl_signal))
    return {"mw_frequencies": mw_frequencies, "field": field, "rabi": om_r,
            "spectra": spectra, "pulsed": pulsed}


def compute_ramsey(model_labels: list[str], free_times: np.ndarray, field: float = 0.0,
                   t_prepare: float = 10.0, t_wait: float = 1.0, t_readout: float = 6.0,
                   rabi: float | None = None) -> dict:
    """Ramsey contrast-normalized signal vs interrogation time, for the given models.
    ``rabi`` overrides the Rabi frequency (pi/T2, 2pi/T2, 10pi/T2 variants)."""
    drive = P.D_GS - P.MU_E * field
    field_vec = B0(field, 0.0, 0.0)
    rabi = P.default_rabi if rabi is None else rabi
    signals = {}
    for label in model_labels:
        model, has_hf = MODELS[label]
        exp_ops = measurement_operators(has_hf)
        state = initial_state(has_hf)
        pl_operator = exp_ops[3] + exp_ops[4] + exp_ops[5]
        s0, s1, _, _ = P.readout_contrast(model, state, pl_operator, field,
                                          P.default_pump_rate, P.ramsey_pump_rate, rabi)
        _, pl_signal, *_ = P.ramsey_sweep(t_prepare, t_wait, free_times, t_readout, model,
                                          state, exp_ops, field_vec, rabi,
                                          P.default_pump_rate, drive)
        signals[label] = {"signal": np.real(pf.ramsey_contrast_signal(pl_signal, s0, s1)),
                          "has_hyperfine": has_hf}
    return {"free_times": np.asarray(free_times), "field": field, "rabi": rabi,
            "signals": signals}


def compute_pl_vs_b(model_labels: list[str], fields: np.ndarray | None = None,
                    angles_deg: list[float] | None = None,
                    laser_time: float = 10.0, free_time: float = 1.0,
                    pump: float = 2.0) -> dict:
    """Steady-state PL (excited populations) vs magnetic field, for several orientation
    angles. FULL masters defaults: B = 0..1300 G step 5, angles {0,0.5,1,4,8,10} deg."""
    if fields is None:
        fields = np.arange(0.0, 1301.0, 5.0)
    if angles_deg is None:
        angles_deg = [0.0, 0.5, 1.0, 4.0, 8.0, 10.0]
    rabi = P.default_rabi
    curves: dict[str, dict[str, np.ndarray]] = {}
    for label in model_labels:
        model, has_hf = MODELS[label]
        state0 = initial_state(has_hf)
        exp_ops = measurement_operators(has_hf)
        pl_operator = exp_ops[3] + exp_ops[4] + exp_ops[5]
        per_angle = {}
        for adeg in tqdm(angles_deg, desc=f"PL vs B [{label}] angles"):
            angle = np.deg2rad(adeg)
            pl = []
            for b in tqdm(fields, desc=f"  {adeg}deg fields", leave=False):
                field_vec = B0(b, 0.0, angle)
                drive = P.D_GS - P.MU_E * field_vec[2]
                state = state0
                for dt, mode in [(laser_time, "Laser"), (free_time, "Free"), (laser_time, "Laser")]:
                    _, r = model(dt, state, b=field_vec, om_r=rabi, om=drive, w_p=pump,
                                 ti=0.0, mode=mode, progress_bar="OFF")
                    state = r.states[-1]
                pl.append(float(np.real(qt.expect(pl_operator, state))))
            per_angle[f"{adeg}deg"] = np.asarray(pl)
        curves[label] = per_angle
    return {"fields": np.asarray(fields), "angles_deg": np.asarray(angles_deg), "curves": curves}


def compute_snr(readtimes: np.ndarray | None = None, readout_pumps: np.ndarray | None = None,
                rabis: np.ndarray | None = None, init_pumps: np.ndarray | None = None,
                field: float = 100.0, rabi0: float = 15.8, init_pump: float = 2.0) -> dict:
    """Iteratively optimize the readout SNR ``(s0-s1)/sqrt((s0+s1)/2)`` over readtime, readout
    pump, Rabi, and init pump. FULL masters grids by default."""
    model = dynamics_no
    state = initial_state(False)
    pl_operator = N4 + N5 + N6

    def snr_at(readtime, w_read, rabi, w_init):
        s0, s1, _, _ = P.readout_contrast(model, state, pl_operator, field, w_init, w_read,
                                          rabi, readtime=readtime)
        denom = np.sqrt(0.5 * (s0 + s1))
        return float(np.real((s0 - s1) / denom)) if denom > 0 else 0.0

    readtimes = np.linspace(0.5, 4.0, 35) if readtimes is None else readtimes
    readout_pumps = np.linspace(0.5, 5.5, 50) if readout_pumps is None else readout_pumps
    rabis = np.linspace(4.0, 22.0, 100) if rabis is None else rabis
    init_pumps = np.linspace(0.5, 5.5, 50) if init_pumps is None else init_pumps

    snr_rt = np.array([snr_at(t, init_pump, rabi0, init_pump)
                       for t in tqdm(readtimes, desc="SNR: readtime")])
    t_opt = float(readtimes[np.argmax(snr_rt)])
    snr_wr = np.array([snr_at(t_opt, w, rabi0, init_pump)
                       for w in tqdm(readout_pumps, desc="SNR: readout pump")])
    wr_opt = float(readout_pumps[np.argmax(snr_wr)])
    snr_om = np.array([snr_at(t_opt, wr_opt, om, init_pump)
                       for om in tqdm(rabis, desc="SNR: Rabi")])
    om_opt = float(rabis[np.argmax(snr_om)])
    snr_wi = np.array([snr_at(t_opt, wr_opt, om_opt, w)
                       for w in tqdm(init_pumps, desc="SNR: init pump")])
    wi_opt = float(init_pumps[np.argmax(snr_wi)])
    return {
        "readtimes": np.asarray(readtimes), "snr_readtime": snr_rt,
        "readout_pumps": np.asarray(readout_pumps), "snr_readout": snr_wr,
        "rabis": np.asarray(rabis), "snr_rabi": snr_om,
        "init_pumps": np.asarray(init_pumps), "snr_init": snr_wi,
        "optimum": {"readtime": t_opt, "readout_pump": wr_opt, "rabi": om_opt, "init_pump": wi_opt},
    }


def _benchmark_single(payload: tuple[str, float]) -> int:
    """One laser->free->MW sequence for a model (top-level so it is picklable by loky)."""
    label, field = payload
    model, has_hf = MODELS[label]
    state = initial_state(has_hf)
    field_vec = B0(field, 0.0, 0.0)
    drive = P.D_GS - P.MU_E * field
    for dt, mode in [(10.0, "Laser"), (1.0, "Free"), (5.0, "MW")]:
        _, r = model(dt, state, b=field_vec, om_r=P.default_rabi, om=drive,
                     w_p=P.default_pump_rate, ti=0.0, mode=mode, progress_bar="OFF")
        state = r.states[-1]
    return 0


def compute_benchmark(model_labels: list[str] | None = None, runs: int = 8, repeats: int = 5,
                      field: float = 100.0) -> dict:
    """Time ``runs`` sequences per model, serial vs parallel (loky), over ``repeats`` trials."""
    if model_labels is None:
        model_labels = ["no-HF", "Doherty", "Duarte"]
    results = {}
    for label in tqdm(model_labels, desc="benchmark"):
        payloads = [(label, field)] * runs
        seq_times, par_times = [], []
        for _ in tqdm(range(repeats), desc=f"  {label} repeats", leave=False):
            t0 = time.perf_counter()
            for p in payloads:
                _benchmark_single(p)
            seq_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            qt.loky_pmap(_benchmark_single, payloads, map_kw={"fail_fast": True})  # type: ignore
            par_times.append(time.perf_counter() - t0)
        results[label] = {
            "seq_mean": float(np.mean(seq_times)), "seq_std": float(np.std(seq_times)),
            "par_mean": float(np.mean(par_times)), "par_std": float(np.std(par_times)),
        }
    return {"labels": model_labels, "results": results, "runs": runs, "repeats": repeats}


def compute_spin_echo(model_labels: list[str], echo_times: np.ndarray | None = None,
                      field: float = 0.0, t_prepare: float = 10.0, t_wait: float = 1.0,
                      t_readout: float = 6.0) -> dict:
    """Hahn echo: contrast-normalized signal vs total echo time, per model."""
    if echo_times is None:
        echo_times = np.linspace(0.05, 5.0, 40)
    drive = P.D_GS - P.MU_E * field
    field_vec = B0(field, 0.0, 0.0)
    rabi = P.default_rabi
    signals = {}
    for label in model_labels:
        model, has_hf = MODELS[label]
        exp_ops = measurement_operators(has_hf)
        state = initial_state(has_hf)
        pl_operator = exp_ops[3] + exp_ops[4] + exp_ops[5]
        s0, s1, _, _ = P.readout_contrast(model, state, pl_operator, field,
                                          P.default_pump_rate, P.ramsey_pump_rate, rabi)
        _, pl_signal, *_ = P.spin_echo_sweep(t_prepare, t_wait, echo_times, t_readout, model,
                                             state, exp_ops, field_vec, rabi,
                                             P.default_pump_rate, drive)
        signals[label] = np.real(pf.ramsey_contrast_signal(pl_signal, s0, s1))
    return {"echo_times": np.asarray(echo_times), "field": field, "signals": signals}


def compute_transition_rate_comparison(field: float = 150.0, n_points: int = 15,
                                       k_indices: list[int] | None = None) -> dict:
    """CW-ODMR PL spectra for each K_S transition-rate set (sweeps the model's ``k_index``)."""
    if k_indices is None:
        k_indices = list(range(len(K_S)))
    sweep = P.odmr_frequency_sweep(field, n_points=n_points)
    exp_ops = measurement_operators(False)
    state = initial_state(False)
    spectra = {}
    for k in k_indices:
        model = partial(dynamics_no, k_index=k)
        _, pl_signal, *_ = P.cw_odmr(10.0, sweep, model, state, exp_ops, b=B0(field, 0.0, 0.0))
        spectra[f"K_S[{k}]"] = np.real(pl_signal)
    return {"mw_frequencies": sweep, "field": field, "spectra": spectra}


def compute_pure_vs_prepared(field: float = 150.0, n_points: int = 15) -> dict:
    """CW-ODMR for a pure |0> initial state vs the prepared (GS-equilibrium) state."""
    sweep = P.odmr_frequency_sweep(field, n_points=n_points)
    exp_ops = measurement_operators(False)
    spectra = {}
    for name, state in [("pure |0>", N1), ("prepared (GS eq.)", initial_state(False))]:
        _, pl_signal, *_ = P.cw_odmr(10.0, sweep, dynamics_no, state, exp_ops, b=B0(field, 0.0, 0.0))
        spectra[name] = np.real(pl_signal)
    return {"mw_frequencies": sweep, "field": field, "spectra": spectra, "pulsed": False}
