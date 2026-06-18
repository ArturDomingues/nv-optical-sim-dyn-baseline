"""ODMR protocols: the frequency sweep, CW-ODMR and pulsed ODMR (serial + parallel)."""

from __future__ import annotations

from functools import partial

import numpy as np
import qutip as qt
from tqdm.auto import tqdm

from ..models import A_GS, B, D_GS, MU_E
from ._defaults import default_pump_rate, odmr_field, odmr_pump_factor, odmr_pump_rate, odmr_rabi


def odmr_frequency_sweep(field: float = odmr_field, n_points: int | None = None) -> np.ndarray:
    """MW frequency array for an ODMR sweep (centered on ``D_GS ± MU_E*field``).

    ``n_points=None`` reproduces the masters multi-segment sweep (dense around the two
    Zeeman-split resonances + their hyperfine windows). An integer ``n_points`` returns a
    coarse sweep over the same range with the two resonances guaranteed present (useful for
    quick runs).
    """
    res_low = D_GS - MU_E * field   # m_s = 0 -> -1 transition
    res_high = D_GS + MU_E * field  # m_s = 0 -> +1 transition

    if n_points is not None:
        lo = res_low - MU_E * field * 0.4
        hi = res_high + MU_E * field * 0.4
        coarse = np.linspace(lo, hi, max(n_points - 2, 2))
        return np.unique(np.concatenate([coarse, [res_low, res_high]]))

    # Dense multi-segment sweep (original terse: wslhfl/wslhf/.../wshhfh).
    frac, n_hf, n_side = (2 / 5, 25, 10) if field < 500 else (1 / 5, 25, 20)
    edge = frac * field
    hf = 4 * A_GS[0]  # hyperfine half-window
    low_branch_lo = np.linspace(res_low - MU_E * field * frac, res_low - hf, 2 * n_side)
    low_hyperfine = np.linspace(res_low - hf, res_low + hf, n_hf)
    low_branch_hi = np.linspace(res_low + hf, res_low + MU_E * field * frac, 2 * n_side)
    middle = np.linspace(
        res_low + MU_E * field * frac + edge,
        res_high - MU_E * field * frac - edge,
        n_side + n_side // 2,
    )
    high_branch_lo = np.linspace(res_high - MU_E * field * frac, res_high - hf, 2 * n_side)
    high_hyperfine = np.linspace(res_high - hf, res_high + hf, n_hf)
    high_branch_hi = np.linspace(res_high + hf, res_high + MU_E * field * frac, 2 * n_side)
    return np.unique(np.hstack([
        low_branch_lo, low_hyperfine, low_branch_hi, middle,
        high_branch_lo, high_hyperfine, high_branch_hi,
    ]))


# --------------------------------------------------------------------------- #
# CW-ODMR (continuous MW + laser)
# --------------------------------------------------------------------------- #
def cw_odmr_serial(
    duration, mw_frequencies, model, init_state, exp_ops,
    b=B, om_r=odmr_rabi, w_p=odmr_pump_rate,
):
    """CW-ODMR, serial over the frequency sweep. Returns
    ``(mw_frequencies, pl_signal, times, populations, results)``."""
    results = np.array([])
    pl_signal = np.array([])
    for w in tqdm(mw_frequencies):
        _, result = model(
            duration, init_state, b=b, om_r=om_r, om=w,
            w_p=w_p / odmr_pump_factor, ti=0.0, mode="Laser-MW", progress_bar="OFF",
        )  # type: ignore
        results = np.append(results, result)
        pl = np.add(np.add(result.expect[3], result.expect[4]), result.expect[5])
        pl_signal = np.append(pl_signal, np.sum(pl))

    populations = np.array(
        [np.concatenate([qt.expect(op, res.states) for res in results], axis=0) for op in exp_ops]
    )
    times = results[0].times
    return mw_frequencies, pl_signal, times, populations, results


def _single_frequency_cw_odmr(w, model, duration, init_state, b, om_r, w_p):
    _, result = model(
        duration, init_state, b=b, om_r=om_r, om=w,
        w_p=w_p / odmr_pump_factor, ti=0.0, mode="Laser-MW", progress_bar="OFF",
    )
    return result


def cw_odmr(
    duration, mw_frequencies, model, init_state, exp_ops,
    b=B, om_r=odmr_rabi, w_p=odmr_pump_rate,
):
    """CW-ODMR, parallel over the frequency sweep (``qt.loky_pmap``)."""
    run_single = partial(
        _single_frequency_cw_odmr, model=model, duration=duration,
        init_state=init_state, b=b, om_r=om_r, w_p=w_p,
    )
    results = list(qt.loky_pmap(run_single, mw_frequencies, progress_bar="tqdm",
                                map_kw={"fail_fast": True}))  # type: ignore
    populations = np.array([[qt.expect(op, res.states) for op in exp_ops] for res in results])
    pl_signal = np.array([np.sum(pops[3] + pops[4] + pops[5]) for pops in populations])
    times = results[0].times  # type: ignore
    return mw_frequencies, pl_signal, times, populations, results


# --------------------------------------------------------------------------- #
# Pulsed ODMR: laser (t_prepare) -> free (t_wait) -> pi-pulse -> laser readout (t_readout)
# --------------------------------------------------------------------------- #
def pulsed_odmr_serial(
    t_prepare, t_wait, t_readout, mw_frequencies, model, init_state, exp_ops,
    b=B, om_r=odmr_rabi, w_p=default_pump_rate,
):
    """Pulsed ODMR, serial over the frequency sweep."""
    pl_signal = np.array([])
    times = None
    population_stack = None
    result_stack = None
    for w in tqdm(mw_frequencies):
        segment_results = np.array([])
        ti = 0.0
        tf, result = model(t_prepare, init_state, b=b, om_r=om_r, om=w, w_p=w_p,
                           ti=ti, mode="Laser", progress_bar="OFF")
        segment_results = np.append(segment_results, result)
        state = result.states[-1]
        tf, result = model(t_wait, state, b=b, om_r=om_r, om=w, w_p=w_p,
                           ti=tf, mode="Free", progress_bar="OFF")
        segment_results = np.append(segment_results, result)
        state = result.states[-1]
        t_pi = np.pi / om_r
        tf, result = model(t_pi, state, b=b, om_r=om_r, om=w, w_p=w_p,
                           ti=tf, mode="MW", progress_bar="OFF")
        segment_results = np.append(segment_results, result)
        state = result.states[-1]
        # Readout laser always pumps at odmr_pump_rate (matches the masters notebook).
        tf, result = model(t_readout, state, b=b, om_r=om_r, om=w, w_p=odmr_pump_rate,
                           ti=tf, mode="Laser", progress_bar="OFF")
        segment_results = np.append(segment_results, result)

        populations = np.array(
            [np.concatenate([qt.expect(op, res.states) for res in segment_results]) for op in exp_ops]
        )
        pl_signal = np.append(pl_signal, np.sum(populations[3] + populations[4] + populations[5]))
        segment_times = np.concatenate([res.times for res in segment_results])
        if times is None:
            times = np.array([segment_times])
            result_stack = np.array([segment_results])
            population_stack = np.array([populations])
        else:
            times = np.append(times, [segment_times], axis=0)
            result_stack = np.append(result_stack, [segment_results], axis=0)
            population_stack = np.append(population_stack, [populations], axis=0)
    return mw_frequencies, pl_signal, times, population_stack, result_stack


def _single_frequency_pulsed_odmr(w, t_prepare, t_wait, t_readout, model, init_state, exp_ops,
                                  b, om_r, w_p):
    state = init_state
    segment_results = np.array([])
    ti = 0.0
    tf, result = model(t_prepare, state, b=b, om_r=om_r, om=w, w_p=w_p / odmr_pump_factor,
                       ti=ti, mode="Laser", progress_bar="OFF")
    segment_results = np.append(segment_results, result)
    state = result.states[-1]
    tf, result = model(t_wait, state, b=b, om_r=om_r, om=w, w_p=w_p,
                       ti=tf, mode="Free", progress_bar="OFF")
    segment_results = np.append(segment_results, result)
    state = result.states[-1]
    t_pi = np.pi / om_r
    tf, result = model(t_pi, state, b=b, om_r=om_r, om=w, w_p=w_p,
                       ti=tf, mode="MW", progress_bar="OFF")
    segment_results = np.append(segment_results, result)
    state = result.states[-1]
    # FIX: readout laser must pump at odmr_pump_rate (the masters serial version hardcodes it;
    # the previous parallel version wrongly reused the passed w_p).
    tf, result = model(t_readout, state, b=b, om_r=om_r, om=w, w_p=odmr_pump_rate,
                       ti=tf, mode="Laser", progress_bar="OFF")
    segment_results = np.append(segment_results, result)

    populations = np.array(
        [np.concatenate([qt.expect(op, res.states) for res in segment_results]) for op in exp_ops]
    )
    pl_signal = np.sum(qt.expect(exp_ops[3] + exp_ops[4] + exp_ops[5], result.states))
    segment_times = np.concatenate([res.times for res in segment_results])
    return (w, pl_signal, segment_times, populations, segment_results)


def pulsed_odmr(
    t_prepare, t_wait, t_readout, mw_frequencies, model, init_state, exp_ops,
    b=B, om_r=odmr_rabi, w_p=odmr_pump_rate,
):
    """Pulsed ODMR, parallel over the frequency sweep."""
    run_single = partial(
        _single_frequency_pulsed_odmr, t_prepare=t_prepare, t_wait=t_wait, t_readout=t_readout,
        model=model, init_state=init_state, exp_ops=exp_ops, b=b, om_r=om_r, w_p=w_p,
    )
    outs = qt.loky_pmap(run_single, mw_frequencies, progress_bar="tqdm",
                        map_kw={"fail_fast": True})  # type: ignore
    pl_signal = np.array([item[1] for item in outs])      # type: ignore
    # Pre-allocate + fill an object array (robust to ragged per-sequence times/populations; lengths
    # are uniform for a pure frequency sweep, but this matches the coherence-sweep helpers).
    n = len(outs)
    times = np.empty(n, dtype=object)
    populations = np.empty(n, dtype=object)
    results = np.empty(n, dtype=object)
    for i, item in enumerate(outs):  # type: ignore
        times[i], populations[i], results[i] = item[2], item[3], item[4]
    return mw_frequencies, pl_signal, times, populations, results
