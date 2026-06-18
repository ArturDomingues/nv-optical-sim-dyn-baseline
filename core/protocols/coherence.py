"""Coherence protocols: Ramsey and Hahn spin echo (single sequence + parallel sweep)."""

from __future__ import annotations

from functools import partial

import numpy as np
import qutip as qt

from ..models import B
from ._defaults import default_drive, default_pump_rate, default_rabi, ramsey_pump_rate


# --------------------------------------------------------------------------- #
# Ramsey: laser -> free -> pi/2 -> free(t) -> pi/2 -> laser readout
# --------------------------------------------------------------------------- #
def ramsey(
    free_time, t_prepare, t_wait, t_readout, model, init_state, exp_ops,
    b=B, om_r=default_rabi, w_p=default_pump_rate, om=default_drive,
):
    """One Ramsey sequence at interrogation time ``free_time``. Returns
    ``(free_time, pl_signal, times, populations, results)``."""
    segment_results = np.array([])
    ti = 0.0
    tf, result = model(t_prepare, init_state, b=b, om_r=om_r, om=om, w_p=w_p,
                       ti=ti, mode="Laser", progress_bar="OFF")
    segment_results = np.append(segment_results, result)
    state = result.states[-1]
    tf, result = model(t_wait, state, b=b, om_r=om_r, om=om, w_p=w_p,
                       ti=tf, mode="Free", progress_bar="OFF")
    segment_results = np.append(segment_results, result)
    state = result.states[-1]
    t_pi_half = 0.5 * np.pi / om_r
    tf, result = model(t_pi_half, state, b=b, om_r=om_r, om=om, w_p=w_p,
                       ti=tf, mode="MW", progress_bar="OFF")
    segment_results = np.append(segment_results, result)
    state = result.states[-1]
    tf, result = model(free_time, state, b=b, om_r=om_r, om=om, w_p=w_p,
                       ti=tf, mode="Free", progress_bar="OFF")
    segment_results = np.append(segment_results, result)
    state = result.states[-1]
    tf, result = model(t_pi_half, state, b=b, om_r=om_r, om=om, w_p=w_p,
                       ti=tf, mode="MW", progress_bar="OFF")
    segment_results = np.append(segment_results, result)
    state = result.states[-1]
    tf, result = model(t_readout, state, b=b, om_r=om_r, om=om, w_p=ramsey_pump_rate,
                       ti=tf, mode="Laser", progress_bar="OFF")
    segment_results = np.append(segment_results, result)

    populations = np.array(
        [np.concatenate([qt.expect(op, res.states) for res in segment_results]) for op in exp_ops]
    )
    pl_signal = np.sum(qt.expect(exp_ops[3] + exp_ops[4] + exp_ops[5], result.states))
    times = np.concatenate([res.times for res in segment_results])
    return free_time, pl_signal, times, populations, segment_results


def ramsey_sweep(t_prepare, t_wait, free_times, t_readout, model, init_state, exp_ops,
                 b, om_r, w_p, om):
    """Ramsey, parallel over the interrogation-time sweep ``free_times``."""
    run_single = partial(
        ramsey, t_prepare=t_prepare, t_wait=t_wait, t_readout=t_readout,
        model=model, init_state=init_state, exp_ops=exp_ops, b=b, om_r=om_r, w_p=w_p, om=om,
    )
    outs = qt.loky_pmap(run_single, free_times, progress_bar="tqdm",
                        map_kw={"fail_fast": True})  # type: ignore
    free_times_out = np.array([item[0] for item in outs])  # type: ignore
    pl_signal = np.array([item[1] for item in outs])       # type: ignore
    # Pre-allocate + fill: per-sequence times (1-D) and populations (2-D (8, T)) are ragged when
    # the swept free time crosses the dynamics_* t_bins switch at dt=5 us (1000 -> 5000 points);
    # np.array(..., dtype=object) mis-broadcasts the 2-D case, so fill an object array by hand.
    n = len(outs)
    times = np.empty(n, dtype=object)
    populations = np.empty(n, dtype=object)
    results = np.empty(n, dtype=object)
    for i, item in enumerate(outs):  # type: ignore
        times[i], populations[i], results[i] = item[2], item[3], item[4]
    return free_times_out, pl_signal, times, populations, results


# --------------------------------------------------------------------------- #
# Spin echo: laser -> free -> pi/2 -> free(t/2) -> pi -> free(t/2) -> pi/2 -> readout
# --------------------------------------------------------------------------- #
def spin_echo(
    echo_time, t_prepare, t_wait, t_readout, model, init_state, exp_ops,
    b=B, om_r=default_rabi, w_p=default_pump_rate, om=default_drive,
):
    """One Hahn spin-echo sequence with total free-evolution ``echo_time`` (split by the
    refocusing pi pulse into two equal halves). Returns
    ``(echo_time, pl_signal, times, populations, results)``."""
    half = 0.5 * echo_time
    t_pi_half = 0.5 * np.pi / om_r
    t_pi = np.pi / om_r
    segments = [
        (t_prepare, "Laser", w_p),
        (t_wait, "Free", w_p),
        (t_pi_half, "MW", w_p),
        (half, "Free", w_p),
        (t_pi, "MW", w_p),
        (half, "Free", w_p),
        (t_pi_half, "MW", w_p),
        (t_readout, "Laser", ramsey_pump_rate),
    ]
    segment_results = np.array([])
    state = init_state
    ti = 0.0
    for duration, mode, pump in segments:
        tf, result = model(duration, state, b=b, om_r=om_r, om=om, w_p=pump,
                           ti=ti, mode=mode, progress_bar="OFF")
        segment_results = np.append(segment_results, result)
        state = result.states[-1]
        ti = tf
    populations = np.array(
        [np.concatenate([qt.expect(op, res.states) for res in segment_results]) for op in exp_ops]
    )
    pl_signal = np.sum(qt.expect(exp_ops[3] + exp_ops[4] + exp_ops[5], result.states))
    times = np.concatenate([res.times for res in segment_results])
    return echo_time, pl_signal, times, populations, segment_results


def spin_echo_sweep(t_prepare, t_wait, echo_times, t_readout, model, init_state, exp_ops,
                    b, om_r, w_p, om):
    """Spin echo, parallel over the echo-time sweep ``echo_times``."""
    run_single = partial(
        spin_echo, t_prepare=t_prepare, t_wait=t_wait, t_readout=t_readout,
        model=model, init_state=init_state, exp_ops=exp_ops, b=b, om_r=om_r, w_p=w_p, om=om,
    )
    outs = qt.loky_pmap(run_single, echo_times, progress_bar="tqdm",
                        map_kw={"fail_fast": True})  # type: ignore
    echo_out = np.array([item[0] for item in outs])     # type: ignore
    pl_signal = np.array([item[1] for item in outs])    # type: ignore
    # Pre-allocate + fill: per-sequence times (1-D) and populations (2-D (8, T)) are ragged when
    # the swept echo time crosses the dynamics_* t_bins switch at dt=5 us (1000 -> 5000 points).
    n = len(outs)
    times = np.empty(n, dtype=object)
    populations = np.empty(n, dtype=object)
    results = np.empty(n, dtype=object)
    for i, item in enumerate(outs):  # type: ignore
        times[i], populations[i], results[i] = item[2], item[3], item[4]
    return echo_out, pl_signal, times, populations, results
