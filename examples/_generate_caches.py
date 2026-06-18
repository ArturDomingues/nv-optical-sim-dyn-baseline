"""Populate the shipped tutorial caches at SMOKE (tiny) resolution.

The Tutorial notebooks default to the FULL masters parameters (so `RECOMPUTE=True` reproduces
the masters exactly). This helper just confirms every dataset computes end-to-end and writes a
small cache so the shipped notebooks (`RECOMPUTE=False`) render without bugs. The sizes here
are deliberately tiny; set `RECOMPUTE=True` in a notebook for real resolution.

Dataset names MUST match the `run_or_load(...)` names used in the notebooks.

Run:  uv run python examples/_generate_caches.py
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from examples import protocol_runs as E  # noqa: E402

ALL_MODELS = ["no-HF", "Doherty", "Duarte"]


def _do(name: str, compute_fn) -> None:
    t0 = time.perf_counter()
    try:
        E.run_or_load(name, compute_fn, recompute=True)
        print(f"  [{time.perf_counter() - t0:6.1f}s] OK   {name}", flush=True)
    except Exception as exc:  # keep going if one dataset fails
        print(f"  [FAILED] {name}: {exc!r}", flush=True)


def main() -> None:
    print("Generating SMOKE-resolution caches ...", flush=True)

    for model in ALL_MODELS:
        _do(f"energy_levels_{model}",
            lambda m=model: E.compute_energy_levels(np.linspace(0.0, 1500.0, 200), model=m))

    _do("populations_no-HF", lambda: E.compute_populations(model_label="no-HF", mw_time=5.0))
    _do("populations_Doherty", lambda: E.compute_populations(model_label="Doherty", mw_time=5.0))
    _do("populations_pi_pulse", lambda: E.compute_populations(model_label="no-HF", mw_time=None))

    _do("photoluminescence", lambda: E.compute_photoluminescence(np.linspace(0.5, 5.0, 4)))

    _do("pl_vs_b", lambda: E.compute_pl_vs_b(
        ["no-HF"], fields=np.linspace(0.0, 1300.0, 6), angles_deg=[0.0, 10.0],
        laser_time=5.0, free_time=1.0))

    _do("snr", lambda: E.compute_snr(
        readtimes=np.linspace(0.5, 4.0, 4), readout_pumps=np.linspace(0.5, 5.5, 4),
        rabis=np.linspace(4.0, 22.0, 4), init_pumps=np.linspace(0.5, 5.5, 4)))

    _do("odmr", lambda: E.compute_odmr(ALL_MODELS, n_points=6))
    _do("odmr_pulsed", lambda: E.compute_odmr(["no-HF"], n_points=6, pulsed=True))

    _do("ramsey", lambda: E.compute_ramsey(ALL_MODELS, np.linspace(0.05, 5.0, 6)))

    _do("spin_echo", lambda: E.compute_spin_echo(["no-HF"], np.linspace(0.05, 5.0, 6)))

    _do("benchmark", lambda: E.compute_benchmark(ALL_MODELS, runs=1, repeats=1))

    _do("transition_rates", lambda: E.compute_transition_rate_comparison(n_points=6))
    _do("pure_vs_prepared", lambda: E.compute_pure_vs_prepared(n_points=6))

    # extra completions (smoke sizes; notebooks use full params on RECOMPUTE=True)
    _do("populations_rates", lambda: E.compute_populations_rates(
        "no-HF", k_indices=[0, 2], t_prepare=2.0, t_wait=0.5, mw_time=1.0, t_readout=2.0))
    for _w in ("magaletti", "hirose"):
        _do(f"pl_traces_{_w}", lambda w=_w: E.compute_pl_traces(
            np.linspace(0.5, 4.0, 3), weighting=w, laser_time=3.0))

    # ODMR vs B (no-HF only, sparse) + Rabi variants
    for _b in (0.0, 100.0, 500.0, 1000.0):
        _do(f"odmr_b{int(_b)}", lambda b=_b: E.compute_odmr(["no-HF"], field=b, n_points=8))
    t2 = 1.5
    for _tag, _om, _fields in (("piT2", np.pi / t2, (0.0, 100.0, 518.0)),
                               ("2piT2", 2 * np.pi / t2, (0.0, 500.0))):
        for _b in _fields:
            _do(f"odmr_{_tag}_b{int(_b)}",
                lambda b=_b, o=_om: E.compute_odmr(["no-HF"], field=b, rabi=o, n_points=8))

    # Ramsey vs B and Rabi (no-HF, sparse)
    for _tag, _om in (("piT2", np.pi / t2), ("2piT2", 2 * np.pi / t2), ("10piT2", 10 * np.pi / t2)):
        for _b in (0.0, 100.0, 510.0, 1020.0):
            _do(f"ramsey_{_tag}_b{int(_b)}",
                lambda b=_b, o=_om: E.compute_ramsey(
                    ["no-HF"], np.linspace(0.05, 5.0, 6), field=b, rabi=o))

    # K_S-vs-population overlay (Populations tutorial)
    _do("populations_ks_no-HF", lambda: E.compute_populations_ks_comparison(
        "no-HF", k_indices=[0, 2], t_prepare=2.0, t_wait=0.5, mw_time=1.0, t_readout=2.0))

    # PL pure-vs-prepared preparations (PL tutorial; no-HF + HF Doherty)
    for _m in ("no-HF", "Doherty"):
        _do(f"pl_preparations_{_m}", lambda m=_m: E.compute_pl_preparations(
            m, field=100.0, laser_time=3.0, free_time=0.5))

    # ODMR CW vs pulsed overlay (ODMR tutorial; no-HF only, sparse)
    for _b in (0.0, 150.0, 510.0):
        _do(f"odmr_cw_b{int(_b)}", lambda b=_b: E.compute_odmr(["no-HF"], field=b, n_points=8))
        _do(f"odmr_pulsed_b{int(_b)}",
            lambda b=_b: E.compute_odmr(["no-HF"], field=b, n_points=8, pulsed=True))

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
