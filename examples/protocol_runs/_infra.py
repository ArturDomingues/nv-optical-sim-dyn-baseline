"""Engine infrastructure: model registry, per-model state/operators, caching, figure saving."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

import core.plot_funcs as pf
from core.models import (
    ID_N15,
    N1, N2, N3, N4, N5, N6, N7, NC,
    H_doh_hf,
    H_dua_hf,
    H_no,
    dynamics_doh_hf,
    dynamics_dua_hf,
    dynamics_no,
)

# Static Hamiltonian (no drive) per model, for energy-level diagonalization.
STATIC_HAMILTONIANS: dict[str, Callable] = {
    "no-HF": lambda b: H_no(b, 0.0)[0],
    "Doherty": lambda b: H_doh_hf(b, 0.0)[0],
    "Duarte": lambda b: H_dua_hf(b, 0.0, 0.0)[0],
}
# Ground / excited sub-manifold indices: 7-level (no-HF) vs 14-level (NV(7) x N15(2)).
SUBSPACE_INDICES = {
    False: {"GS": [0, 1, 2], "ES": [3, 4, 5]},
    True: {"GS": [0, 1, 2, 3, 4, 5], "ES": [6, 7, 8, 9, 10, 11]},
}

REPO_ROOT = Path(__file__).resolve().parents[2]
PRES_FIG_DIR = REPO_ROOT / "Presentations" / "2026-06-19-first-group-meeting" / "figures"
EX_FIG_DIR = REPO_ROOT / "examples" / "Figures"
RAW_DIR = REPO_ROOT / "examples" / "RawData"

# Model registry: label -> (dynamics function, has_hyperfine)
MODELS: dict[str, tuple[Callable, bool]] = {
    "no-HF": (dynamics_no, False),
    "Doherty": (dynamics_doh_hf, True),
    "Duarte": (dynamics_dua_hf, True),
}

GS_PROJECTORS = (N1, N2, N3)  # ground-state populations


# --------------------------------------------------------------------------- #
# Caching + figure saving
# --------------------------------------------------------------------------- #
def run_or_load(dataset_name: str, compute_fn: Callable[[], dict], recompute: bool = False,
                data_dir: str | Path | None = None) -> dict:
    """Compute + cache, or load cached results, for a dataset.

    Saves the whole result dict as a single pickled `.npy`. With `recompute=True` (or when no
    cache exists) it runs `compute_fn()` and saves; otherwise it loads the cached arrays.
    """
    base = Path(data_dir) if data_dir else RAW_DIR / f"example_{dataset_name}"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{dataset_name}.npy"
    if recompute or not path.exists():
        data = compute_fn()
        np.save(path, np.array(data, dtype=object), allow_pickle=True)
        print(f"computed + saved -> {path}")
    else:
        # allow_pickle is safe here: we only ever load cache files this engine wrote itself
        # (under examples/RawData/), never untrusted input.
        data = np.load(path, allow_pickle=True).item()
        print(f"loaded cached -> {path}")
    return data


def save_figure(fig, name: str) -> None:
    """Save a figure into both the presentation figures/ and examples/Figures/."""
    for directory in (PRES_FIG_DIR, EX_FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"saved figure -> {path}")


def save_both_themes(build, name: str):
    """Render ``build()`` (returns a Figure) under BOTH themes and save each as
    ``<stem>_dark_theme<ext>`` and ``<stem>_light_theme<ext>`` (in both figure dirs). Returns
    the dark figure for inline display; closes the light one. Same curve hues, tuned per theme."""
    stem, ext = os.path.splitext(name)
    figs = {}
    for theme, setter in (("light", pf.use_light_theme), ("dark", pf.use_dark_theme)):
        setter()
        fig = build()
        for figdir in (PRES_FIG_DIR, EX_FIG_DIR):
            figdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(figdir / f"{stem}_{theme}_theme{ext}", dpi=150,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
        figs[theme] = fig
    pf.use_dark_theme(True)  # restore default theme
    plt.close(figs["light"])
    print(f"saved {stem}_dark_theme{ext} + {stem}_light_theme{ext} "
          f"(presentation figures/ + examples/Figures/)")
    return figs["dark"]


# --------------------------------------------------------------------------- #
# Per-model state / operators
# --------------------------------------------------------------------------- #
def measurement_operators(has_hyperfine: bool) -> list:
    """[n1..n7, n_coherence], tensored with the N-15 identity for hyperfine models."""
    ops = [N1, N2, N3, N4, N5, N6, N7, NC]
    return [op & ID_N15 for op in ops] if has_hyperfine else ops


def initial_state(has_hyperfine: bool):
    """Evenly populated ground triplet (mixed); tensored with N-15 for hyperfine models."""
    ground = N1 + N2 + N3
    return (ground & ID_N15) / 6 if has_hyperfine else ground / 3


def _energy_basis_labels(has_hf: bool):
    if has_hf:
        gs = [rf"$|g_{j}\rangle$" for j in range(6)]
        es = [rf"$|e_{j}\rangle$" for j in range(6)]
    else:
        gs = [r"$|0\rangle$", r"$|{+}1\rangle$", r"$|{-}1\rangle$"]
        es = [r"$|0\rangle_e$", r"$|{+}1\rangle_e$", r"$|{-}1\rangle_e$"]
    return gs, es
