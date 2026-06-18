"""Proof-of-concept: control-dependent dissipation (heating & cooling).

Validates the :mod:`core.open_control` framework on a qubit (the paper's TLS demo) and
applies it to the NV ground-state spin-1 triplet (Model 4). For each system it optimizes a
CRAB field to (i) maximize and (ii) minimize the final von Neumann entropy, and plots the
entropy trajectories and control fields.

Run:  uv run python examples/poc_open_control.py
Saves the figure into the first-group-meeting presentation's figures/ folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import core  # noqa: E402
from core.open_control import (  # noqa: E402
    ControlProblem,
    crab_envelope,
    optimize_crab,
    simulate,
    vn_entropy,
)

# Slide palette (navy / gold / cyan / teal)
NAVY, GOLD, CYAN, TEAL = "#1E2148", "#F2CB6E", "#5BC8D4", "#2E6B7E"

# Figures are saved to two places: the presentation folder and examples/Figures/.
PRES_FIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "Presentations" / "2026-06-19-first-group-meeting" / "figures"
)
EX_FIG_DIR = Path(__file__).resolve().parent / "Figures"


def save_figure(fig, name: str) -> None:
    """Save a figure into both the presentation figures/ and examples/Figures/."""
    for directory in (PRES_FIG_DIR, EX_FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        fig.savefig(path, dpi=150)
        print(f"saved figure -> {path}")


def run_case(problem: ControlProblem, n_coeffs: int, *, restarts: int, maxiter: int):
    """Optimize heating and cooling; return trajectories + fields for plotting."""
    t = problem.tlist
    tau, sigma = float(t[-1]), problem.sigma
    _, _, ent_free = simulate(problem, np.zeros(n_coeffs))

    out = {"t": t, "ent_free": ent_free, "S0": vn_entropy(problem.rho0)}
    for task in ("heating", "cooling"):
        coeffs, s_final = optimize_crab(
            problem, task, n_coeffs, n_restarts=restarts, amp_scale=2.0, maxiter=maxiter
        )
        _, _, ent = simulate(problem, coeffs)
        field = crab_envelope(t, coeffs, problem.freqs, tau, sigma)
        out[task] = {"ent": ent, "field": np.atleast_1d(field), "s_final": s_final}
    return out


def plot(qubit, nv):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    rows = [
        ("Qubit (TLS)", qubit, np.log(2), r"$\ln 2$"),
        ("NV ground triplet (Model 4)", nv, np.log(3), r"$\ln 3$"),
    ]
    for r, (name, d, smax, slabel) in enumerate(rows):
        ax_s, ax_f = axes[r, 0], axes[r, 1]
        t = d["t"]
        ax_s.axhline(smax, ls=":", c="grey", lw=1)
        ax_s.axhline(0.0, ls=":", c="grey", lw=1)
        ax_s.text(t[-1], smax, " " + slabel, va="center", c="grey", fontsize=8)
        ax_s.plot(t, d["ent_free"], c="grey", lw=1.5, label="free (no control)")
        ax_s.plot(t, d["heating"]["ent"], c=GOLD, lw=2.2, label="heating")
        ax_s.plot(t, d["cooling"]["ent"], c=CYAN, lw=2.2, label="cooling")
        ax_s.set_title(f"{name}: entropy", color=NAVY, fontsize=11)
        ax_s.set_xlabel("time")
        ax_s.set_ylabel(r"$S = -\mathrm{tr}(\rho\ln\rho)$")
        ax_s.legend(fontsize=8, loc="center right")

        ax_f.plot(t, d["heating"]["field"], c=GOLD, lw=1.8, label="heating field")
        ax_f.plot(t, d["cooling"]["field"], c=CYAN, lw=1.8, label="cooling field")
        ax_f.axhline(0.0, c="lightgrey", lw=0.8)
        ax_f.set_title(f"{name}: control field $\\epsilon(t)$", color=NAVY, fontsize=11)
        ax_f.set_xlabel("time")
        ax_f.set_ylabel(r"$\epsilon(t)$")
        ax_f.legend(fontsize=8, loc="upper right")
        for ax in (ax_s, ax_f):
            for s in ax.spines.values():
                s.set_color(TEAL)
    fig.suptitle(
        "Control-dependent dissipation: steering open-system entropy with CRAB",
        color=NAVY,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    print("== Qubit (validation) ==")
    qubit_p = core.open_control.build_qubit(
        n_times=90, t_final=6.0, n_freqs=4, temperature=0.5, bath_coupling=0.06
    )
    qubit = run_case(qubit_p, 4, restarts=4, maxiter=200)
    print(
        f"  heating S={qubit['heating']['s_final']:.4f}  "
        f"cooling S={qubit['cooling']['s_final']:.4f}  (S0={qubit['S0']:.4f})"
    )

    print("== NV ground triplet (proof of concept, Model 4) ==")
    nv_p = core.model4_ground_triplet(
        zfs=1.0,
        zeeman=0.3,
        temperature=0.5,
        bath_coupling=0.1,
        t_final=12.0,
        n_times=110,
        n_freqs=5,
    )
    nv = run_case(nv_p, 5, restarts=5, maxiter=240)
    print(
        f"  heating S={nv['heating']['s_final']:.4f}  "
        f"cooling S={nv['cooling']['s_final']:.4f}  (S0={nv['S0']:.4f})"
    )

    fig = plot(qubit, nv)
    save_figure(fig, "poc_open_control.png")


if __name__ == "__main__":
    main()
