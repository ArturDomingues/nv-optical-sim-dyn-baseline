"""Photoluminescence, energy-level and benchmark plotters."""

import matplotlib.pyplot as plt
import numpy as np

from .theme import theme_colors, theme_tune


def plot_photoluminescence(x_values, pl_curves, labels=None, ax=None,
                           xlabel="laser power (a.u.)"):
    """Plot photoluminescence curves (PL vs laser power or B-field). ``pl_curves`` is a list
    of 1-D arrays (one per spin preparation / model), ``x_values`` the shared x axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))
    for i, pl in enumerate(pl_curves):
        ax.plot(x_values, pl, lw=4, label=None if labels is None else labels[i])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel("PL (a.u.)", fontsize=20)
    ax.set_title("Photoluminescence", fontsize=20)
    if labels:
        ax.legend(fontsize=14)
    return ax


def plot_energy_levels(fields, energies, manifold="GS", ax=None,
                       colors=None, linestyles=None, alphas=None):
    """Plot eigen-energies vs magnetic field. ``energies`` has shape ``(n_fields, n_levels)``
    (e.g. from diagonalizing the GS or ES Hamiltonian block at each field).

    Optional per-level ``colors`` / ``linestyles`` / ``alphas`` (lists indexed by level) pin a
    consistent style so a given level keeps its color across full/zoom panels; the colors are
    theme-tuned (lightened on dark, darkened on light) via :func:`theme_tune`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))
    energies = np.asarray(energies)
    n_levels = energies.shape[1]
    for level in range(n_levels):
        color = theme_tune(colors[level]) if colors is not None else None
        ls = linestyles[level] if linestyles is not None else "-"
        alpha = alphas[level] if alphas is not None else 1.0
        ax.plot(fields, energies[:, level], lw=2.5, color=color, ls=ls, alpha=alpha)
    ax.set_xlabel("Magnetic field $B$ (G)", fontsize=16)
    ax.set_ylabel("Eigenenergy (MHz)", fontsize=16)
    ax.set_title(f"NV {manifold} energy levels", fontsize=18)
    ax.minorticks_on()
    return ax


def plot_pl_vs_b(fields, pl_curves, labels=None, ax=None):
    """Plot photoluminescence vs magnetic field for one or more curves (angles / models).
    ``pl_curves`` is a list of 1-D arrays sharing the ``fields`` x-axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    for i, pl in enumerate(pl_curves):
        ax.plot(fields, np.real(pl), lw=3, label=None if labels is None else labels[i])
    ax.set_xlabel("Magnetic field $B$ (G)", fontsize=18)
    ax.set_ylabel("PL (a.u.)", fontsize=18)
    ax.set_title("Photoluminescence vs magnetic field", fontsize=18)
    if labels:
        ax.legend(fontsize=11)
    ax.minorticks_on()
    return ax


def plot_benchmark(labels, seq_means, seq_stds, par_means, par_stds, ax=None):
    """Grouped bar chart of serial vs parallel runtime per model (with std error bars)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(labels))
    width = 0.38
    colors = theme_colors()
    edge = plt.rcParams["axes.edgecolor"]  # theme-aware error bars (not black on dark)
    ax.bar(x - width / 2, seq_means, width, yerr=seq_stds, capsize=4, ecolor=edge,
           label="serial", color=colors[0])
    ax.bar(x + width / 2, par_means, width, yerr=par_stds, capsize=4, ecolor=edge,
           label="parallel", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("runtime (s)", fontsize=16)
    ax.set_title("Runtime: serial vs parallel", fontsize=16)
    ax.legend(fontsize=12)
    return ax
