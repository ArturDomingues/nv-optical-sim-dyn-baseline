"""ODMR / Ramsey / SNR plotters (fit functions live in core.utils)."""

import matplotlib.pyplot as plt
import numpy as np

from ..constants import D_GS, MU_E
from ..utils import damp_cos, expo, max_reg, stand
from .signals import fit_plot_data


def plot_odmr(mw_frequencies, pl_signal, field, rabi, ax=None, label=None,
              color=None, normalize=True, mark_resonances=True):
    """Plot an ODMR spectrum: (normalized) PL vs MW frequency, with the two Zeeman
    resonances ``D_GS ± MU_E*field`` marked. ``mw_frequencies``/``pl_signal`` from a
    ``cw_odmr``/``pulsed_odmr`` run.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))
    signal = max_reg(pl_signal) if normalize else pl_signal
    ax.plot(mw_frequencies, signal, lw=4, label=label, color=color)
    if mark_resonances:
        for res, name in ((D_GS - MU_E * field, r"$m_s=-1$"),
                          (D_GS + MU_E * field, r"$m_s=+1$")):
            ax.axvline(res, ls="dotted", color="grey", lw=1.5)
        ax.text(0.02, 0.02, r"$2\,\mu_e B = $" + f"{2 * MU_E * field:.1f} MHz",
                transform=ax.transAxes, fontsize=12, color="grey")
    ax.set_xlabel("Frequency (MHz)", fontsize=20)
    ax.set_ylabel("PL (a.u.)", fontsize=20)
    ax.set_title(f"ODMR\n B={field:.1f} G, $\\Omega_r$={rabi:.2f} MHz", fontsize=20)
    if label:
        ax.legend(fontsize=14)
    return ax


def plot_ramsey(free_times, signal, field, rabi, ax=None, label=None, has_hyperfine=False,
                fit=True, t2=1.5):
    """Plot Ramsey fringes: standardized signal vs free-evolution time ``tau``. Fits an
    exponential decay (no hyperfine) or a damped cosine (hyperfine), returning the fit params.
    Returns ``(ax, fitted_params | None)``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))
    y = stand(np.asarray(signal))
    fitted = None
    if fit:
        if has_hyperfine:
            drive = 2 * np.pi * MU_E * field
            guess = [1.0, drive, np.pi / 2, t2, (np.max(y) + np.min(y)) / 2]
            ax, fitted = fit_plot_data(
                damp_cos, free_times, y, ["A", "om", "phi", "tau", "D"], guess,
                ax=ax, label=label, xlabel=r"$\tau\ (\mu s)$", ylabel="PL (a.u.)")
        else:
            guess = [0.5, t2, (np.max(y) + np.min(y)) / 2]
            ax, fitted = fit_plot_data(
                expo, free_times, y, ["A", "tau", "D"], guess,
                ax=ax, label=label, xlabel=r"$\tau\ (\mu s)$", ylabel="PL (a.u.)")
    else:
        ax.plot(free_times, y, lw=3, label=label)
        ax.set_xlabel(r"$\tau\ (\mu s)$", fontsize=20)
        ax.set_ylabel("PL (a.u.)", fontsize=20)
        if label:
            ax.legend(fontsize=14)
    ax.set_title(f"Ramsey\n B={field:.1f} G, $\\Omega_r$={rabi:.2f} MHz", fontsize=20)
    return ax, fitted


def plot_snr(variable, snr, xlabel, ax=None, label=None, mark_optimum=True):
    """Plot the SNR ``(s0-s1)/sqrt((s0+s1)/2)`` against a swept variable; mark the optimum."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    variable = np.asarray(variable)
    snr = np.real(np.asarray(snr))
    ax.plot(variable, snr, lw=3, label=label)
    if mark_optimum:
        k = int(np.argmax(snr))
        ax.axvline(variable[k], ls="dotted", c="grey", lw=1.5)
        ax.plot(variable[k], snr[k], "o", c="crimson", ms=8,
                label=f"optimum @ {variable[k]:.3g}")
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("SNR", fontsize=18)
    ax.set_title("Readout SNR optimization", fontsize=18)
    ax.legend(fontsize=12)
    return ax
