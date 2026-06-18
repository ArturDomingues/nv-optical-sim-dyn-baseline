"""Signal helpers: FFT amplitude spectrum, curve fitting, Ramsey contrast normalization."""

import matplotlib.pyplot as plt
import numpy as np
import scipy as scp


def plot_fft(t, y, ax=None, label=None, unit="MHz"):
    """
    Plot the single-sided amplitude spectrum |Y(f)| of a real-valued signal y(t).

    Parameters
    ----------
    t : 1-D array
        Time axis [µs, ms, s …] - must be equally spaced.
    y : 1-D array
        Signal values at the same sampling points as `t`.
    ax : matplotlib Axes, optional
        If given, plot into this Axes; otherwise create a new figure.
    label : str, optional
        Legend label for this spectrum.
    unit : str
        Text for the x-axis (e.g. "MHz", "kHz", "Hz").
    """

    # ---- 2. FFT and frequency axis -----------------------------------------
    Y = scp.fft.fftshift(scp.fft.fft(y))  # normalised DFT
    f = scp.fft.fftshift(
        scp.fft.fftfreq(t.shape[-1], d=t[1] - t[0])
    )  # matching frequencies

    # Find peaks for closer look to data (need to be reformulated
    # when ran with multiple data use same axis)

    # floor = 1e-3 * f.max().real
    # peaks, abu = scp.signal.find_peaks(np.abs(Y), height=floor)
    ##print(f"peaks:{peaks}")
    ##print(f"The other output form signal:{abu}")
    # first_peak_freq = f[peaks[0]]
    # last_peak_freq  = f[peaks[-1]]
    # ---- 3. plot ------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(
        f, np.abs(Y.real) + 0.5, label=f"Re{{{label}}}", lw=5
    )  # factor 2 for single-sided
    ax.plot(
        f, -np.abs(Y.imag) - 0.5, label=f"Im{{{label}}}", lw=5
    )  # factor 2 for single-sided
    ax.set_xlabel(f"Frequency [{unit}]", fontsize=25)
    ax.set_ylabel(r"$|\mathcal{F}(f)|$", fontsize=25)
    ax.set_title("Amplitude spectrum (FFT)", fontsize=25)
    margin = 5
    ax.set_xlim(-margin, margin)
    ax.minorticks_on()
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=20)
    return ax


def fit_plot_data(func, x_data, y_data, param_names, guess, ax=None, label=None,
                  xlabel=None, ylabel=None, title=None, maxfev=20000):
    """Fit ``func`` to ``(x_data, y_data)`` with ``scipy.optimize.curve_fit`` and plot the
    data + fitted curve. Returns ``(ax, {param_name: (value, std_error)})``.
    """
    popt, pcov = scp.optimize.curve_fit(func, x_data, y_data, p0=guess, maxfev=maxfev)
    perr = np.sqrt(np.diag(pcov))
    fitted = {name: (val, err) for name, val, err in zip(param_names, popt, perr)}

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 8))
    ax.plot(x_data, y_data, "o", ms=4, alpha=0.6, label=label)
    x_dense = np.linspace(np.min(x_data), np.max(x_data), 1000)
    ax.plot(x_dense, func(x_dense, *popt), lw=3,
            label=None if label is None else f"{label} fit")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20)
    if title:
        ax.set_title(title, fontsize=20)
    ax.legend(fontsize=14)
    return ax, fitted


def ramsey_contrast_signal(pl_signal, s0, s1):
    """Contrast-normalize a Ramsey PL sweep: ``(S - s1) / (s0 - s1)`` (bright s0, dark s1)."""
    return (np.asarray(pl_signal) - s1) / (s0 - s1)
