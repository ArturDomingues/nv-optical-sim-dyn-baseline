"""Plotting helpers for the NV simulations.

The SAME curve hues are used in both the dark and light themes (only tuned in lightness);
importing this package applies the dark theme. Split into ``theme`` (colors/rcParams),
``signals`` (FFT/fit/contrast), ``populations``, ``spectra`` (ODMR/Ramsey/SNR) and ``levels``
(PL/energy-levels/benchmark); the full public API is re-exported here so
``import core.plot_funcs as pf`` keeps working unchanged.
"""

from .theme import (  # theme import first -> applies the dark theme on import
    BASE_COLORS,
    theme_colors,
    theme_tune,
    use_dark_theme,
    use_light_theme,
)
from .levels import (
    plot_benchmark,
    plot_energy_levels,
    plot_photoluminescence,
    plot_pl_vs_b,
)
from .populations import plot_popul, plot_popul_comp
from .signals import fit_plot_data, plot_fft, ramsey_contrast_signal
from .spectra import plot_odmr, plot_ramsey, plot_snr

__all__ = [
    "BASE_COLORS",
    "fit_plot_data",
    "plot_benchmark",
    "plot_energy_levels",
    "plot_fft",
    "plot_odmr",
    "plot_photoluminescence",
    "plot_pl_vs_b",
    "plot_popul",
    "plot_popul_comp",
    "plot_ramsey",
    "plot_snr",
    "ramsey_contrast_signal",
    "theme_colors",
    "theme_tune",
    "use_dark_theme",
    "use_light_theme",
]
