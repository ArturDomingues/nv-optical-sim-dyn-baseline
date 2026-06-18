"""Figure theme. The SAME curve hues are used in both themes, only tuned in lightness so the
curves contrast with the background: lighter tones on the dark theme, darker tones on the
light theme. Importing the package applies the dark theme; toggle with use_dark_theme()/
use_light_theme().
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from cycler import cycler

_DARK_BG = "#15172B"  # very dark navy (matches the presentation)
_LIGHT_INK = "#1A1C2E"  # near-black navy ink for the light theme

# Base curve claudecode suggested hues (teal, gold, green, pink, purple, orange, blue, magenta).
# BASE_COLORS = [
#    "#2BB6C4", "#E0A93B", "#4FB477", "#E0607E",
#    "#7E63C4", "#D98E2B", "#5B8DEF", "#C44E9B",
# ]
# Base curve hues from matplotlib hex for the named colors
# ("dodgerblue", "chocolate", "darkgoldenrod", "mediumpurple","mediumseagreen", "lightskyblue", "magenta", "forestgreen")
# the same colors as the my masters dissertation

BASE_COLORS = [
    "#1E90FF",
    "#D2691E",
    "#B8860B",
    "#9370DB",
    "#3CB371",
    "#87CEFA",
    "#FF00FF",
    "#228B22",
]


def _tune(hex_color: str, amount: float) -> tuple[float, float, float]:
    """Lighten (amount>0, toward white) or darken (amount<0, toward black) a color."""
    r, g, b = mcolors.to_rgb(hex_color)
    if amount >= 0:
        f = amount
        return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)
    f = -amount
    return (r * (1 - f), g * (1 - f), b * (1 - f))


def _tuned_cycle(amount: float):
    return cycler(color=[_tune(c, amount) for c in BASE_COLORS])


def theme_colors() -> list:
    """Current theme's tuned curve colors (for plotters that set explicit colors)."""
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def theme_tune(color: str | tuple) -> tuple[float, float, float]:
    """Tune a named/explicit color for the active theme: lighten on the dark background,
    darken on the light one (same amounts as the curve cycle). For plotters that pin specific
    colors (e.g. energy levels) so they still contrast with the current background."""
    is_dark = mcolors.to_rgb(plt.rcParams["axes.facecolor"]) == mcolors.to_rgb(_DARK_BG)
    return _tune(mcolors.to_hex(color), 0.28 if is_dark else -0.22)


def use_dark_theme(enable: bool = True) -> None:
    """Dark theme: dark background, white axes/labels/titles, lighter curve tones."""
    if not enable:
        use_light_theme()
        return
    plt.rcParams.update(
        {
            "figure.facecolor": _DARK_BG,
            "axes.facecolor": _DARK_BG,
            "savefig.facecolor": _DARK_BG,
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "axes.titlecolor": "white",
            "figure.titlesize": "large",
            "figure.edgecolor": _DARK_BG,
            "text.color": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "legend.facecolor": _DARK_BG,
            "legend.edgecolor": "white",
            "legend.labelcolor": "white",
            "grid.color": "#A9A9A9",
            "axes.prop_cycle": _tuned_cycle(0.28),  # lighten curves for the dark bg
        }
    )


def use_light_theme() -> None:
    """Light theme: light background, dark axes/labels/titles, darker curve tones."""
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": _LIGHT_INK,
            "axes.labelcolor": _LIGHT_INK,
            "axes.titlecolor": _LIGHT_INK,
            "text.color": _LIGHT_INK,
            "xtick.color": _LIGHT_INK,
            "ytick.color": _LIGHT_INK,
            "legend.edgecolor": _LIGHT_INK,
            "legend.labelcolor": _LIGHT_INK,
            "axes.prop_cycle": _tuned_cycle(-0.22),  # darken curves for the light bg
        }
    )


use_dark_theme(True)
