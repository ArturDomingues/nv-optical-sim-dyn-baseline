"""Plot wrappers: build a figure, dual-theme save, return the dark figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import core.plot_funcs as pf
from core import protocols as P

from ._infra import MODELS, save_both_themes

# Masters per-level palette (plots.ipynb cells 34/36/38): GS uses the first 3 hues, ES the next 3.
_LEVEL_PALETTE = ["dodgerblue", "chocolate", "darkgoldenrod", "mediumpurple",
                  "mediumseagreen", "lightskyblue", "magenta", "forestgreen"]


def _energy_level_styles(has_hf: bool, manifold: str):
    """Per-level (colors, linestyles, alphas) for an energy panel, matching the masters scheme.

    no-HF: 3 distinct solid hues per manifold. HF: 6 levels = 3 hyperfine pairs; the pair shares a
    hue, spin-down (even index) solid at alpha 0.55, spin-up (odd) dashed at alpha 1.0.
    """
    base = _LEVEL_PALETTE[0:3] if manifold == "GS" else _LEVEL_PALETTE[3:6]
    if not has_hf:
        return list(base), ["-", "-", "-"], [1.0, 1.0, 1.0]
    colors, lss, alphas = [], [], []
    for i in range(6):  # 3 pairs (down, up)
        colors.append(base[i // 2])
        lss.append("-" if i % 2 == 0 else "--")
        alphas.append(0.55 if i % 2 == 0 else 1.0)
    return colors, lss, alphas


def plot_energy_levels_result(data: dict, save_as: str | None = "energy_levels.png"):
    f, eg, ee = data["fields"], data["energies_gs"], data["energies_es"]
    zoom_gs, zoom_es = data.get("zoom_gs"), data.get("zoom_es")
    has_hf = MODELS.get(data.get("model", "no-HF"), (None, False))[1]
    gs_c, gs_ls, gs_a = _energy_level_styles(has_hf, "GS")
    es_c, es_ls, es_a = _energy_level_styles(has_hf, "ES")

    def build():
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        for ax, energies, title, (c, ls, a) in (
            (axes[0, 0], eg, "Ground states — full range", (gs_c, gs_ls, gs_a)),
            (axes[0, 1], eg, "Ground states — zoom (GSLAC)", (gs_c, gs_ls, gs_a)),
            (axes[1, 0], ee, "Excited states — full range", (es_c, es_ls, es_a)),
            (axes[1, 1], ee, "Excited states — zoom (ESLAC / ISC region)", (es_c, es_ls, es_a)),
        ):
            pf.plot_energy_levels(f, energies, manifold="", ax=ax,
                                  colors=c, linestyles=ls, alphas=a)
            ax.set_title(title, fontsize=14)
        for ax, zoom in ((axes[0, 1], zoom_gs), (axes[1, 1], zoom_es)):
            if not zoom:
                continue
            ax.set_xlim(zoom[0] - zoom[1], zoom[0] + zoom[1])
            if len(zoom) >= 3:  # tight y-window about the crossing (~0 MHz)
                ax.set_ylim(-zoom[2], zoom[2])
        fig.suptitle(f"{data.get('model', '')} NV energy levels vs B", fontsize=17)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_energy_levels_angle_result(data: dict, angle_deg: float,
                                    save_as: str | None = "energy_levels_angle.png"):
    """Full-range GS|ES eigen-energies vs field for a single B orientation (``angle_deg`` =
    angle of B to the NV axis). Used by the angle-sweep section of the Energy-Levels tutorial."""
    f, eg, ee = data["fields"], data["energies_gs"], data["energies_es"]
    has_hf = MODELS.get(data.get("model", "no-HF"), (None, False))[1]
    gs_c, gs_ls, gs_a = _energy_level_styles(has_hf, "GS")
    es_c, es_ls, es_a = _energy_level_styles(has_hf, "ES")

    def build():
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        pf.plot_energy_levels(f, eg, manifold="", ax=axes[0],
                              colors=gs_c, linestyles=gs_ls, alphas=gs_a)
        pf.plot_energy_levels(f, ee, manifold="", ax=axes[1],
                              colors=es_c, linestyles=es_ls, alphas=es_a)
        axes[0].set_title("Ground states", fontsize=14)
        axes[1].set_title("Excited states", fontsize=14)
        fig.suptitle(f"{data.get('model', '')} NV energy levels vs B "
                     rf"($\theta={angle_deg:g}^\circ$)", fontsize=17)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_populations_result(data: dict, save_as: str | None = "populations.png"):
    def build():
        return pf.plot_popul(data["populations"], data["times"], data["tis"], data["tfs"],
                             show=False)
    return save_both_themes(build, save_as) if save_as else build()


def plot_populations_comparison_result(data_a: dict, data_b: dict, name_a="7", name_b="14",
                                       save_as: str | None = "populations_comparison.png"):
    def build():
        return pf.plot_popul_comp(
            data_a["populations"], data_a["times"], data_a["tis"], data_a["tfs"],
            data_b["populations"], data_b["times"], data_b["tis"], data_b["tfs"],
            name_1=name_a, name_2=name_b, show=False)
    return save_both_themes(build, save_as) if save_as else build()


def plot_populations_rates_result(data: dict, save_as: str | None = "populations_rates.png"):
    def build():
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, pl in data["curves"].items():
            ax.plot(data["times"], np.real(pl), lw=3, label=label)
        ax.set_xlabel(r"Time ($\mu$s)", fontsize=18)
        ax.set_ylabel("PL (a.u.)", fontsize=18)
        ax.set_title(f"Readout vs transition-rate set K_S ({data['model']})", fontsize=15)
        ax.legend(fontsize=11)
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_populations_ks_result(data: dict, save_as: str | None = "populations_ks.png"):
    """Overlay population dynamics for every K_S set; the highlighted set is drawn in the per-state
    palette, the others greyed (masters plots.ipynb Cell 96 ``plot_regime``)."""
    times, per_k, hk = data["times"], data["per_k"], data["highlight_k"]
    markers = ["o", "s", "^", "v", "P", "X", "D", "*"]
    st = ["1", "2", "3", "4", "5", "6", "7", "c"]
    every = max(len(times) // 12, 1)

    def build():
        fig, ax = plt.subplots(figsize=(12, 7))
        for k, pops in per_k.items():
            mk = markers[k % len(markers)]
            for i, trace in enumerate(np.real(pops)):
                if k == hk:
                    ax.plot(times, trace, color=pf.theme_tune(_LEVEL_PALETTE[i]), lw=4,
                            marker=mk, markevery=every, ms=7,
                            label=rf"$K_{{{k + 1}}}$, $n_{{{st[i]}}}$")
                else:
                    ax.plot(times, trace, color="gray", lw=2.5, alpha=0.4, marker=mk,
                            markevery=every, ms=6, label=(rf"$K_{{{k + 1}}}$" if i == 0 else None))
        ax.set_xlabel(r"Time ($\mu$s)", fontsize=18)
        ax.set_ylabel("Population", fontsize=18)
        ax.set_title(f"Populations vs transition-rate set K_S ({data['model']}, "
                     rf"highlight $K_{{{hk + 1}}}$)", fontsize=14)
        ax.legend(ncol=2, fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_pl_result(data: dict, save_as: str | None = "photoluminescence.png"):
    def build():
        fig, ax = plt.subplots(figsize=(10, 6))
        pf.plot_photoluminescence(data["powers"], list(data["pl"]),
                                  labels=list(data["labels"]), ax=ax)
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_pl_traces_result(data: dict, save_as: str | None = "pl_traces.png"):
    def build():
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, pl in data["curves"].items():
            ax.plot(data["times"], np.real(pl), lw=3, label=label)
        ax.set_xlabel(r"Time ($\mu$s)", fontsize=18)
        ax.set_ylabel("PL (a.u.)", fontsize=18)
        ax.set_title(f"{data['weighting'].title()} PL traces "
                     f"(B={data['field']:.0f} G, init {data['init']})", fontsize=15)
        ax.legend(fontsize=10)
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_pl_preparations_result(data: dict, save_as: str | None = "pl_preparations.png"):
    """PL readout per spin preparation: pure (dashed) overlaid on prepared (solid), same per-state
    color (masters pure-vs-prepared; dissertation Fig 19)."""
    t, labels = data["times"], data["labels"]

    def build():
        fig, ax = plt.subplots(figsize=(12, 7))
        for i, lbl in enumerate(labels):
            color = pf.theme_tune(_LEVEL_PALETTE[i])
            ax.plot(t, data["prepared"][lbl], color=color, lw=3, label=f"prepared {lbl}")
            ax.plot(t, data["pure"][lbl], color=color, lw=3, ls="--", label=f"pure {lbl}")
        ax.set_xlabel(r"Time ($\mu$s)", fontsize=18)
        ax.set_ylabel("PL (a.u.)", fontsize=18)
        ax.set_title(f"PL: pure vs prepared ({data['model']}, B={data['field']:.0f} G, "
                     rf"$\Omega_r$={data['rabi']:.2f} MHz)", fontsize=14)
        ax.legend(fontsize=11)
        fig.tight_layout()
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_pl_contrast_result(data: dict, save_as: str | None = "pl_contrast.png"):
    """Prepared-readout contrast: PL(|0>), PL(|1>) and the normalized contrast (PL0-PL1)/max
    (masters contrast figures at the end of plots.ipynb; dissertation Fig 20)."""
    t, labels = data["times"], data["labels"]
    pl0, pl1 = data["prepared"][labels[0]], data["prepared"][labels[2]]
    norm = np.max(pl0) if np.max(pl0) != 0 else 1.0

    def build():
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(t, pl0, color=pf.theme_tune(_LEVEL_PALETTE[0]), lw=3, label=f"prepared {labels[0]}")
        ax.plot(t, pl1, color=pf.theme_tune(_LEVEL_PALETTE[1]), lw=3, label=f"prepared {labels[2]}")
        ax.plot(t, pl0 / norm - pl1 / norm, color=pf.theme_tune("forestgreen"), lw=3,
                label="contrast $(PL_0-PL_1)/\\max$")
        ax.set_xlabel(r"Time ($\mu$s)", fontsize=18)
        ax.set_ylabel("PL (a.u.)", fontsize=18)
        ax.set_title(f"PL readout contrast ({data['model']}, B={data['field']:.0f} G)", fontsize=14)
        ax.legend(fontsize=11)
        fig.tight_layout()
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_odmr_result(data: dict, save_as: str | None = "odmr.png"):
    def build():
        fig, ax = plt.subplots(figsize=(11, 6))
        offset = 0.0
        for label, signal in data["spectra"].items():
            pf.plot_odmr(data["mw_frequencies"], signal + offset, data["field"], P.odmr_rabi,
                         ax=ax, label=label, mark_resonances=(offset == 0.0))
            offset += 0.05
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_odmr_cw_vs_pulsed_result(cw_data: dict, pulsed_data: dict,
                                  save_as: str | None = "odmr_cw_vs_pulsed.png"):
    """Overlay CW (dashed) vs pulsed (solid) ODMR per model, normalized PL vs MW frequency
    (masters CW-vs-pulsed comparison; dissertation Figs 26-28)."""
    field = cw_data["field"]

    def _norm(sig):
        sig = np.real(np.asarray(sig))
        m = np.max(sig)
        return sig / m if m != 0 else sig

    def build():
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = pf.theme_colors()
        for i, label in enumerate(cw_data["spectra"]):
            c = colors[i % len(colors)]
            ax.plot(cw_data["mw_frequencies"], _norm(cw_data["spectra"][label]),
                    color=c, lw=3, ls="--", label=f"{label} CW")
            if label in pulsed_data["spectra"]:
                ax.plot(pulsed_data["mw_frequencies"], _norm(pulsed_data["spectra"][label]),
                        color=c, lw=3, label=f"{label} pulsed")
        for res in (P.D_GS - P.MU_E * field, P.D_GS + P.MU_E * field):
            ax.axvline(res, ls="dotted", color="grey", lw=1.2)
        ax.set_xlabel("Frequency (MHz)", fontsize=18)
        ax.set_ylabel("PL (a.u.)", fontsize=18)
        ax.set_title(f"ODMR: CW (dashed) vs pulsed (solid), B={field:.0f} G", fontsize=14)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_ramsey_result(data: dict, save_as: str | None = "ramsey.png"):
    fits: dict = {}

    def build():
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, entry in data["signals"].items():
            _, fit = pf.plot_ramsey(data["free_times"], entry["signal"], data["field"],
                                    data["rabi"], ax=ax, label=label,
                                    has_hyperfine=entry["has_hyperfine"])
            fits[label] = fit
        return fig
    fig = save_both_themes(build, save_as) if save_as else build()
    return fig, fits


def plot_pl_vs_b_result(data: dict, save_as: str | None = "pl_vs_b.png"):
    def build():
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, per_angle in data["curves"].items():
            for angle_key, pl in per_angle.items():
                pf.plot_pl_vs_b(data["fields"], [pl], labels=[f"{label} {angle_key}"], ax=ax)
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_snr_result(data: dict, save_as: str | None = "snr.png"):
    def build():
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        pf.plot_snr(data["readtimes"], data["snr_readtime"], "readout time ($\\mu$s)", ax=axes[0, 0])
        pf.plot_snr(data["readout_pumps"], data["snr_readout"], "readout pump (MHz)", ax=axes[0, 1])
        pf.plot_snr(data["rabis"], data["snr_rabi"], "$\\Omega_r$ (MHz)", ax=axes[1, 0])
        pf.plot_snr(data["init_pumps"], data["snr_init"], "init pump (MHz)", ax=axes[1, 1])
        fig.tight_layout()
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_benchmark_result(data: dict, save_as: str | None = "benchmark.png"):
    labels = data["labels"]
    r = data["results"]

    def build():
        fig, ax = plt.subplots(figsize=(9, 6))
        pf.plot_benchmark(
            labels,
            [r[m]["seq_mean"] for m in labels], [r[m]["seq_std"] for m in labels],
            [r[m]["par_mean"] for m in labels], [r[m]["par_std"] for m in labels], ax=ax)
        return fig
    return save_both_themes(build, save_as) if save_as else build()


def plot_spin_echo_result(data: dict, save_as: str | None = "spin_echo.png"):
    def build():
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, signal in data["signals"].items():
            ax.plot(data["echo_times"], signal, lw=3, label=label)
        ax.set_xlabel(r"echo time $2\tau$ ($\mu$s)", fontsize=18)
        ax.set_ylabel("PL (a.u.)", fontsize=18)
        ax.set_title(f"Spin echo (B={data['field']:.0f} G)", fontsize=18)
        ax.legend(fontsize=12)
        return fig
    return save_both_themes(build, save_as) if save_as else build()
