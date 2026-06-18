"""Population-dynamics plotters (single run + two-model comparison)."""

import matplotlib.pyplot as plt
import numpy as np

from .theme import theme_colors


def plot_popul(n_exp, times, tis, tfs, show=True):
    # Theme-tuned curve colors (lighter on dark, darker on light)
    colors = theme_colors()
    # Defines figure size
    fig = plt.figure(figsize=(14, 8))
    # Plot the population of each state
    for i in range(len(n_exp) - 1):
        plt.plot(times, n_exp[i], label=f"$n_{i + 1}$", color=colors[i], lw=3)
    plt.plot(times, n_exp[-1], label="$n_c$", color=colors[-1], lw=3)

    # Highlighting the times where the laser and microwaves are on and the metastable state depletion evolutions
    n = 0
    eps = 5 * 1e-2
    for i in zip(tis, tfs):
        if n % 3 == 0:
            plt.fill_betweenx(
                [np.min(n_exp) - eps, np.max(n_exp) + eps],
                i[0],
                i[1],
                color="palegreen",
                alpha=0.5 ** (int(n / 3) + 1),
                label=f"Laser ON #{int(n / 3) + 1}",
            )
        else:
            if (n - 1) % 3 == 0:
                plt.fill_betweenx(
                    [np.min(n_exp) - eps, np.max(n_exp) + eps],
                    i[0],
                    i[1],
                    color="lightblue",
                    alpha=0.5 ** (int(n / 3) + 1),
                    label=f"MS depl #{int(n / 3) + 1}",
                )
            else:
                plt.fill_betweenx(
                    [np.min(n_exp) - eps, np.max(n_exp) + eps],
                    i[0],
                    i[1],
                    color="orchid",
                    alpha=0.5 ** (int(n / 3) + 1),
                    label=f"MW ON #{int(n / 3) + 1}",
                )
        n += 1
    plt.ylabel("Population", fontsize=20)
    plt.xlabel(r"Time($\mu$s)", fontsize=20)

    # set the time limits you want to show in the plot
    t_lim = (-5 * eps, tfs[-1] + 5 * eps)
    plt.xlim(t_lim)
    if tfs[-1] < 30:
        plt.xticks(np.arange(0, tfs[-1] + 1, 1))
    plt.minorticks_on()
    plt.ylim((np.min(n_exp) - eps, np.max(n_exp) + eps))
    plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)
    if show:
        plt.show()
    return fig


def plot_popul_comp(
    n_exp_1,
    times_1,
    tis_1,
    tfs_1,
    n_exp_2,
    times_2,
    tis_2,
    tfs_2,
    name_1="M1",
    name_2="M2",
    show=True,
):
    # Theme-tuned curve colors (lighter on dark, darker on light)
    colors = theme_colors()
    fig = plt.figure(figsize=(14, 8))
    # Plot the population of each state
    for i in range(len(n_exp_1) - 1):
        plt.plot(times_1, n_exp_1[i], label=f"$n^{{{name_1}}}_{{{i + 1}}}$",
                 color=colors[i], lw=3)
    plt.plot(times_1, n_exp_1[-1], label=f"$n^{{{name_1}}}_c$", color=colors[-1], lw=3)
    for i in range(len(n_exp_2) - 1):
        plt.plot(times_2, n_exp_2[i], label=f"$n^{{{name_2}}}_{{{i + 1}}}$",
                 color=colors[i], ls="--", lw=3)
    plt.plot(times_2, n_exp_2[-1], label=f"$n^{{{name_2}}}_c$", color=colors[-1], ls="--", lw=3)
    # Highlighting the times where the laser and microwaves are on and the metastable state depletion evolutions
    n = 0
    eps = 5 * 1e-2
    for i in zip(tis_1, tfs_1):
        if n % 3 == 0:
            plt.fill_betweenx(
                [
                    np.min(np.concatenate((n_exp_1, n_exp_2))) - eps,
                    np.max(np.concatenate((n_exp_1, n_exp_2))) + eps,
                ],
                i[0],
                i[1],
                color="palegreen",
                alpha=0.5 ** (int(n / 3) + 1),
                label=f"Laser ON #{int(n / 3) + 1}",
            )
            t_laser = i[1] - i[0]
        else:
            if (n - 1) % 3 == 0:
                plt.fill_betweenx(
                    [
                        np.min(np.concatenate((n_exp_1, n_exp_2))) - eps,
                        np.max(np.concatenate((n_exp_1, n_exp_2))) + eps,
                    ],
                    i[0],
                    i[1],
                    color="lightblue",
                    alpha=0.5 ** (int(n / 3) + 1),
                    label=f"MS depl #{int(n / 3) + 1}",
                )
                t_free = i[1] - i[0]
            else:
                plt.fill_betweenx(
                    [
                        np.min(np.concatenate((n_exp_1, n_exp_2))) - eps,
                        np.max(np.concatenate((n_exp_1, n_exp_2))) + eps,
                    ],
                    i[0],
                    i[1],
                    color="orchid",
                    alpha=0.5 ** (int(n / 3) + 1),
                    label=f"MW ON #{int(n / 3) + 1}",
                )
                t_mw = i[1] - i[0]
        n += 1
    plt.ylabel("Population", fontsize=20)
    plt.xlabel(r"Time($\mu$s)", fontsize=20)
    try:
        name_1 = name_1.replace("{", "").replace("}", "")
        name_2 = name_2.replace("{", "").replace("}", "")
    except Exception:
        pass
    plt.title(
        f"Comparisson between {name_1} and {name_2} evolution\n Laser:{t_laser:.2f} $\\mu$s, MW:{t_mw:.2f} $\\mu$s, Free:{t_free:.2f} $\\mu$s",
        fontsize=20,
    )
    # set the time limits you want to show in the plot
    t_lim = (
        -5 * eps + np.min(np.array([tis_1[0], tis_2[0]])),
        np.max(np.array([tfs_1[-1], tfs_2[-1]])) + 5 * eps,
    )
    plt.xlim(t_lim)
    plt.ylim(
        (
            np.min(np.concatenate((n_exp_1, n_exp_2))) - eps,
            np.max(np.concatenate((n_exp_1, n_exp_2))) + eps,
        )
    )
    plt.minorticks_on()
    plt.legend(ncol=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)
    if show:
        plt.show()
    return fig
