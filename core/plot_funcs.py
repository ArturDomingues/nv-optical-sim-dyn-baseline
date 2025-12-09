import matplotlib.pyplot as plt
import scipy as scp
import numpy as np

def plot_fft(t, y, ax=None, label=None, unit='MHz'):
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
            Y = scp.fft.fftshift(scp.fft.fft(y))                         # normalised DFT
            f = scp.fft.fftshift(scp.fft.fftfreq(t.shape[-1],d=t[1]-t[0]))                   # matching frequencies

            # Find peaks for closer look to data (need to be reformulated 
            # when ran with multiple data use same axis)
            
            #floor = 1e-3 * f.max().real
            #peaks, abu = scp.signal.find_peaks(np.abs(Y), height=floor)
            ##print(f"peaks:{peaks}")
            ##print(f"The other output form signal:{abu}")
            #first_peak_freq = f[peaks[0]]
            #last_peak_freq  = f[peaks[-1]]
            # ---- 3. plot ------------------------------------------------------------
            if ax is None:
                fig, ax = plt.subplots(figsize=(14, 8))
            ax.plot(f, np.abs(Y.real)+0.5, label=f"Re{{{label}}}",lw=5)          # factor 2 for single-sided
            ax.plot(f, -np.abs(Y.imag)-0.5, label=f"Im{{{label}}}",lw=5)          # factor 2 for single-sided
            ax.set_xlabel(f"Frequency [{unit}]",fontsize=25)
            ax.set_ylabel(r"$|\mathcal{F}(f)|$",fontsize=25)
            ax.set_title("Amplitude spectrum (FFT)",fontsize=25)
            margin = 5
            ax.set_xlim(- margin, margin)
            ax.minorticks_on()
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",fontsize=20)
            return ax
        
## Plot of populations

def plot_popul(n_exp,times,tis,tfs):
    # Set the colors of each state
    colors = [
        "dodgerblue",
        "chocolate",
        "gold",
        "mediumpurple",
        "mediumseagreen",
        "lightskyblue",
        "magenta",
        "forestgreen",
    ]
    # Defines figure size
    plt.figure(figsize=(14, 8))
    # Plot the population of each state
    for i in range(len(n_exp) - 1):
        plt.plot(times, n_exp[i], label=f"$n_{i+1}$", color=colors[i])
    plt.plot(times, n_exp[-1], label="$n_c$", color=colors[-1])

    # Highlighting the times where the laser and microwaves are on and the metastable state depletion evolutions
    n = 0
    eps = 5*1e-2
    for i in zip(tis, tfs):
        if n % 3 == 0:
            plt.fill_betweenx(
                [np.min(n_exp) - eps, np.max(n_exp) + eps],
                i[0],
                i[1],
                color="palegreen",
                alpha=0.75 ** (int(n / 3) + 1),
                label=f"Laser ON #{int(n/3)+1}",
            )
        else:
            if (n - 1) % 3 == 0:
                plt.fill_betweenx(
                    [np.min(n_exp) - eps, np.max(n_exp) + eps],
                    i[0],
                    i[1],
                    color="lightblue",
                    alpha=0.75 ** (int(n / 3) + 1),
                    label=f"MS depl #{int(n/3)+1}",
                )
            else:
                plt.fill_betweenx(
                    [np.min(n_exp) - eps, np.max(n_exp) + eps],
                    i[0],
                    i[1],
                    color="orchid",
                    alpha=0.75 ** (int(n / 3) + 1),
                    label=f"MW ON #{int(n/3)+1}",
                )
        n += 1
    plt.ylabel(f"Population",fontsize=20)
    plt.xlabel(r"Time($\mu$s)",fontsize=20)

    # set the time limits you want to show in the plot
    t_lim = (-5 * eps, tfs[-1] + 5 * eps)
    plt.xlim(t_lim)
    if tfs[-1]<30:
        plt.xticks(np.arange(0,tfs[-1]+1,1))
    plt.minorticks_on()
    plt.ylim((np.min(n_exp) - eps, np.max(n_exp) + eps))
    plt.legend(ncol=2,bbox_to_anchor=(1.05, 1), loc="upper left",fontsize=16)
    plt.show()
    

def plot_popul_comp(n_exp_1,times_1,tis_1,tfs_1,n_exp_2,times_2,tis_2,tfs_2,name_1="M1",name_2="M2"):
    # Set the colors of each state
    colors = [
        "dodgerblue",
        "chocolate",
        "gold",
        "mediumpurple",
        "mediumseagreen",
        "lightskyblue",
        "magenta",
        "forestgreen",
    ]
    # Plot the population of each state
    for i in range(len(n_exp_1) - 1):
        plt.plot(times_1, n_exp_1[i], label=f"$n^{name_1}_{i+1}$", color=colors[i])
    plt.plot(times_1, n_exp_1[-1], label=f"$n^{name_1}_c$", color=colors[-1])
    for i in range(len(n_exp_2) - 1):
        plt.plot(times_2, n_exp_2[i], label=f"$n^{name_2}_{i+1}$", color=colors[i],ls='--')
    plt.plot(times_2, n_exp_2[-1], label=f"$n^{name_2}_c$", color=colors[-1],ls='--')
    # Highlighting the times where the laser and microwaves are on and the metastable state depletion evolutions
    n = 0
    eps = 5*1e-2
    for i in zip(tis_1, tfs_1):
        if n % 3 == 0:
            plt.fill_betweenx(
                [np.min(np.concatenate((n_exp_1,n_exp_2))) - eps, np.max(np.concatenate((n_exp_1,n_exp_2))) + eps],
                i[0],
                i[1],
                color="palegreen",
                alpha=0.5 ** (int(n / 3) + 1),
                label=f"Laser ON #{int(n/3)+1}",
            )
            t_laser=i[1]-i[0]
        else:
            if (n - 1) % 3 == 0:
                plt.fill_betweenx(
                    [np.min(np.concatenate((n_exp_1,n_exp_2))) - eps, np.max(np.concatenate((n_exp_1,n_exp_2))) + eps],
                    i[0],
                    i[1],
                    color="lightblue",
                    alpha=0.5 ** (int(n / 3) + 1),
                    label=f"MS depl #{int(n/3)+1}",
                )
                t_free=i[1]-i[0]
            else:
                plt.fill_betweenx(
                    [np.min(np.concatenate((n_exp_1,n_exp_2))) - eps, np.max(np.concatenate((n_exp_1,n_exp_2))) + eps],
                    i[0],
                    i[1],
                    color="orchid",
                    alpha=0.5 ** (int(n / 3) + 1),
                    label=f"MW ON #{int(n/3)+1}",
                )
                t_mw=i[1]-i[0]
        n += 1
    plt.ylabel(f"Population",fontsize=20)
    plt.xlabel(r"Time($\mu$s)",fontsize=20)
    try:
        name_1=name_1.replace("{","").replace("}","")
        name_2=name_2.replace("{","").replace("}","")
    except:
        pass
    plt.title(f"Comparisson between {name_1} and {name_2} evolution\n Laser:{t_laser:.2f} $\\mu$s, MW:{t_mw:.2f} $\\mu$s, Free:{t_free:.2f} $\\mu$s",fontsize=20)
    # set the time limits you want to show in the plot
    t_lim = (-5 * eps + np.min(np.array([tis_1[0],tis_2[0]])), np.max(np.array([tfs_1[-1],tfs_2[-1]])) + 5 * eps)
    plt.xlim(t_lim)
    plt.ylim((np.min(np.concatenate((n_exp_1,n_exp_2))) - eps, np.max(np.concatenate((n_exp_1,n_exp_2))) + eps))
    plt.minorticks_on()
    plt.legend(ncol=2,bbox_to_anchor=(1.05, 1), loc="upper left",fontsize=16)
    plt.show()