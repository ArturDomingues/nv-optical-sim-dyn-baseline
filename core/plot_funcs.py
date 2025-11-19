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