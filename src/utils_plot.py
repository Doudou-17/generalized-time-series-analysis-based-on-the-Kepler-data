# -*- coding: utf-8 -*-
"""
Plot utilities
Default: keep scientific notation 1e6 on x-axis; top secondary axis shows time in "days".
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def plot_raw_sec_with_sci(time_sec, flux, out_path,
                          shift_to_t0: bool = True,
                          title: str = "Raw Light Curve (NaNs removed, NO QUALITY filter)") -> None:
    """
    Plot raw light curve in seconds with scientific notation (fixed at 1e6) on x-axis.
    Optionally shift start time to t0 = 0; add a secondary top axis in days.
    """
    t = np.asarray(time_sec).copy()
    y = np.asarray(flux)
    if shift_to_t0:
        t = t - t.min()
        xlabel = "Time since start (sec)"
        top_xlabel = "Time since start (day)"
    else:
        xlabel = "Time (sec)"
        top_xlabel = "Time (day)"
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, y, ".", ms=2, color="black")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useOffset=True)
    ax.xaxis.get_major_formatter().set_powerlimits((6, 6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Flux (e-/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Top axis: in days
    sec_to_day = lambda x: x / 86400.0
    day_to_sec = lambda d: d * 86400.0
    ax_top = ax.secondary_xaxis('top', functions=(sec_to_day, day_to_sec))
    ax_top.set_xlabel(top_xlabel)
    ax_top.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_segments_sec_with_sci(time1, flux1, time2, flux2, out_path,
                               shift_base: float | None = None) -> None:
    """
    Plot light curve split into two segments (before and after the largest gap),
    in seconds with scientific notation fixed at 1e6 on x-axis, and per-segment demeaned flux.
    Top axis shows time in days.
    """
    if shift_base is None:
        shift_base = float(min(np.min(time1), np.min(time2)))
    t1 = np.asarray(time1) - shift_base
    t2 = np.asarray(time2) - shift_base
    f1 = np.asarray(flux1)
    f2 = np.asarray(flux2)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t1, f1, ".", ms=2, label="Before gap")
    ax.plot(t2, f2, ".", ms=2, label="After gap")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useOffset=True)
    ax.xaxis.get_major_formatter().set_powerlimits((6, 6))
    ax.set_xlabel("Time since start (sec)")
    ax.set_ylabel("Flux - mean (e-/s)")
    ax.set_title("Light Curve Segments (per-segment demeaned)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    sec_to_day = lambda x: x / 86400.0
    day_to_sec = lambda d: d * 86400.0
    ax_top = ax.secondary_xaxis('top', functions=(sec_to_day, day_to_sec))
    ax_top.set_xlabel("Time since start (day)")
    ax_top.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_phase_folded(phi, flux, out_path, title="Phase-folded light curve",
                      show_two_cycles: bool = True,
                      binned: tuple | None = None):
    """
    Plot a phase-folded light curve.
    phi: folded phase (recommended range [-0.5,0.5) or [0,1)).
    binned: optionally provide (centers, y_med, y_err) to overlay binned median curve with error bars.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(phi, flux, ".", ms=2, alpha=0.6, label="data")
    if binned is not None:
        c, ym, ye = binned
        m = np.isfinite(ym)
        ax.errorbar(c[m], ym[m], ye[m], fmt="-", lw=2, capsize=2, label="binned median")
    ax.set_xlabel("Phase (cycle)")
    ax.set_ylabel("Flux (e-/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Optionally: duplicate 0–1 cycle to display two full cycles for better readability
    if show_two_cycles:
        # Duplicate data to +1 phase cycle
        ax.plot(np.asarray(phi) + 1.0, flux, ".", ms=2, alpha=0.3, color="C0")
        if binned is not None and np.any(np.isfinite(ym)):
            ax.plot(c[m] + 1.0, ym[m], "-", lw=2, color="C1")
        ax.set_xlim(min(phi.min(), 0.0), max(phi.max() + 1.0, 1.0))
    else:
        ax.set_xlim(phi.min(), phi.max())

    fig.tight_layout(); fig.savefig(out_path, dpi=300); plt.close(fig)
