# === Module 4 (Manual): Kurtz recursive DFT scan + spectral window ===
# This script performs a Kurtz recursive DFT scan on the light curve data.
# This script uses the src.utils_freq.
# This script includes frequency grid + Kurtz recursive DFT + peak finding / spectral window.
# Time unit: seconds (sec); Frequency unit: μHz
# Objectives:
#   1) All key parameters are “manually set”, allowing reproduction and tuning when reading papers;
#   2) Explicitly indicate the Kurtz formulas used in code comments;
#   3) Generate: full periodogram, zoomed-in low-frequency periodogram, spectral window (log amplitude), and top-K peaks table.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.utils_freq import (
    FreqBand, kurtz_scan, spectral_window,
    find_top_peaks, refine_peak_quadratic
)
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, LogFormatterMathtext
# ----------------------------- Manual parameters -----------------------------
NPZ_PREP   = "./output/cleaned_segments.npz"
OUT_DIR    = "./output"
# Data source (manual selection): 'full' | 'before' | 'after'
## [Adjustable] full: highest resolution, but window sidelobes more complex; 
## [Adjustable] before/after: lower resolution, but cleaner window.
## This directly affects Rayleigh resolution and window function structure.
DATA_SOURCE = 'after'

# Frequency bands (manual specification): (fmin_μHz, fmax_μHz, S=oversampling per Rayleigh)
## Larger S → smaller step size, slower computation, higher accuracy for peak interpolation.
## fmax should not exceed the Nyquist limit.
## Formula: Δν_μHz = 1e6 / (S * T_sec); spacing is uniform within each band (required for Kurtz recursion).
## High-frequency band covers a wide range (50–8000 μHz), so S=12 is chosen to ensure peak localization accuracy.
## Low-frequency band is narrow (0.58–5 μHz); theoretically S=4 is sufficient, but S=6 gives finer sampling.
## [Kurtz/DFT relevance] MIN_SEP_BINS × Δν corresponds to the physical minimum peak separation threshold.
## Current manual band configuration
FREQ_BANDS = [
    (1.0,   5.0,   10.0),   
    (5.0,   50.0,   10.0),   
    (50.0, 8000.0, 12.0),   
]
## Frequency bands for full segment.
#FREQ_BANDS = [
#    (0.58,   5.0,   6.0),   
#    (5.0,   50.0,   8.0),   
#    (50.0, 8000.0, 12.0),   
#]
## Frequency bands for before segment.
#FREQ_BANDS = [
#    (1.0,   5.0,   10.0),   
#    (5.0,   50.0,   10.0),   
#    (50.0, 8000.0, 12.0),   
#]
## Frequency bands for after segment.
#FREQ_BANDS = [
#    (1.0,   5.0,   10.0),   
#    (5.0,   50.0,   10.0),  
#    (50.0, 8000.0, 12.0),   
#]

# Peak selection (manual)
## Increase TOP_K when noise is strong or multiple peaks are densely packed.
## Save the top K peaks (default = 8). In this dissertation, we use K=8 to make the analysis.
TOP_K = 8            
## When peaks are very close, MIN_SEP_BINS can be reduced, 
## but beware of splitting one broad peak into spurious multiple peaks.
## Minimum separation between peaks in “grid bins” (default = 5).
MIN_SEP_BINS = 5       

# Detrending (manual): 0 = none, 1 = first-order polynomial, 2 = quadratic
## Purpose: suppress very slow drifts contaminating ultra-low frequencies, avoiding a dominant around 0 μHz peak.
DETREND_DEG = 0

# Spectral window display range (manual, optional)
## Focus on 5–3000 μHz, excluding 0-frequency main lobe (adjust as needed)
WINDOW_XLIM_UHZ = (5, 3000)   

# X-axis range for zoomed low-frequency plot (manual)
ZOOM_XLIM_UHZ = (0.0, 20.0)

# --------------------------- Helper functions (axis scaling) -------------------------

def uHz_to_day(x_uHz):
    # Prevent division by zero warnings; used only for axis conversion
    x = np.maximum(np.asarray(x_uHz, dtype=float), 1e-9)
    return 1.0 / ((x * 1e-6) * 86400.0)

def day_to_uHz(p_day):
    p = np.maximum(np.asarray(p_day, dtype=float), 1e-9)
    return (1.0 / (p * 86400.0)) * 1e6

# ------------------------------- Main procedure -------------------------------
# Load “full concatenated data with real gap preserved” (output of Module 2)
z = np.load(NPZ_PREP)
if DATA_SOURCE == 'before':
    time_in = z["time1"].astype(np.float64)
    flux_in_raw = z["flux1"].astype(np.float64)   # Already demeaned per segment
    suffix = "_before"
elif DATA_SOURCE == 'after':
    time_in = z["time2"].astype(np.float64)
    flux_in_raw = z["flux2"].astype(np.float64)   # Already demeaned per segment
    suffix = "_after"
else:  # 'full'
    time_in = z["time_full_sec"].astype(np.float64)                # Keep real gap
    flux_in_raw = z["flux_full_global_demean"].astype(np.float64)  # Global demeaned
    suffix = ""

# Output file naming
os.makedirs(OUT_DIR, exist_ok=True)
PEAKS_TXT = os.path.join(OUT_DIR, f"kurtz_top_peaks{suffix}.txt")
PLOT_PSD  = os.path.join(OUT_DIR, f"kurtz_periodogram_full{suffix}.png")
PLOT_ZOOM = os.path.join(OUT_DIR, f"kurtz_periodogram_zoom_0_20uHz{suffix}.png")
PLOT_WIN  = os.path.join(OUT_DIR, f"kurtz_spectral_window_logamp{suffix}.png")

# Print baseline and Rayleigh resolution (physical limit)
T = float(time_in.max() - time_in.min())  # sec
rayleigh_uHz = (1.0 / T) * 1e6
print(f"N={time_in.size}, T={T:.0f} s (~{T/86400:.2f} d), Rayleigh ≈ {rayleigh_uHz:.3f} μHz")
## [Kurtz/DFT relevance] Subsequent Δν and S settings are based on this value.

# (Optional) Detrending: reduce low-frequency power caused by very slow drifts
if DETREND_DEG in (1, 2):
    t0 = time_in.mean()
    tt = time_in - t0
    coef  = np.polyfit(tt, flux_in_raw, deg=DETREND_DEG)
    trend = np.polyval(coef, tt)
    flux_in = flux_in_raw - trend
else:
    flux_in = flux_in_raw
    print("Detrend: OFF")

# Construct manual frequency band objects
bands = [FreqBand(fmin, fmax, S) for (fmin, fmax, S) in FREQ_BANDS]
print("Manual bands:")
for b in bands:
    df_uHz = 1e6 / (b.oversample * T)
    print(f"  {b.fmin_uHz:.3f}–{b.fmax_uHz:.3f} μHz | S={b.oversample} | Δν={df_uHz:.6f} μHz")
## [Kurtz/DFT relevance] Kurtz recursion requires uniform spacing; Δν is constant within each band, satisfying this condition.
## Note: Different bands can use different S, leading to different Δν across bands, but constant within each band.

# =============== Kurtz recursive DFT scan (core) ==============
# [Key formulas from Kurtz (implemented in utils_freq._kurtz_segment, with comments in source)]:
# - Real/Imag DFT components (with weights):
#     FR(ν) = Σ_k w_k*[f_k cos(2πνt_k)]     ← Eq.[DFT-Real]
#     FI(ν) = Σ_k w_k*[f_k sin(2πνt_k)]     ← Eq.[DFT-Imag]
# - Frequency grid (uniform spacing within each band):
#     ν_j = ν_min + j·Δν                     ← Eq.[Grid]
# - Recursion (roll across frequency bins):
#     c_{j,k} = c_{j-1,k}·CΔ − s_{j-1,k}·SΔ  ← Eq.[Recursion-1]
#     s_{j,k} = s_{j-1,k}·CΔ + c_{j-1,k}·SΔ  ← Eq.[Recursion-2]
#   where CΔ=cos(2πΔνt_k), SΔ=sin(2πΔνt_k)
# - Power spectrum:
#     P(ν) = FR(ν)^2 + FI(ν)^2              ← Eq.[Power]
# - Amplitude (normalized by N or Σw):
#     A(ν) = (2/Σw_k) · sqrt(P)             ← Eq.[Amp]


res = kurtz_scan(time_in, flux_in, bands, weights=None, verbose=True)
## [Kurtz/DFT relevance] This is the core process; uniform spacing and recursive rolling ensure efficiency and consistency.

# =============== Spectral window (f(t) ≡ 1) ==============
# Using the same timestamps, set f_k ≡ 1, and apply the same Kurtz recursion to obtain the sampling response.
win = spectral_window(time_in, bands, weights=None, verbose=False)

# =============== Peak finding and sub-grid refinement ==============
## First, identify local maxima on the discrete grid;
idx = find_top_peaks(res.freq_uHz, res.power, k=TOP_K, min_separation_bins=MIN_SEP_BINS)
peaks = []
## Then apply 3-point parabolic interpolation for sub-grid refinement to improve frequency estimates.
for i in idx:
    f_hat, p_hat = refine_peak_quadratic(res.freq_uHz, res.power, i)  # Quadratic (3-point) interpolation
    peaks.append((i, res.freq_uHz[i], f_hat, p_hat))
## Parabolic interpolation can refine frequency precision to ~O(Δν/SNR), 
## where SNR = Signal-to-Noise Ratio.

with open(PEAKS_TXT, "w") as f:
    f.write("# i  f_bin_uHz  f_refined_uHz  power\n")
    for (i, fb, fr, ph) in peaks:
        f.write(f"{i:6d}  {fb:12.6f}  {fr:12.6f}  {ph:12.6e}\n")
print(f"Saved top-{TOP_K} peaks -> {PEAKS_TXT}")

# Start plotting.
def set_scientific_ticks(ax, *, logy=False, nbins_x=8, nbins_y=6):
    """Force both axes to use scientific notation (without offset text) and control the number of major ticks."""
    # X-axis
    fx = ScalarFormatter(useMathText=True)
    fx.set_scientific(True)
    fx.set_powerlimits((0, 0))   # Always use scientific notation
    fx.set_useOffset(False)      # Do not display offset in the top left corner (avoiding overlap)
    ax.xaxis.set_major_formatter(fx)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins_x))

    # Y-axis: linear or logarithmic
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=nbins_y))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))  # 显示为 10^n
    else:
        fy = ScalarFormatter(useMathText=True)
        fy.set_scientific(True)
        fy.set_powerlimits((0, 0))
        fy.set_useOffset(False)
        ax.yaxis.set_major_formatter(fy)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins_y))

# -------------------------- Plot: Full periodogram --------------------------
## Purpose: visualize the periodogram; top axis shows period scale for clarity.
fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)

ax.plot(res.freq_uHz, res.power, lw=1.0)
ax.set_xlabel("Frequency (μHz)")
ax.set_ylabel("DFT Power (FR^2+FI^2)")
ax.set_title(f"Kurtz recursive DFT ({DATA_SOURCE} data)")
ax.grid(True, alpha=0.3)

# Peak markers: draw only vertical lines (avoid cluttering with text annotations.
for (_, _, fr, _) in peaks:
    ax.axvline(fr, color="r", ls="--", lw=0.7, alpha=0.5)

# Enforce scientific notation on both axes (linear y-axis)
set_scientific_ticks(ax, logy=False, nbins_x=8, nbins_y=6)

fig.savefig(PLOT_PSD, dpi=200); plt.close(fig)
print(f"Saved periodogram -> {PLOT_PSD}")


# -------------------------- Plot: Zoomed low-frequency periodogram --------------------------
## Purpose: zoomed-in view of low-frequency range; top axis shows period scale for clarity.
fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)

ax.plot(res.freq_uHz, res.power, lw=1.0)
ax.set_xlim(*ZOOM_XLIM_UHZ)

ax.set_xlabel("Frequency (μHz)")
ax.set_ylabel("DFT Power (FR^2+FI^2)")
ax.set_title(f"Kurtz recursive DFT (zoom: {ZOOM_XLIM_UHZ[0]}–{ZOOM_XLIM_UHZ[1]} μHz)")
ax.grid(True, alpha=0.3)

set_scientific_ticks(ax, logy=False, nbins_x=8, nbins_y=6)

fig.savefig(PLOT_ZOOM, dpi=200); plt.close(fig)
print(f"Saved zoomed periodogram -> {PLOT_ZOOM}")

# -------------------------- Plot: Spectral window (log amplitude + optional xlim) ------------------
## Purpose: display the normalized spectral window in logarithmic scale, useful for examining sidelobes.
fig, ax = plt.subplots(figsize=(11, 3), constrained_layout=True)

wp_amp  = np.sqrt(win.power)
wp_norm = wp_amp / np.max(wp_amp)
ax.plot(win.freq_uHz, wp_norm, lw=1.0)  # Plot first, then set y-axis to log scale

ax.set_xlabel("Frequency (μHz)")
ax.set_ylabel("Window (norm, amplitude)")
ax.set_title("Spectral window (f(t)≡1)")
ax.grid(True, alpha=0.3)
if WINDOW_XLIM_UHZ is not None:
    ax.set_xlim(*WINDOW_XLIM_UHZ)

# Apply logarithmic y-axis with scientific notation; x-axis also in scientific notation
set_scientific_ticks(ax, logy=True, nbins_x=8, nbins_y=6)

fig.savefig(PLOT_WIN, dpi=200); plt.close(fig)
print(f"Saved spectral window -> {PLOT_WIN}")