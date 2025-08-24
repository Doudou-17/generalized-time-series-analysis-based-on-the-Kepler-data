# === Module 7: Residual periodogram (Kurtz) ===
# This script performs a Kurtz recursive DFT scan on the residuals of a light curve.
# This script uses the src.utils_resid, src.utils_freq and src.utils_significance.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt

from src.utils_resid import compute_residual, summarize_timebase
from src.utils_freq import FreqBand, kurtz_scan, find_top_peaks, refine_peak_quadratic
from src.utils_significance import local_snr_from_amp

# ------------------- Manual parameters -------------------
# [User choice] Must be consistent with the data source used in Module 5 fit 
# Otherwise residuals will not match the same version.
DATA_SOURCE = 'after'    # 'full' | 'before' | 'after'
NPZ_PREP = "./output/cleaned_segments.npz"
OUT_DIR  = "./output"

# [User choice] Frequency bands for residual scan
# Frequency bands (Δν still determined automatically by T and S)
FREQ_BANDS = [
    (1.0,   20.0,   26.0),
    (20.0,  200.0, 30.0),
    (200.0, 8500.0, 32.0),
]
## Frequency bands for full segment.
###FREQ_BANDS = [
###    (0.8,   20.0,   8.0),
###    (20.0,  200.0, 10.0),
###    (200.0, 8500.0, 12.0),
###]
## Frequency bands for before segment.
###FREQ_BANDS = [
###    (1.0,   20.0,  26.0),
###    (20.0,  200.0, 30.0),
###    (200.0, 8500.0, 32.0),
###]
## Frequency bands for after segment.
###FREQ_BANDS = [
###    (1.0,   20.0,  26.0),
###    (20.0,  200.0, 30.0),
###    (200.0, 8500.0, 32.0),
###]

# Number of top residual peaks
TOP_K = 8
# Minimum separation: at least 5 grid bins between peaks, 
# to avoid splitting one main lobe into multiple spurious peaks.
MIN_SEP_BINS = 5
# Noise window width for SNR estimation (±50 Rayleigh units)
NOISE_WIN_RAYLEIGH = 50.0
# Exclude ±1 Rayleigh around the peak when estimating background noise
EXCLUDE_CORE_RAYLEIGH = 1.0
# Use MAD (robust) to estimate noise instead of mean/std
NOISE_STAT = "mad"

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------- Construct residuals -------------------
SUFFIX_MAP = {"before": "_before", "after": "_after", "full": ""}
suffix = SUFFIX_MAP.get(DATA_SOURCE, "_full")

# Default: use Module 5 results directly (no new fitting)
# NPZ_FIT = os.path.join(OUT_DIR, f"multisine_fit_results{suffix}.npz")
# If using refined results (e.g., Module 5 nonlinear adjustment), change accordingly:
# NPZ_FIT = os.path.join(OUT_DIR, f"multisine_fit_results{suffix}_refined.npz")
NPZ_FIT = os.path.join(OUT_DIR, f"multisine_fit_results{suffix}.npz")
if not os.path.exists(NPZ_FIT):
    raise FileNotFoundError(f"Fit results not found: {NPZ_FIT} (run Module 5 first)")

time_in, resid, meta = compute_residual(NPZ_PREP, NPZ_FIT, DATA_SOURCE)
T, rayleigh_uHz = summarize_timebase(time_in)
print(f"[M7] DATA_SOURCE={DATA_SOURCE} | T={T:.0f}s (~{T/86400:.2f} d), Rayleigh≈{rayleigh_uHz:.3f} μHz")

# -------- Automatically limit to Nyquist to avoid invalid high-frequency scans --------
if time_in.size > 1:
    dt_med = float(np.median(np.diff(np.sort(time_in))))
else:
    dt_med = np.inf
nyquist_uHz = (0.5 / max(dt_med, 1e-12)) * 1e6  # μHz
FREQ_BANDS_CAPPED = [(lo, min(hi, nyquist_uHz), S) for (lo, hi, S) in FREQ_BANDS if lo < min(hi, nyquist_uHz)]
if not FREQ_BANDS_CAPPED:
    raise ValueError(f"No valid bands after Nyquist cap ({nyquist_uHz:.2f} μHz). Check FREQ_BANDS.")
print(f"[M7] Nyquist≈{nyquist_uHz:.2f} μHz | bands -> {FREQ_BANDS_CAPPED}")

# Build frequency band objects (uniform spacing required)
bands = [FreqBand(fmin, fmax, S) for (fmin, fmax, S) in FREQ_BANDS_CAPPED]

# ------------------- Kurtz scan (residuals) -------------------
# Apply DFT to residuals. The result typically contains freq_uHz, power, amp, etc.
res = kurtz_scan(time_in, resid, bands, weights=None, verbose=True)

# ------------------- Peak detection and SNR -------------------
idx = find_top_peaks(res.freq_uHz, res.power, k=TOP_K, min_separation_bins=MIN_SEP_BINS)
rows = []
for i in idx:
    f_ref, p_ref = refine_peak_quadratic(res.freq_uHz, res.power, i)
    snr = local_snr_from_amp(
        res.freq_uHz, res.amp, f_ref, rayleigh_uHz,
        noise_win_rayleigh=NOISE_WIN_RAYLEIGH,
        exclude_core_rayleigh=EXCLUDE_CORE_RAYLEIGH,
        stat=NOISE_STAT
    )
    rows.append((i, res.freq_uHz[i], f_ref, p_ref, float(res.amp[i]), snr))

# ------------------- Save residual peak table -------------------
# Output discrete frequency, refined frequency, power, amplitude, and SNR.
PEAKS_RESID_TXT = os.path.join(OUT_DIR, f"kurtz_resid_top_peaks{suffix}.txt")
with open(PEAKS_RESID_TXT, "w") as f:
    f.write("# i  f_bin_uHz  f_refined_uHz  power         amp           SNR\n")
    for (i, fb, fr, ph, A, s) in rows:
        f.write(f"{i:6d}  {fb:12.6f}  {fr:12.6f}  {ph:12.6e}  {A:12.6e}  {s:8.3f}\n")
print(f"[M7] Saved residual peaks -> {PEAKS_RESID_TXT}")

# ------------------- Plot residual periodogram (full + zoom) -------------------
PLOT_RES_PSD  = os.path.join(OUT_DIR, f"kurtz_resid_periodogram_full{suffix}.png")
PLOT_RES_ZOOM = os.path.join(OUT_DIR, f"kurtz_resid_periodogram_zoom{suffix}.png")

# (1) Full-range periodogram
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(res.freq_uHz, res.power, lw=1.0)
ax.set_xlabel("Frequency (μHz)")
ax.set_ylabel("DFT Power (residual)")
ax.set_title(f"Residual periodogram{suffix}")
ax.grid(True, alpha=0.3)
for (_, _, fr, _, _, _) in rows:
    ax.axvline(fr, color="r", ls="--", alpha=0.6)
fig.tight_layout()
fig.savefig(PLOT_RES_PSD, dpi=200)
plt.close(fig)
print(f"[M7] Saved -> {PLOT_RES_PSD}")

# (2) Zoomed low-frequency periodogram
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(res.freq_uHz, res.power, lw=1.0)
ax.set_xlim(0, 20)
ax.set_xlabel("Frequency (μHz)")
ax.set_ylabel("DFT Power (residual)")
ax.set_title(f"Residual periodogram (zoom){suffix}")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT_RES_ZOOM, dpi=200)
plt.close(fig)
print(f"[M7] Saved -> {PLOT_RES_ZOOM}")
