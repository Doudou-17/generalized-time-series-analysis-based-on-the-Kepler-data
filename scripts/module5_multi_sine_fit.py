# === Module 5 (Manual-aware): Multi-sine fit using peaks from selected segment ===
# This script performs a multi_sine fit on the light curve data.
# This script uses the src.utils_fit.
# Units: time = sec, freq = μHz
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from src.utils_fit import fit_multi_sine

# ----------------------------- Manual parameters (others are automatic) -----------------------------
# Choose the data source for fitting: 'full' | 'before' | 'after'
# - 'full'   : use time_full_sec + flux_full_global_demean, read kurtz_top_peaks.txt
# - 'before' : use time1 + flux1, read kurtz_top_peaks_before.txt
# - 'after'  : use time2 + flux2, read kurtz_top_peaks_after.txt
DATA_SOURCE = 'after'
## [User choice] Must be consistent with the data source used in module 4 when generating peak tables.

# Number of top frequencies from the peak table to use for fitting (manual)
TOP_M = 5 # Take top M peaks, default = 5.
## For "after" data, using 8 often gives better results
## For "full" data, using 8 often gives better results
## In this dissertation, we use TOP_M=5 in all cases.

# Whether to detrend the selected data before fitting (0 = none, 1 = linear, 2 = quadratic)
# Purpose: consistent with Module 4 settings, suppress very slow drifts;
# If Module 4 did not detrend, set this to 0.
DETREND_DEG = 0
## [User choice] Recommended to align with Module 4 settings to ensure consistency.
# File paths
NPZ_PREP  = "./output/cleaned_segments.npz"   # Output of Module 2 (contains three datasets)
OUT_DIR   = "./output"
os.makedirs(OUT_DIR, exist_ok=True)
# -------------------------------------------------------------------------------

# --------------------------- Select arrays & peak table by data source ---------------------------
z = np.load(NPZ_PREP)

if DATA_SOURCE == 'before':
    time_in = z["time1"].astype(np.float64)      # sec
    flux_in = z["flux1"].astype(np.float64)      # e-/s (demeaned per segment)
    suffix  = "_before"
elif DATA_SOURCE == 'after':
    time_in = z["time2"].astype(np.float64)      # sec
    flux_in = z["flux2"].astype(np.float64)      # e-/s (demeaned per segment)
    suffix  = "_after"
else:  # 'full'
    time_in = z["time_full_sec"].astype(np.float64)                 # Keep real gap
    flux_in = z["flux_full_global_demean"].astype(np.float64)       # Global demeaned
    suffix  = ""

PEAKS_TXT = os.path.join(OUT_DIR, f"kurtz_top_peaks{suffix}.txt")
OUT_NPZ   = os.path.join(OUT_DIR, f"multisine_fit_results{suffix}.npz")

print(f"[Module5] DATA_SOURCE = {DATA_SOURCE}")
print(f"[Module5] Read peaks from: {PEAKS_TXT}")
print(f"[Module5] Save results to: {OUT_NPZ}")

# ------------------------------ Read peak table (automatic) ------------------------------
if not os.path.exists(PEAKS_TXT):
    raise FileNotFoundError(
        f"Peaks file not found: {PEAKS_TXT}\n"
        f"Please run module4_kurtz_scan with the same DATA_SOURCE to generate this peak table first."
    )

freqs_refined = []
with open(PEAKS_TXT, "r") as f:
    for line in f:
        s = line.strip()
        if (not s) or s.startswith("#"):
            continue
        parts = s.split()
        # Peak table format (written by Module 4): # i  f_bin_uHz  f_refined_uHz  power
        # Take the 3rd column (index 2) = refined μHz
        if len(parts) >= 3:
            try:
                freqs_refined.append(float(parts[2]))
            except ValueError:
                pass

if len(freqs_refined) == 0:
    raise RuntimeError(f"No valid frequencies parsed from {PEAKS_TXT}")

freqs_refined = np.array(freqs_refined[:TOP_M], dtype=np.float64)
print(f"[Module5] Frequencies to fit (μHz): {freqs_refined}")

# ------------------------------ Optional detrending (manual) ------------------------------
# Align with Module 4's DETREND_DEG; only for suppressing very slow drifts
## Keep consistent with Module 4 to suppress ultra-low frequency trends.
if DETREND_DEG in (1, 2):
    t0 = time_in.mean()
    tt = time_in - t0
    coef  = np.polyfit(tt, flux_in, deg=DETREND_DEG)
    trend = np.polyval(coef, tt)
    flux_fit = flux_in - trend
    print(f"[Module5] Detrend: degree={DETREND_DEG}, coefficients={coef}")
else:
    flux_fit = flux_in
    print("[Module5] Detrend: OFF")

# ------------------------------ Multi-frequency least squares fit ------------------------------
# Model (consistent with module5_view / module6_view reconstruction):
# y(t) ≈ Σ_m [ C_m cos(2π ν_m t) + S_m sin(2π ν_m t) ]
# This is a generic Fourier multi-sine model solved by linear least squares to obtain C_m, S_m.
# No constant term is included because the input flux has already been demeaned/detrended.
# Frequency values are fixed (no refinement); suitable for pre-whitening / amplitude-phase estimation.
# If frequency refinement is desired, non-linear least squares (e.g. LM) must be applied afterwards.
# Units must be consistent: time in seconds, frequency in μHz ⇒ must convert μHz to Hz (×1e−6) inside fit_multi_sine, 
# otherwise the phase term 2πνt will be incorrect.
# Please confirm that utils_fit.fit_multi_sine handles this conversion.
fit = fit_multi_sine(
    time_sec=time_in,
    flux=flux_fit,
    freqs_uHz=freqs_refined,
    flux_err=None   # Current cleaned_segments.npz has no error column, keep equal weighting
)

# ------------------------------ Save results (with suffix) ------------------------------
np.savez(
    OUT_NPZ,
    freqs_uHz=freqs_refined,
    C=fit["C"], S=fit["S"],
    A=fit["A"], phi=fit["phi"],
    chi2=fit["chi2"], dof=fit["dof"],
    resid=fit["resid"], cov=fit["cov"],
    meta_data_source=DATA_SOURCE,
    meta_detrend_deg=DETREND_DEG,
    meta_note="Multi-sine weighted LS on selected segment; time=sec, freq=μHz"
)

print(f"[Module5] Saved fit results -> {OUT_NPZ}")
N = time_in.size
T = float(time_in.max() - time_in.min())
rayleigh_uHz = (1.0 / T) * 1e6
print(f"[Module5] N={N}, T={T:.0f}s (~{T/86400:.2f} d), Rayleigh≈{rayleigh_uHz:.3f} μHz")
