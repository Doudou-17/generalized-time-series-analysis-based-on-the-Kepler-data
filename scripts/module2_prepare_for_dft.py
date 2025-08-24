# === Module 2 (script): Prepare for DFT (sec + μHz, keep real gap) ===
# This script prepares the light curve data for DFT analysis by cleaning and segmenting.
# This scipt uses the src.utils_data and src.utils_plot.
import os
import numpy as np
from src.utils_data import (
    load_raw_bundle, drop_nan, find_largest_gap, split_by_threshold,
    demean, save_segments
)
from src.utils_plot import plot_segments_sec_with_sci, plot_raw_sec_with_sci

NPZ_PATH = "./data/KIC_1430163_q2_module0_raw_standardized.npz"
OUT_DIR  = "./output"
FIG_RAW  = f"{OUT_DIR}/raw_cleaned_full_sec_sci.png"
FIG_SPLT = f"{OUT_DIR}/lightcurve_split_segments_sec_sci.png"
OUT_NPZ  = f"{OUT_DIR}/cleaned_segments.npz"

# Manual threshold (in seconds); if None, automatically detect the largest gap
MANUAL_GAP_TIME_SEC = None

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load data & clean only NaN values
time_sec, flux, flux_err, quality, meta = load_raw_bundle(NPZ_PATH)
time_clean, flux_clean = drop_nan(time_sec, flux)
T = time_clean.max() - time_clean.min()
print(f"N = {time_clean.size} | T = {T:.0f} sec (~{T/86400:.2f} days) | QUALITY filter: OFF")
plot_raw_sec_with_sci(time_clean, flux_clean, FIG_RAW, shift_to_t0=True)
print(f"Saved full plot -> {FIG_RAW}")

# 2) Determine gap: automatic / manual
if MANUAL_GAP_TIME_SEC is None:
    i, dt, split = find_largest_gap(time_clean)
    print(f"[Auto] Largest gap at indices {i}/{i+1}: Δt = {dt:.3f} sec (~{dt/86400:.4f} d)")
    gap_time_sec = split
else:
    gap_time_sec = float(MANUAL_GAP_TIME_SEC)
    print(f"[Manual] Split at t = {gap_time_sec:.3f} sec")

# 3) Split into segments + subtract mean for each segment
t1, f1_raw, t2, f2_raw = split_by_threshold(time_clean, flux_clean, gap_time_sec)
f1 = demean(f1_raw)
f2 = demean(f2_raw)
print(f"Segment sizes -> before: {t1.size}, after: {t2.size}")
if t1.size > 1:
    print(f"T_before = {(t1.max()-t1.min()):.0f} sec (~{(t1.max()-t1.min())/86400:.2f} d)")
if t2.size > 1:
    print(f"T_after  = {(t2.max()-t2.min()):.0f} sec (~{(t2.max()-t2.min())/86400:.2f} d)")

# 4) Plot segmented light curve (retain 1e6 scientific notation, top axis in days)
plot_segments_sec_with_sci(t1, f1, t2, f2, FIG_SPLT, shift_base=time_clean.min())
print(f"Saved split plot -> {FIG_SPLT}")

# 5) Save both segments and the full dataset (preserving the real gap)
## Concatenate the two segments directly (note: keep the actual gap, do not interpolate evenly)
time_full_sec = np.concatenate([t1, t2])
## After concatenation, subtract the global mean once for the full dataset
flux_full_global_demean = (np.concatenate([f1_raw, f2_raw]) - np.mean(np.concatenate([f1_raw, f2_raw])))
save_segments(OUT_NPZ, t1, f1, t2, f2, time_full_sec, flux_full_global_demean, gap_time_sec)
print(f"Saved segments & full arrays -> {OUT_NPZ}")
