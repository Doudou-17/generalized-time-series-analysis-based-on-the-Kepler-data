# === Module 3 (script): Phase folding (uses full concatenated data) ===
# This script performs phase folding on the light curve data. 
# This script doesn't influence the DFT results.
# This scripts uses the src.utils_data, src.utils_phase and src.utils_plot.
import os
import numpy as np

# --- Required structure for running as a package: python -m scripts.module3_phase_fold ---
from src.utils_data import load_raw_bundle
from src.utils_phase import (period_from_muHz, fold_phase, auto_t0_min_flux, bin_folded)
from src.utils_plot import plot_phase_folded

NPZ_PREP = "./output/cleaned_segments.npz"    # Produced by module2
OUT_DIR  = "./output"

# You can specify the frequency (μHz) to display here.
# Placeholder value: replace this with the primary frequency obtained from the Kurtz DFT.
NU_MUHZ = 2.8         # Replace with your main frequency (μHz). In this project, we find the main frequency to be around 2.8 μHz.
AUTO_T0 = True         # True: use "minimum flux point" as zero phase; False: use t.min() as t0
WRAP_CENTER = 0.5      # Folding into [-0.5, 0.5) is better for eclipsing binaries
NBINS = 200            # Number of phase bins (for binned curve visualization)

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load the “full concatenated data while preserving gap”
z = np.load(NPZ_PREP)
time_full = z["time_full_sec"].astype(np.float64)
flux_full = z["flux_full_global_demean"].astype(np.float64)

# 2) Select reference epoch t0
if AUTO_T0:
    t0 = auto_t0_min_flux(time_full, flux_full, NU_MUHZ, nbins=NBINS)
else:
    t0 = float(time_full.min())

# 3) Perform phase folding
phi = fold_phase(time_full, NU_MUHZ, t0_sec=t0, wrap_center=WRAP_CENTER)

# 4) Bin the folded light curve (optional, useful for shape visualization)
centers, y_med, y_err = bin_folded(phi, flux_full, nbins=NBINS)

# 5) Plot the phase-folded light curve
P_sec = period_from_muHz(NU_MUHZ)
suffix = "wrp05" if WRAP_CENTER == 0.5 else "wrp0"
out_png = f"{OUT_DIR}/phase_fold_{NU_MUHZ:.3f}uHz_{suffix}.png"
plot_phase_folded(phi, flux_full, out_png,
                  title=f"Phase-folded at {NU_MUHZ:.3f} μHz (P={P_sec/86400:.5f} d)",
                  show_two_cycles=True,
                  binned=(centers, y_med, y_err))
print(f"Phase-folded figure saved -> {out_png}")
print(f"t0 (sec) = {t0:.3f}, period = {P_sec:.3f} sec  (~{P_sec/86400:.5f} d)")
