# === Module 1 (script): Plot Raw (NO QUALITY filter, sec + μHz) ===
# This script plots the raw light curve data without QUALITY filtering.
# This script use the src.utils_data and scr.utils_plot.
import os
from src.utils_data import load_raw_bundle, drop_nan
from src.utils_plot import plot_raw_sec_with_sci

# Load the standardized raw dataset (full data)
NPZ_PATH = "./data/KIC_1430163_q2_module0_raw_standardized.npz"
OUT_DIR  = "./output"
PLOT     = f"{OUT_DIR}/raw_lightcurve_no_quality_filter_sec_sci.png"

os.makedirs(OUT_DIR, exist_ok=True)
time_sec, flux, flux_err, quality, meta = load_raw_bundle(NPZ_PATH)
time_plot, flux_plot = drop_nan(time_sec, flux)

T = time_plot.max() - time_plot.min()
print(f"N = {time_plot.size} | T = {T:.0f} sec (~{T/86400:.2f} days) | QUALITY filter: OFF")
plot_raw_sec_with_sci(time_plot, flux_plot, PLOT, shift_to_t0=True)
print(f"Figure saved -> {PLOT}")