# === Module 0 (script): Download & Standardize (sec + μHz) ===
# This script downloads a light curve from the Kepler mission.
# This script use the src.utils_data
from lightkurve import search_lightcurve
import os
from src.utils_data import standardize_lightcurve_to_npz

TARGET  = "KIC 1430163"
MISSION = "Kepler"
QUARTER = 2
AUTHOR  = "Kepler"

OUT_DIR   = "./data"
FITS_PATH = f"{OUT_DIR}/KIC_1430163_q{QUARTER}_original_data.fits"
NPZ_PATH  = f"{OUT_DIR}/KIC_1430163_q{QUARTER}_module0_raw_standardized.npz"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Searching {TARGET} ({MISSION}, Q{QUARTER}) ...")
lc = search_lightcurve(TARGET, mission=MISSION, quarter=QUARTER, author=AUTHOR).download()

# 1) Save original FITS (optional, preserves all fields)
lc.to_fits(path=FITS_PATH, overwrite=True)
print(f"FITS saved -> {FITS_PATH}")

# 2) Standardize to .npz (sec + μHz convention; no QUALITY filtering)
standardize_lightcurve_to_npz(lc, NPZ_PATH, target=TARGET, mission=MISSION, quarter=QUARTER)
print(f"Standardized NPZ saved -> {NPZ_PATH}")
print("Unit policy: time=sec, frequency=μHz")
