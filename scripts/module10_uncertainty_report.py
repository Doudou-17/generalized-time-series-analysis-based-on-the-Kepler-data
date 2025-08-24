# === Module 10: Uncertainty Report  (A,phi via cov; freq via Rayleigh/SNR), SNR & FAP. ===
# This script generates a report on the uncertainties of the multi_sinefit results.
# This script uses the src.utils_resid, src.utils_significance, src.utils_freq and src.utils_errors.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import csv
import matplotlib.pyplot as plt

from src.utils_resid import select_data_segment, apply_detrend, rebuild_model, summarize_timebase
from src.utils_significance import local_snr_from_amp
from src.utils_freq import FreqBand, kurtz_scan
from src.utils_errors import amp_phi_errors_from_cov, sigma_nu_rayleigh_snr

# ------------------- Manual parameters -------------------
DATA_SOURCE = 'before'      # 'full' | 'before' | 'after'
NPZ_PREP = "./output/cleaned_segments.npz"
OUT_DIR  = "./output"

# Frequency bands for SNR evaluation (Kurtz scan on "residual" to estimate local noise)
FREQ_BANDS_SNR = [
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
NOISE_WIN_RAYLEIGH = 50.0
EXCLUDE_CORE_RAYLEIGH = 1.0
NOISE_STAT = 'mad'

# FAP source: if *_with_fap*.txt exists, read and match (tolerance = Rayleigh); otherwise leave empty
USE_RESID_FAP = True   # True: use residual peak table's FAP; False: use original peak table's FAP

# Output files
OUT_TABLE = os.path.join(OUT_DIR, "final_table.csv")
PLOT_SIGMA = os.path.join(OUT_DIR, "sigma_nu_vs_freq.png")

# Suffix mapping (full has no suffix)
SUFFIX_MAP = {"before": "_before", "after": "_after", "full": ""}
suffix = SUFFIX_MAP.get(DATA_SOURCE, "")

# ------------------- Load fit results and data -------------------
db = select_data_segment(NPZ_PREP, DATA_SOURCE)
t, y_raw = db.time, db.flux_raw
T, rayleigh_uHz = summarize_timebase(t)

NPZ_FIT = os.path.join(OUT_DIR, f"multisine_fit_results{suffix}.npz")
if not os.path.exists(NPZ_FIT):
    raise FileNotFoundError(f"Fit results not found: {NPZ_FIT}")
fit = np.load(NPZ_FIT)
freqs = fit["freqs_uHz"].astype(np.float64)
C = fit["C"].astype(np.float64)
S = fit["S"].astype(np.float64)
A_store = fit["A"].astype(np.float64)
phi_store = fit["phi"].astype(np.float64)
resid_fit = fit["resid"].astype(np.float64)
chi2 = float(fit.get("chi2", np.nan))
dof  = int(fit.get("dof", 0))
detrend_deg = int(fit.get("meta_detrend_deg", 0))

# Covariance matrix of design matrix: shape should be (2M, 2M), order assumed as [C0,S0,C1,S1,...]
cov = fit.get("cov", None)
if cov is None:
    print("WARNING: cov not found in fit file; σ_A/σ_φ will be NaN.")
else:
    cov = np.array(cov, dtype=np.float64)

# Apply detrend consistent with the fit
y_in, _ = apply_detrend(t, y_raw, detrend_deg)

# ------------------- Estimate SNR from residual spectrum (local noise) -------------------
bands = [FreqBand(fmin, fmax, S) for (fmin, fmax, S) in FREQ_BANDS_SNR]
res_resid = kurtz_scan(t, resid_fit, bands, weights=None, verbose=False)  # Amplitude/power spectrum of residuals

# ------------------- (Optional) Read FAP table and establish mapping -------------------
if USE_RESID_FAP:
    fap_path = os.path.join(OUT_DIR, f"kurtz_resid_top_peaks_with_fap{suffix}.txt")
else:
    fap_path = os.path.join(OUT_DIR, f"kurtz_top_peaks_with_fap{suffix}.txt")
FAP_map = {}
if os.path.exists(fap_path):
    eps_match = rayleigh_uHz  # 容差（μHz）
    frs, faps = [], []
    with open(fap_path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith('#'): continue
            parts = s.split()
            if len(parts) >= 3:
                try:
                    fr = float(parts[0]); fp = float(parts[2])
                    frs.append(fr); faps.append(fp)
                except ValueError:
                    pass
    frs = np.array(frs, dtype=np.float64); faps = np.array(faps, dtype=np.float64)
    for f0 in freqs:
        j = int(np.argmin(np.abs(frs - f0))) if frs.size else -1
        if j >= 0 and abs(frs[j] - f0) <= eps_match:
            FAP_map[float(f0)] = float(faps[j])

# ------------------- Summarize parameters and uncertainties for each frequency -------------------
rows = []
sigmas = []
for m, f_uHz in enumerate(freqs):
    C_m = float(C[m]); S_m = float(S[m])
    A_m = float(A_store[m]); phi_m = float(phi_store[m])

    # Covariance sub-block (C,S) → (σ_A, σ_φ)
    if cov is not None and cov.ndim == 2 and cov.shape[0] >= 2*len(freqs):
        iC, iS = 2*m, 2*m+1
        cov2 = cov[iC:iS+1, iC:iS+1]
        sigma_A, sigma_phi = amp_phi_errors_from_cov(C_m, S_m, cov2, A_stored=A_m)
    else:
        sigma_A, sigma_phi = float('nan'), float('nan')

    # SNR: estimated on residual amplitude spectrum with local noise window
    SNR = local_snr_from_amp(res_resid.freq_uHz, res_resid.amp, f_uHz,
                             rayleigh_uHz, NOISE_WIN_RAYLEIGH, EXCLUDE_CORE_RAYLEIGH, NOISE_STAT)

    # Frequency uncertainty (Rayleigh/SNR approximation)
    sigma_nu = sigma_nu_rayleigh_snr(T, SNR)
    sigmas.append(sigma_nu)

    # FAP (if available)
    FAP_val = FAP_map.get(float(f_uHz), float('nan'))

    period_day = 1.0 / ((f_uHz * 1e-6) * 86400.0)

    rows.append({
        'source': DATA_SOURCE,
        'freq_uHz': f_uHz,
        'period_day': period_day,
        'A': A_m,
        'sigma_A': sigma_A,
        'phi': phi_m,
        'sigma_phi': sigma_phi,
        'sigma_nu_uHz': sigma_nu,
        'SNR': SNR,
        'FAP': FAP_val,
        'N': t.size,
        'T_sec': T,
        'Rayleigh_uHz': rayleigh_uHz,
        'R2': np.nan,  
        'RMS_resid': float(np.sqrt(np.mean(resid_fit**2))),
        'chi2': chi2,
        'dof': dof,
    })

# ------------------- Write CSV -------------------
fieldnames = ['source','freq_uHz','period_day','A','sigma_A','phi','sigma_phi','sigma_nu_uHz','SNR','FAP','N','T_sec','Rayleigh_uHz','R2','RMS_resid','chi2','dof']
with open(OUT_TABLE, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"Saved final table -> {OUT_TABLE}")

# ------------------- Visualization: σ_ν vs Frequency -------------------
if len(rows) > 0:
    f_arr = np.array([r['freq_uHz'] for r in rows])
    s_arr = np.array([r['sigma_nu_uHz'] for r in rows])
    plt.figure(figsize=(6.5,4))
    plt.plot(f_arr, s_arr, 'o-')
    plt.xlabel('Frequency (μHz)'); plt.ylabel('σ_ν (μHz)')
    plt.title('Frequency uncertainty (Rayleigh/SNR approx.)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(PLOT_SIGMA, dpi=200); plt.close()
    print(f"Saved -> {PLOT_SIGMA}")
