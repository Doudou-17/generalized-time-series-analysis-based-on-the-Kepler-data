# === Module 6: View multi-sine fit (overlay + residuals + summary) ===
# This script views the multi_sine fit results from Module 5.
# Units: time = sec, freq = μHz.
# This module does not perform fitting; it only loads results from Module 5, rebuilds the model,
# plots overlay & residuals, and generates a textual summary.
# [Kurtz/DFT relevance] Used to verify whether the multi-sine model explains the main variations,
# and whether residual periodicities remain.
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# ------------------------- Manually adjustable parameters (others automatic) -------------------------
# Must match Module 4/5: choose data source for viewing fit results 'full' | 'before' | 'after'
DATA_SOURCE = 'after'
NPZ_PREP = "./output/cleaned_segments.npz"   # Output of Module 2
OUT_DIR  = "./output"
os.makedirs(OUT_DIR, exist_ok=True)

SHIFT_TIME_TO_T0 = True  
# Only affects visualization; shifts x-axis to start at 0 without altering true timestamps/gap.
# -----------------------------------------------------------------------------------------------------

# ------------------------- Select dataset according to DATA_SOURCE ------------------------
z = np.load(NPZ_PREP)

if DATA_SOURCE == 'before':
    time_in = z["time1"].astype(np.float64)                      # seconds
    flux_in_raw = z["flux1"].astype(np.float64)                  # e-/s (already demeaned per segment)
    suffix = "_before"
elif DATA_SOURCE == 'after':
    time_in = z["time2"].astype(np.float64)                      # seconds
    flux_in_raw = z["flux2"].astype(np.float64)                  # e-/s (already demeaned per segment)
    suffix = "_after"
else:  # 'full'
    time_in = z["time_full_sec"].astype(np.float64)              # seconds (keep real gap)
    flux_in_raw = z["flux_full_global_demean"].astype(np.float64)# e-/s (global demeaned)
    suffix = ""

# Fit results file (must match Module 5 output)
# Default: use Module 5 results directly (no new fitting)
# NPZ_FIT = os.path.join(OUT_DIR, f"multisine_fit_results{suffix}.npz")
# If using refined results (e.g., Module 5 with nonlinear adjustment), change accordingly:
# NPZ_FIT = os.path.join(OUT_DIR, f"multisine_fit_results{suffix}_refined.npz")
# In dissertation, we just use Module 5 results directly.

NPZ_FIT = os.path.join(OUT_DIR, f"multisine_fit_results{suffix}.npz")
if not os.path.exists(NPZ_FIT):
    raise FileNotFoundError(
        f"Fit results not found: {NPZ_FIT}\n"
        f"Please run module5_multi_sine_fit.py with the same DATA_SOURCE first."
    )

# ------------------------- Load fit results (automatic) ------------------------------
fit = np.load(NPZ_FIT)
freqs = fit["freqs_uHz"].astype(np.float64)   # μHz
C = fit["C"].astype(np.float64)               # Cosine coefficients
S = fit["S"].astype(np.float64)               # Sine coefficients
A = fit["A"].astype(np.float64)               # Amplitude scale (defined in Module 5, for summary)
phi = fit["phi"].astype(np.float64)           # Phase (for summary)
resid_saved = fit["resid"].astype(np.float64)
meta_detrend_deg = int(fit.get("meta_detrend_deg", 0))
meta_source_in_fit = str(fit.get("meta_data_source", DATA_SOURCE))

if meta_source_in_fit != DATA_SOURCE:
    print(f"WARNING: Fit file recorded source '{meta_source_in_fit}', "
          f"but currently viewing '{DATA_SOURCE}'. Continuing display.")

# ------------------------- Apply detrending consistent with Module 5 (automatic) -----------------------
# To ensure consistency: if detrending was applied in Module 5, apply the same order here to flux_in_raw.
if meta_detrend_deg in (1, 2):
    t0 = time_in.mean()
    tt = time_in - t0
    coef  = np.polyfit(tt, flux_in_raw, deg=meta_detrend_deg)
    trend = np.polyval(coef, tt)
    flux_in = flux_in_raw - trend
    print(f"[Module6] Apply same detrend as fit: degree={meta_detrend_deg}, coefficients={coef}")
else:
    flux_in = flux_in_raw
    print("[Module6] Detrend: OFF (consistent with fit)")

# ------------------------- Rebuild model curve (consistent with Module 5) ------------------------
# [Kurtz/DFT relevance] This reconstructs the model in the time domain; unit conversion is critical.
TWOPI = 2.0 * np.pi
y_model = np.zeros_like(flux_in)
for Cm, Sm, f_uHz in zip(C, S, freqs):
    f_Hz = f_uHz * 1e-6
    ph = (TWOPI * f_Hz * time_in) % TWOPI
    y_model += Cm * np.cos(ph) + Sm * np.sin(ph)

resid = flux_in - y_model

# Validation: check consistency with saved residuals (within numerical tolerance)
if not np.allclose(resid, resid_saved, atol=1e-6 * max(1.0, np.nanmax(np.abs(flux_in)))):
    print("WARNING: Residual mismatch vs. saved resid (within tolerance). Continuing with current residuals for plotting.")

# ------------------------- Statistics (R² & RMS) ---------------------------------
ss_tot = float(np.sum((flux_in - np.mean(flux_in))**2))
ss_res = float(np.sum(resid**2))
## R² measures goodness of fit. That means R² measures explained variance (closer to 1 is better)
r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
## RMS of residuals measures remaining scatter
rms_resid = float(np.sqrt(np.mean(resid**2)))
# Note: High R² but residuals showing periodic structure → indicates unmodeled frequencies (or aliases/harmonics).

# ------------------------- Output file names (with suffix) -------------------------------
PLOT_OVERLAY = os.path.join(OUT_DIR, f"fit_overlay_timeseries{suffix}.png")
PLOT_RESID   = os.path.join(OUT_DIR, f"fit_residual_timeseries{suffix}.png")
OUT_SUMMARY  = os.path.join(OUT_DIR, f"multisine_fit_summary{suffix}.txt")

# ------------------------- Summary text: frequency / period / amplitude / phase ----------------------
# For each frequency, output: frequency, period, amplitude, phase. Also include global statistics.
with open(OUT_SUMMARY, "w") as f:
    f.write("# freq_uHz   period_day     A(arb)        phi(rad)\n")
    for fu, Ai, ph0 in zip(freqs, A, phi):
        period_day = 1.0 / ((fu * 1e-6) * 86400.0)
        f.write(f"{fu:10.6f}  {period_day:10.6f}  {Ai:12.6e}  {ph0:10.6f}\n")
    f.write(f"\n# DATA_SOURCE={DATA_SOURCE}, detrend_deg={meta_detrend_deg}\n")
    f.write(f"# N={time_in.size}, R^2={r2:.6f}, RMS(resid)={rms_resid:.6e} e-/s\n")
print(f"Saved summary -> {OUT_SUMMARY}")

# ------------------------- Plotting (overlay and residuals) ---------------------------------
def sec_to_day(x): return x / 86400.0
def day_to_sec(d): return d * 86400.0

tt = time_in.copy()
if SHIFT_TIME_TO_T0:
    tt = tt - tt.min()
    xlabel = "Time since start (sec)"
    top_xlabel = "Time since start (day)"
else:
    xlabel = "Time (sec)"
    top_xlabel = "Time (day)"

# (1) Data vs. model overlay (note: "data" here is flux_in, consistent with the fit)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(tt, flux_in, ".", ms=2, alpha=0.6, label="data (fit-ready)")
ax.plot(tt, y_model, "-", lw=1.2, label="model")

fmtx = ScalarFormatter(useMathText=True)
fmtx.set_scientific(True)      # Use scientific notation
fmtx.set_powerlimits((0, 0))   # Always use scientific notation
fmtx.set_useOffset(False)      # Disable corner offset text
ax.xaxis.set_major_formatter(fmtx)
ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
fmty = ScalarFormatter(useMathText=True)
fmty.set_scientific(True)
fmty.set_powerlimits((0, 0))
fmty.set_useOffset(False)
ax.yaxis.set_major_formatter(fmty)
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

ax.set_xlabel(xlabel); ax.set_ylabel("Flux (e-/s)")
ax.set_title(f"Multi-sine fit overlay{suffix} (R²={r2:.4f}, RMS={rms_resid:.3e} e-/s)")
ax.grid(True, alpha=0.3); ax.legend(loc="best")
fig.tight_layout(); fig.savefig(PLOT_OVERLAY, dpi=220); plt.close(fig)
print(f"Saved overlay -> {PLOT_OVERLAY}")

# (2) Residual time series
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(tt, resid, ".", ms=2)

fmtx = ScalarFormatter(useMathText=True)
fmtx.set_scientific(True)      # Use scientific notation
fmtx.set_powerlimits((0, 0))   # Always use scientific notation
fmtx.set_useOffset(False)      # Disable corner offset text
ax.xaxis.set_major_formatter(fmtx)
ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
fmty = ScalarFormatter(useMathText=True)
fmty.set_scientific(True)
fmty.set_powerlimits((0, 0))
fmty.set_useOffset(False)
ax.yaxis.set_major_formatter(fmty)
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

ax.set_xlabel(xlabel); ax.set_ylabel("Residual (e-/s)")
ax.set_title(f"Residual time series{suffix}")
ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(PLOT_RESID, dpi=220); plt.close(fig)
print(f"Saved residuals -> {PLOT_RESID}")
