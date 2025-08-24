# === Module 8: Significance of peaks (FAP via permutation) ===
# Evaluates significance of peaks in (residual) periodogram using permutation-based global FAP.
# Outputs:
#   1) kurtz_resid_top_peaks_with_fap{suffix}.txt
#   2) perm_thresholds{suffix}.txt  (90/95/99%)
#   3) kurtz_resid_periodogram_with_thresholds{suffix}.png
#   4) kurtz_resid_periodogram_zoom_with_thresholds{suffix}.png
#   5) perm_cdf_maxpower{suffix}.png
#   6) snr_window_XX{suffix}.png

# --- Speed safety: limit BLAS threads BEFORE importing numpy ---
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils_resid import select_data_segment, apply_detrend, rebuild_model
from src.utils_freq import FreqBand, kurtz_scan
from src.utils_significance import (
    local_snr_from_amp,
    permutation_max_powers,
    fap_from_permutation,
    thresholds_from_permutation,
)

# ------------------- Manual parameters -------------------
DATA_SOURCE = 'after'        # 'full' | 'before' | 'after'
SERIES = 'resid'             # 'resid' (recommended) or 'flux'
# In this dissertation, we focus on the residual series.
NPZ_PREP = "./output/cleaned_segments.npz"
OUT_DIR  = "./output"

# Frequency bands + oversampling (must be consistent with M4/M7 to keep the same grid)
## If we consider the residual series, we use the Frequency bands from Module 7.
## If we consider the flux series, we use the Frequency bands from Module 5.
FREQ_BANDS = [
    (1.0,   20.0,   26.0),
    (20.0,  200.0, 30.0),
    (200.0, 8500.0, 32.0),
]
# Module 7 frequency Bands:
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

# Permutation settings (fixed at 300)
## If we want to make the analysis more robust, we can increase the number of permutations.
N_PERM = 300
RANDOM_SEED = 42
USE_RESID_PEAKS = True

# Local SNR window settings
NOISE_WIN_RAYLEIGH = 50.0     # Noise window width (±50 Rayleigh)
EXCLUDE_CORE_RAYLEIGH = 1.0   # Exclude ±1 Rayleigh around the peak
NOISE_STAT = "mad"            # Use MAD (robust) instead of mean/std

# Parallelism (mild parallelism; set to 1 for serial run if needed)
N_JOBS = min(4, os.cpu_count())

# ------------------- Helpers -------------------
def _bands_signature(bands_cfg):
    arr = np.asarray(bands_cfg, dtype=float).ravel()
    return tuple(np.round(arr, 8).tolist())

def run_permutation_with_cache(cache_path, time_in, series, bands_cfg, n_perm, seed, n_jobs,
                               data_source, series_name):
    """Thin cache around permutation_max_powers to avoid re-running."""
    T = float(time_in.max() - time_in.min()) if time_in.size else 0.0
    meta = dict(
        n_perm=int(n_perm),
        seed=int(seed),
        N=int(time_in.size),
        T=float(T),
        bands=_bands_signature(bands_cfg),
        data_source=str(data_source),
        series=str(series_name),
    )
    if os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=True)
            mp = np.asarray(z["max_powers"], dtype=np.float64)
            old = z["meta"].item() if isinstance(z["meta"], np.ndarray) else z["meta"]
            ok = (
                int(old.get("n_perm",-1)) == meta["n_perm"] and
                int(old.get("seed",-2)) == meta["seed"] and
                int(old.get("N",-3)) == meta["N"] and
                float(old.get("T",-4.0)) == meta["T"] and
                tuple(old.get("bands",())) == meta["bands"] and
                str(old.get("data_source","")) == meta["data_source"] and
                str(old.get("series","")) == meta["series"]
            )
            if ok and mp.size == n_perm:
                print(f"[M8] Loaded permutation cache: {cache_path} (N={mp.size})")
                return mp
            else:
                print("[M8] Cache mismatch; recomputing...")
        except Exception as e:
            print(f"[M8] Cache read failed ({e}); recomputing...")

    bands = [FreqBand(fmin, fmax, S) for (fmin, fmax, S) in bands_cfg]
    mp = permutation_max_powers(time_in, series, bands, n_perm=n_perm, seed=seed, n_jobs=n_jobs)
    try:
        np.savez(cache_path, max_powers=np.asarray(mp, dtype=np.float64), meta=meta)
        print(f"[M8] Saved permutation cache -> {cache_path}")
    except Exception as e:
        print(f"[M8] Warning: failed to save cache ({e})")
    return np.asarray(mp, dtype=np.float64)

def _plot_resid_with_thresholds(res, peaks, qs, out_dir, suffix, zoom=False):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(res.freq_uHz, res.power, lw=1.0)
    if zoom:
        ax.set_xlim(0, 20)
    for q, thr in qs:
        ax.axhline(thr, ls='--', alpha=0.7, label=f'{int(q*100)}% threshold')
    for fr in peaks:
        ax.axvline(fr, color='r', ls=':', lw=0.9, alpha=0.6)
    ttl = "Residual periodogram with permutation thresholds" + (" (zoom 0–20 μHz)" if zoom else "")
    ax.set_title(ttl + suffix)
    ax.set_xlabel("Frequency (μHz)")
    ax.set_ylabel("DFT Power (residual)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fn = os.path.join(out_dir, f"kurtz_resid_periodogram_{'zoom_' if zoom else ''}with_thresholds{suffix}.png")
    fig.savefig(fn, dpi=200)
    plt.close(fig)
    print(f"[M8] Saved -> {fn}")

def _plot_local_window(fr, kidx, res_true, rayleigh_uHz, peaks, snr_list, out_dir, suffix):
    W = NOISE_WIN_RAYLEIGH * rayleigh_uHz
    core = EXCLUDE_CORE_RAYLEIGH * rayleigh_uHz
    f = res_true.freq_uHz
    A = res_true.amp
    m = (f >= fr - W) & (f <= fr + W)
    if not np.any(m):
        return
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(f[m], A[m], lw=1.2)
    plt.axvspan(fr - W, fr + W, alpha=0.06, color='C0', label=f'±{NOISE_WIN_RAYLEIGH:.0f} Rayleigh')
    plt.axvspan(fr - core, fr + core, alpha=0.15, color='C1', label=f'±{EXCLUDE_CORE_RAYLEIGH:.0f} Rayleigh (excluded)')
    plt.axvline(fr, ls='--', lw=1.0, color='r')
    idx = int(np.argmin(np.abs(peaks - fr)))
    ttl = f"Local amplitude window around {fr:.3f} μHz  |  SNR={snr_list[idx]:.2f}"
    plt.title(ttl)
    plt.xlabel("Frequency (μHz)")
    plt.ylabel("Amplitude (norm)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    fn = os.path.join(out_dir, f"snr_window_{kidx:02d}{suffix}.png")
    plt.savefig(fn, dpi=200)
    plt.close()
    print(f"[M8] Saved -> {fn}")

# ------------------- Main -------------------
def main():
    t0 = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)
    SUFFIX_MAP = {"before": "_before", "after": "_after", "full": ""}
    suffix = SUFFIX_MAP.get(DATA_SOURCE, "_full")

    # 1) Load time series
    db = select_data_segment(NPZ_PREP, DATA_SOURCE)
    time_in, flux_base = db.time, db.flux_raw

    # 2) Residual or raw flux
    if SERIES == 'resid':
        fit_path = os.path.join(OUT_DIR, f"multisine_fit_results{suffix}.npz")
        if not os.path.exists(fit_path):
            raise FileNotFoundError(f"Fit results not found: {fit_path}（先跑 Module 5）")
        fit = np.load(fit_path)
        freqs = fit["freqs_uHz"].astype(np.float64)
        C = fit["C"].astype(np.float64)
        S = fit["S"].astype(np.float64)
        detrend_deg = int(fit.get("meta_detrend_deg", 0))
        flux_in, _ = apply_detrend(time_in, flux_base, detrend_deg)
        y_model = rebuild_model(time_in, freqs, C, S)
        series = flux_in - y_model
    else:
        series = flux_base

    # 3) True scan (Kurtz) on the same grid
    bands = [FreqBand(fmin, fmax, S) for (fmin, fmax, S) in FREQ_BANDS]
    res_true = kurtz_scan(time_in, series, bands, weights=None, verbose=False)

    # 4) Read peaks (refined frequency list)
    txt_in = os.path.join(
        OUT_DIR,
        f"kurtz_resid_top_peaks{suffix}.txt" if USE_RESID_PEAKS else f"kurtz_top_peaks{suffix}.txt"
    )
    if not os.path.exists(txt_in):
        raise FileNotFoundError(f"Peaks file not found: {txt_in}")

    peaks = []
    with open(txt_in, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 3:
                try:
                    peaks.append(float(parts[2]))
                except ValueError:
                    pass
    peaks = np.array(peaks, dtype=np.float64)

    # 5) Observed powers at nearest grid
    def obs_power_at(fr_uHz: float) -> float:
        i = int(np.argmin(np.abs(res_true.freq_uHz - fr_uHz)))
        return float(res_true.power[i])
    p_obs = np.array([obs_power_at(fr) for fr in peaks], dtype=np.float64)

    # 6) Local SNR per peak
    T_sec = float(time_in.max() - time_in.min()) if time_in.size else 0.0
    rayleigh_uHz = (1.0 / max(T_sec, 1e-12)) * 1e6
    snr_list = [
        local_snr_from_amp(
            res_true.freq_uHz, res_true.amp, float(fr), rayleigh_uHz,
            noise_win_rayleigh=NOISE_WIN_RAYLEIGH,
            exclude_core_rayleigh=EXCLUDE_CORE_RAYLEIGH,
            stat=NOISE_STAT
        )
        for fr in peaks
    ]
    snr_list = np.array(snr_list, dtype=float)
    print(f"[M8] SNR (per peak):", np.array2string(snr_list, precision=2, separator=', '))

    # 7) Plot local SNR windows
    for k, fr in enumerate(peaks):
        _plot_local_window(float(fr), k, res_true, rayleigh_uHz, peaks, snr_list, OUT_DIR, suffix)

    # 8) Permutation test with cache
    cache_path = os.path.join(OUT_DIR, f"perm_cache_{DATA_SOURCE}_{SERIES}.npz")
    max_powers = run_permutation_with_cache(
        cache_path, time_in, series, FREQ_BANDS, N_PERM, RANDOM_SEED, N_JOBS,
        data_source=DATA_SOURCE, series_name=SERIES
    )

    # 9) FAP + thresholds
    fap = fap_from_permutation(max_powers, p_obs)
    qs = thresholds_from_permutation(max_powers, quantiles=(0.90, 0.95, 0.99))

    # 10) Write tables
    out_with_fap = os.path.join(
        OUT_DIR,
        f"{'kurtz_resid_top_peaks' if USE_RESID_PEAKS else 'kurtz_top_peaks'}_with_fap{suffix}.txt"
    )
    with open(out_with_fap, "w") as f:
        f.write("# f_refined_uHz   power_obs       FAP_global    SNR_local\n")
        for fr, po, fp, sr in zip(peaks, p_obs, fap, snr_list):
            f.write(f"{fr:12.6f}  {po:12.6e}  {fp:10.6f}  {sr:10.4f}\n")
    print(f"[M8] Saved FAP table -> {out_with_fap}")

    thr_txt = os.path.join(OUT_DIR, f"perm_thresholds{suffix}.txt")
    with open(thr_txt, "w") as f:
        f.write("# quantile   threshold_power\n")
        for q, thr in qs:
            f.write(f"{q:.2f}  {thr:.6e}\n")
    print(f"[M8] Saved thresholds -> {thr_txt}")

    # 11) Plots: full/zoom + CDF
    _plot_resid_with_thresholds(res_true, peaks, qs, OUT_DIR, suffix, zoom=False)
    _plot_resid_with_thresholds(res_true, peaks, qs, OUT_DIR, suffix, zoom=True)

    plt.figure(figsize=(6,4))
    xs = np.sort(max_powers)
    ys = np.linspace(0,1,xs.size,endpoint=False)
    plt.plot(xs, ys, lw=1.5, label="Permutation CDF of max power")
    for po in np.sort(p_obs):
        plt.axvline(po, color="r", ls="--", alpha=0.4)
    plt.xlabel("Max DFT power over grid (permutation)")
    plt.ylabel("Empirical CDF")
    plt.title(f"Global FAP via permutation{suffix} (N={len(max_powers)})")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"perm_cdf_maxpower{suffix}.png"), dpi=200)
    plt.close()
    print(f"[M8] Saved CDF plot -> ./output/perm_cdf_maxpower{suffix}.png")

    print(f"[M8] Runtime: {time.perf_counter() - t0:.2f} s, N_JOBS={N_JOBS}, N_PERM={N_PERM}")

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()
