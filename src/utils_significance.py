# ==============================
"""
Significance utilities: local-SNR on amplitude spectrum and permutation-based global FAP.
Relies on a Kurtz scan result (FR/FI power & amp) from utils_freq.kurtz_scan.
"""
from typing import Sequence, Tuple, List, Iterable
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from src.utils_freq import FreqBand, kurtz_scan


def local_snr_from_amp(
    freq_uHz: np.ndarray,
    amp: np.ndarray,
    f0_uHz: float,
    rayleigh_uHz: float,
    noise_win_rayleigh: float = 50.0,
    exclude_core_rayleigh: float = 1.0,
    stat: str = "mad",
) -> float:
    """
    Compute local SNR at f0 using amplitude spectrum (amp) with a robust noise estimate.
    - Uses a +/- (noise_win_rayleigh * Rayleigh) window excluding a core +/- (exclude_core_rayleigh * Rayleigh).
    - stat='mad': noise=max(median, MAD/1.4826); stat='median': noise=median; stat='rms': noise=rms.
    Returns SNR = A0 / noise.
    """
    if freq_uHz.size < 5:
        return np.nan
    df_uHz = float(np.median(np.diff(freq_uHz))) if freq_uHz.size > 1 else np.nan
    if not np.isfinite(df_uHz) or df_uHz <= 0:
        return np.nan

    half_win_bins = int(np.round((noise_win_rayleigh * rayleigh_uHz) / df_uHz))
    core_bins = int(np.round((exclude_core_rayleigh * rayleigh_uHz) / df_uHz))

    i0 = int(np.argmin(np.abs(freq_uHz - f0_uHz)))
    i_lo = max(0, i0 - half_win_bins)
    i_hi = min(freq_uHz.size - 1, i0 + half_win_bins)
    mask = np.ones(i_hi - i_lo + 1, dtype=bool)
    lo_core = max(i_lo, i0 - core_bins)
    hi_core = min(i_hi, i0 + core_bins)
    mask[(lo_core - i_lo):(hi_core - i_lo + 1)] = False

    A_loc = amp[i_lo:i_hi+1][mask]
    if A_loc.size < 5:
        return np.nan

    stat_l = str(stat).lower()
    if stat_l == "mad":
        med = float(np.median(A_loc))
        mad = float(np.median(np.abs(A_loc - med)))
        sigma = (mad / 1.4826) if mad > 0 else float(np.std(A_loc))
        noise = max(med, sigma)
    elif stat_l == "rms":
        noise = float(np.sqrt(np.mean(A_loc**2)))
    else:  # 'median'
        noise = float(np.median(A_loc))

    A0 = float(amp[i0])
    return A0 / max(noise, 1e-12)


# ------------------ helpers for permutation (parallel-friendly) ------------------

def _bands_to_tuples(bands: Iterable) -> Tuple[Tuple[float, float, float], ...]:
    """
    Normalize FreqBand or (fmin,fmax,S) into tuples for pickling to worker processes.
    Compatible with both FreqBand(fmin,fmax,S) objects and user-provided (fmin,fmax,S) tuples.
    """
    out: List[Tuple[float, float, float]] = []
    for b in bands:
        if isinstance(b, FreqBand):
            fmin = getattr(b, "fmin", getattr(b, "fmin_uHz", None))
            fmax = getattr(b, "fmax", getattr(b, "fmax_uHz", None))
            S    = getattr(b, "S", getattr(b, "oversample", 1.0))
            if fmin is None or fmax is None:
                raise AttributeError("FreqBand object missing fmin/fmax attributes.")
            out.append((float(fmin), float(fmax), float(S)))
        else:
            fmin, fmax, S = b
            out.append((float(fmin), float(fmax), float(S)))
    return tuple(out)


def _worker_perm_max(seed_i: int,
                     time_sec: np.ndarray,
                     series: np.ndarray,
                     bands_tuples: Tuple[Tuple[float, float, float], ...]) -> float:
    """
    One permutation: shuffle series, Kurtz scan on the same grid, return max power.
    """
    rng = np.random.default_rng(int(seed_i))
    perm = rng.permutation(series.size)
    y_perm = series[perm]
    bands = [FreqBand(*t) for t in bands_tuples]
    res_perm = kurtz_scan(time_sec, y_perm, bands, weights=None, verbose=False)
    return float(np.max(res_perm.power))


# ------------------ public API ------------------

def permutation_max_powers(
    time_sec: np.ndarray,
    series: np.ndarray,
    bands: Sequence[FreqBand] | Iterable,
    n_perm: int = 300,
    seed: int = 42,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Permutation test: shuffle series while keeping time fixed; for each shuffle, run Kurtz scan
    and record the maximum power across the grid. Returns array of length n_perm.
    n_jobs > 1 enables multi-process parallelism. With n_jobs=1, behavior matches your original code.
    """
    time_sec = np.asarray(time_sec, dtype=float)
    series = np.asarray(series, dtype=float)
    assert time_sec.ndim == 1 and series.ndim == 1 and time_sec.size == series.size, "time/series shape mismatch"

    bands_tuples = _bands_to_tuples(bands)

    rng = np.random.default_rng(seed)
    seed_list = rng.integers(0, 2**63 - 1, size=int(n_perm), dtype=np.int64)

    if int(n_jobs) <= 1:
        max_powers = np.empty(n_perm, dtype=np.float64)
        for b in range(n_perm):
            max_powers[b] = _worker_perm_max(int(seed_list[b]), time_sec, series, bands_tuples)
        return max_powers

    with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
        futs = [ex.submit(_worker_perm_max, int(s), time_sec, series, bands_tuples) for s in seed_list]
        out = [f.result() for f in futs]
    return np.asarray(out, dtype=np.float64)


def fap_from_permutation(max_powers: np.ndarray, power_obs: np.ndarray) -> np.ndarray:
    """
    Empirical global FAP for observed powers: P(max_power >= power_obs).
    Uses the (1 + count)/(1 + N) unbiased estimator.
    """
    mp = np.asarray(max_powers, dtype=float).ravel()
    po = np.asarray(power_obs, dtype=float).ravel()
    N = mp.size
    if N == 0:
        return np.ones_like(po, dtype=float)
    sort_mp = np.sort(mp)
    idx = np.searchsorted(sort_mp, po, side='left')  # first index where mp >= po
    k = (N - idx).astype(float)
    fap = (1.0 + k) / (1.0 + N)
    return fap


def thresholds_from_permutation(max_powers: np.ndarray, quantiles: Sequence[float] = (0.90, 0.95, 0.99)) -> List[Tuple[float, float]]:
    """Return list of (quantile, threshold_power)."""
    mp = np.asarray(max_powers, dtype=float).ravel()
    if mp.size == 0:
        return [(float(q), float('nan')) for q in quantiles]
    th = np.quantile(mp, np.asarray(quantiles, dtype=float))
    return list(zip([float(q) for q in quantiles], [float(v) for v in th]))
