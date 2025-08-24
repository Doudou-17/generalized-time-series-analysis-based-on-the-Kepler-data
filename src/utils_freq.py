# =========================
# Kurtz recursive DFT for uneven sampling (time in sec, freq in μHz)
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

TWOPI = 2.0 * np.pi

@dataclass
class FreqBand:
    fmin_uHz: float
    fmax_uHz: float
    oversample: float  # S: samples per Rayleigh peak

@dataclass
class KurtzResult:
    freq_uHz: np.ndarray  # (M,)
    FR: np.ndarray        # (M,)
    FI: np.ndarray        # (M,)
    power: np.ndarray     # (M,) = FR^2 + FI^2 (or weighted)
    amp: np.ndarray       # (M,) = 2/N_eff * sqrt(power)
    meta: Dict[str, object]


def _kurtz_segment(time_sec: np.ndarray,
                   y: np.ndarray,
                   weights: Optional[np.ndarray],
                   f0_uHz: float,
                   M: int,
                   df_uHz: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FR, FI on a segment of equally spaced frequencies using Kurtz recursion.
    Vectorized over time, loops over frequency steps.
    time_sec: (N,)
    y:        (N,)  (already demeaned if you want)
    weights:  (N,)  or None
    f0_uHz:   starting frequency (μHz)
    M:        number of steps
    df_uHz:   step in μHz (>0)
    Returns: FR(M,), FI(M,)
    """
    t = np.asarray(time_sec, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = t.size
    if weights is None:
        w = np.ones(N, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)

    # Convert to Hz
    f0_Hz = f0_uHz * 1e-6
    df_Hz = df_uHz * 1e-6

    # Initial phase φ0_k = 2π f0 t_k (mod 2π) for numerical stability
    phi0 = (TWOPI * f0_Hz * t) % TWOPI
    c = np.cos(phi0)
    s = np.sin(phi0)

    # Step phase per point θ_k = 2π df t_k (mod 2π)
    theta = (TWOPI * df_Hz * t) % TWOPI
    cD = np.cos(theta)
    sD = np.sin(theta)

    FR = np.zeros(M, dtype=np.float64)
    FI = np.zeros(M, dtype=np.float64)

    # Vectorized recursion across time; loop over frequency index
    wy = w * y
    for j in range(M):
        # accumulate at current frequency
        FR[j] = np.dot(wy, c)
        FI[j] = np.dot(wy, s)
        # update c,s for next frequency step
        if j != M - 1:
            cn = c * cD - s * sD
            sn = s * cD + c * sD
            c, s = cn, sn
    return FR, FI


def build_freq_grid(time_sec: np.ndarray,
                    bands: List[FreqBand]) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """
    Build a concatenated frequency grid (μHz) from bands.
    For each band, step is df = 1e6 / (S * T), where T = (tmax - tmin) in sec.
    Returns: (freq_uHz, segments_meta)
      - freq_uHz: concatenated 1D array
      - segments_meta: list of (start_index, length, df_uHz) for each band
    """
    t = np.asarray(time_sec, dtype=np.float64)
    T = float(np.max(t) - np.min(t))
    if T <= 0:
        raise ValueError("Non-positive time span.")

    freq_list = []
    segs: List[Tuple[int, int, float]] = []
    start = 0
    for band in bands:
        df_uHz = 1e6 / (band.oversample * T)  # μHz
        fmin = float(band.fmin_uHz)
        fmax = float(band.fmax_uHz)
        if fmax <= fmin:
            continue
        # ensure inclusive start, exclusive end
        grid = np.arange(fmin, fmax, df_uHz, dtype=np.float64)
        if grid.size == 0:
            continue
        freq_list.append(grid)
        segs.append((start, grid.size, df_uHz))
        start += grid.size
    if not freq_list:
        raise ValueError("No frequencies produced. Check bands.")
    freq_uHz = np.concatenate(freq_list)
    return freq_uHz, segs


def kurtz_scan(time_sec: np.ndarray,
               flux: np.ndarray,
               bands: List[FreqBand],
               weights: Optional[np.ndarray] = None,
               compute_amp: bool = True,
               verbose: bool = True) -> KurtzResult:
    """
    Run Kurtz recursive DFT over (possibly multiple) frequency bands.
    Returns FR, FI, power (=FR^2+FI^2), and amplitude scaling (2/N_eff * sqrt(power)).
    """
    t = np.asarray(time_sec, dtype=np.float64)
    y = np.asarray(flux, dtype=np.float64)
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=np.float64)

    # Build grid
    freq_uHz, segs = build_freq_grid(t, bands)

    # Process each segment with its own df
    FR_all = np.zeros_like(freq_uHz)
    FI_all = np.zeros_like(freq_uHz)

    for (start, length, df_uHz), band in zip(segs, bands):
        stop = start + length
        f0_uHz = freq_uHz[start]
        if verbose:
            print(f"[Kurtz] Band {band.fmin_uHz:.3f}-{band.fmax_uHz:.3f} μHz | df={df_uHz:.6f} μHz | M={length}")
        FR, FI = _kurtz_segment(t, y, w, f0_uHz=f0_uHz, M=length, df_uHz=df_uHz)
        FR_all[start:stop] = FR
        FI_all[start:stop] = FI

    power = FR_all**2 + FI_all**2

    if compute_amp:
        # Effective weight count ~ sum(w) for scaling; classical unweighted 2/N
        Neff = np.sum(w)
        amp = (2.0 / Neff) * np.sqrt(power)
    else:
        amp = np.full_like(power, np.nan)

    meta = {
        "bands": [band.__dict__ for band in bands],
        "segments": segs,
        "time_span_sec": float(np.max(t) - np.min(t))
    }
    return KurtzResult(freq_uHz=freq_uHz, FR=FR_all, FI=FI_all, power=power, amp=amp, meta=meta)


def spectral_window(time_sec: np.ndarray,
                    bands: List[FreqBand],
                    weights: Optional[np.ndarray] = None,
                    verbose: bool = True) -> KurtzResult:
    """Compute spectral window by setting y ≡ 1 (or sqrt(weights))."""
    t = np.asarray(time_sec, dtype=np.float64)
    if weights is None:
        y = np.ones_like(t)
        w = None
    else:
        # Use y = sqrt(w) so that power ~ sum(w) normalization
        w = None
        y = np.sqrt(np.asarray(weights, dtype=np.float64))
    return kurtz_scan(t, y, bands, weights=w, compute_amp=False, verbose=verbose)


def find_top_peaks(freq_uHz: np.ndarray, power: np.ndarray, k: int = 5,
                   min_separation_bins: int = 3) -> List[int]:
    """Find indices of top-k local maxima with a minimum separation in bins."""
    p = np.asarray(power)
    # local maxima mask
    m = (p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])
    idx = np.where(m)[0] + 1
    if idx.size == 0:
        # fallback: just take global maxima
        return [int(np.argmax(p))]
    # sort by power desc
    idx = idx[np.argsort(p[idx])[::-1]]
    picked = []
    for i in idx:
        if all(abs(i - j) >= min_separation_bins for j in picked):
            picked.append(int(i))
        if len(picked) >= k:
            break
    return picked


def refine_peak_quadratic(freq_uHz: np.ndarray, power: np.ndarray, i: int) -> Tuple[float, float]:
    """
    Quadratic interpolation using (i-1, i, i+1) to estimate sub-bin peak.
    Returns (f_hat_uHz, p_hat).
    """
    if i <= 0 or i >= len(power) - 1:
        return float(freq_uHz[i]), float(power[i])
    x1, x2, x3 = freq_uHz[i-1:i+2]
    y1, y2, y3 = power[i-1:i+2]
    # fit parabola y = ax^2 + bx + c
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denom == 0:
        return float(x2), float(y2)
    a = (x3*(y2 - y1) + x2*(y1 - y3) + x1*(y3 - y2)) / denom
    b = (x3**2*(y1 - y2) + x2**2*(y3 - y1) + x1**2*(y2 - y3)) / denom
    if a == 0:
        return float(x2), float(y2)
    x_peak = -b / (2*a)
    y_peak = a*x_peak**2 + b*x_peak + (y1 - a*x1**2 - b*x1)
    return float(x_peak), float(y_peak)


def direct_FR_FI(time_sec: np.ndarray, y: np.ndarray, f_uHz: float, weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Compute FR, FI at a single frequency via direct trig sums (for refinement)."""
    t = np.asarray(time_sec, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=np.float64)
    f_Hz = f_uHz * 1e-6
    phi = (TWOPI * f_Hz * t) % TWOPI
    c = np.cos(phi)
    s = np.sin(phi)
    FR = np.dot(w * y, c)
    FI = np.dot(w * y, s)
    return float(FR), float(FI)


def default_bands_from_data(time_sec: np.ndarray,
                            Pmax_days: float = 40.0,
                            fmax_factor: float = 0.5,
                            S_low: float = 3.0,
                            S_mid: float = 6.0,
                            S_high: float = 10.0) -> List[FreqBand]:
    """
    Create three bands based on data span and median cadence.
    - fmin = 1 / Pmax_days
    - fmax ≈ fmax_factor / median(dt) (in Hz → μHz)
    - split at ~[8, 30] days periods
    """
    t = np.asarray(time_sec, dtype=np.float64)
    dt = np.diff(np.sort(t))
    dt_med = float(np.median(dt)) if dt.size else 60.0
    fmax_Hz = fmax_factor / max(dt_med, 1e-6)
    fmax_uHz = fmax_Hz * 1e6
    fmin_uHz = (1.0 / (Pmax_days * 86400.0)) * 1e6

    # Split by period thresholds
    split1_uHz = (1.0 / (30.0 * 86400.0)) * 1e6  # ~30 d
    split2_uHz = (1.0 / (8.0  * 86400.0)) * 1e6  # ~8 d

    bands: List[FreqBand] = []
    # Low (long period)
    bands.append(FreqBand(fmin_uHz, min(split1_uHz, fmax_uHz), S_low))
    # Mid
    bands.append(FreqBand(min(split1_uHz, fmax_uHz), min(split2_uHz, fmax_uHz), S_mid))
    # High (short period)
    bands.append(FreqBand(min(split2_uHz, fmax_uHz), fmax_uHz, S_high))

    # Filter zero-length bands
    bands = [b for b in bands if b.fmax_uHz > b.fmin_uHz]
    return bands

