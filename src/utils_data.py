# -*- coding: utf-8 -*-
"""
Data utilities
Unit convention: time = seconds (sec), frequency = μHz (used only in later modules)
No QUALITY filtering; only NaN removal, so that downstream modules can control filtering.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

def standardize_lightcurve_to_npz(lc, npz_path: str,
                                  target: str = "UNKNOWN",
                                  mission: str = "Kepler",
                                  quarter: Optional[str] = None) -> None:
    """
    Standardize a lightkurve LightCurve object into .npz format:
    - time: BKJD (day) -> sec
    - flux, flux_err (if available)
    - quality (if available)
    - Metadata: time_unit='sec', freq_unit='μHz'
    No detrending or mean subtraction; only NaN removal is performed.
    """
    # --- Extract from LightCurve (preferred) ---
    time_day = np.asarray(lc.time.value, dtype=np.float64)       # days (BKJD)
    flux     = np.asarray(lc.flux.value, dtype=np.float64)       # e-/s
    # Optional arrays
    try:
        flux_err = (None if getattr(lc, "flux_err", None) is None
                    else np.asarray(lc.flux_err.value, dtype=np.float64))
    except Exception:
        flux_err = None
    try:
        quality = (None if getattr(lc, "quality", None) is None
                   else np.asarray(lc.quality, dtype=np.int32))
    except Exception:
        quality = None

    # --- Minimal cleaning: remove NaNs from time/flux ---
    mask = np.isfinite(time_day) & np.isfinite(flux)
    time_day = time_day[mask]
    flux     = flux[mask]
    if flux_err is None:
        flux_err = np.full_like(flux, np.nan, dtype=np.float64)
    else:
        flux_err = flux_err[mask]
    if quality is None:
        quality = np.zeros_like(flux, dtype=np.int32)
    else:
        quality = quality[mask]

    # --- Unit conversion: day -> sec ---
    time_sec = time_day * 86400.0

    np.savez(
        npz_path,
        time_sec=time_sec.astype(np.float64),
        flux=flux.astype(np.float64),
        flux_err=flux_err.astype(np.float64),
        quality=quality.astype(np.int32),
        meta_target=str(target),
        meta_mission=str(mission),
        meta_quarter="" if quarter is None else str(quarter),
        meta_time_unit="sec",
        meta_freq_unit="μHz",
        meta_note="Standardized from LightCurve; NaNs removed; no detrend/mean-subtraction."
    )

def load_raw_bundle(npz_path: str):
    """Load the standardized raw dataset (sec + μHz convention) exported by Module 0."""
    z = np.load(npz_path)
    return (z["time_sec"].astype(np.float64),
            z["flux"].astype(np.float64),
            z["flux_err"].astype(np.float64),
            z["quality"].astype(np.int32),
            {k: z[k].item() for k in z.files if k.startswith("meta_")})

def drop_nan(time_sec: np.ndarray, flux: np.ndarray):
    """Remove NaNs only (no QUALITY filtering)."""
    m = np.isfinite(time_sec) & np.isfinite(flux)
    return time_sec[m], flux[m]

def find_largest_gap(time_sec: np.ndarray) -> Tuple[int, float, float]:
    """
    Find the largest gap between consecutive samples.
    Returns: (index on the left side of the gap i, gap size Δt(sec), split threshold = time_sec[i])
    """
    dt = np.diff(time_sec)
    i  = int(np.argmax(dt))
    return i, float(dt[i]), float(time_sec[i])

def split_by_threshold(time_sec: np.ndarray, flux: np.ndarray, gap_time_sec: float):
    """Split into two segments using a given threshold (< threshold = first segment, others = second segment)."""
    before = time_sec < gap_time_sec
    return (time_sec[before], flux[before],
            time_sec[~before], flux[~before])

def demean(x: np.ndarray) -> np.ndarray:
    """Subtract the mean (no detrending)."""
    return x - np.mean(x)

def save_segments(out_npz: str,
                  time1, flux1, time2, flux2,
                  time_full_sec, flux_full_global_demean,
                  gap_time_sec: float) -> None:
    """Save segmented and full datasets for downstream DFT usage."""
    np.savez(
        out_npz,
        time1=time1, flux1=flux1,
        time2=time2, flux2=flux2,
        time_full_sec=time_full_sec,
        flux_full_global_demean=flux_full_global_demean,
        gap_time_sec=float(gap_time_sec),
        meta_time_unit="sec",
        meta_freq_unit="μHz",
        meta_note="Per-segment demean; kept real gap; also saved full concatenated arrays."
    )
