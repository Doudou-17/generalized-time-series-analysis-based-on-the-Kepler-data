# ==============================
"""
Utilities for residual construction and data segment selection.
Units policy: time=sec, freq=μHz.
This module centralizes common steps so Module 7 (residual scan) and Module 8 (significance)
can stay thin orchestrators.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np

SUFFIX_MAP = {"before": "_before", "after": "_after", "full": ""}

@dataclass
class DataBundle:
    time: np.ndarray
    flux_raw: np.ndarray
    suffix: str


def select_data_segment(npz_prep_path: str, data_source: str) -> DataBundle:
    """
    Load ./output/cleaned_segments.npz and select one of: 'full' | 'before' | 'after'.
    Returns DataBundle(time, flux_raw, suffix).
    - For 'full': time_full_sec + flux_full_global_demean
    - For 'before': time1 + flux1 (already per-segment demeaned)
    - For 'after' : time2 + flux2 (already per-segment demeaned)
    """
    z = np.load(npz_prep_path)
    ds = (data_source or 'full').lower()
    if ds == 'before':
        time = z["time1"].astype(np.float64)
        flux = z["flux1"].astype(np.float64)
    elif ds == 'after':
        time = z["time2"].astype(np.float64)
        flux = z["flux2"].astype(np.float64)
    else:
        time = z["time_full_sec"].astype(np.float64)
        flux = z["flux_full_global_demean"].astype(np.float64)
        ds = 'full'
    suffix = SUFFIX_MAP.get(ds, "_full")
    return DataBundle(time=time, flux_raw=flux, suffix=suffix)


def apply_detrend(time: np.ndarray, flux_raw: np.ndarray, deg: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply polynomial detrend of given degree (0,1,2). If deg=0, returns flux_raw and coef=None.
    """
    if deg in (1, 2):
        t0 = float(np.mean(time))
        tt = time - t0
        coef = np.polyfit(tt, flux_raw, deg=deg)
        trend = np.polyval(coef, tt)
        return flux_raw - trend, coef
    return flux_raw, None


def rebuild_model(time: np.ndarray, freqs_uHz: np.ndarray, C: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Multi-sine model reconstruction per LS design:
      y_model(t) = Σ_m [ C_m cos(2π f_m t) + S_m sin(2π f_m t) ]
    (time in sec, freqs in μHz)
    """
    TWOPI = 2.0 * np.pi
    y_model = np.zeros_like(time, dtype=np.float64)
    for Cm, Sm, f_uHz in zip(C, S, freqs_uHz):
        f_Hz = float(f_uHz) * 1e-6
        ph = (TWOPI * f_Hz * time) % TWOPI
        y_model += Cm * np.cos(ph) + Sm * np.sin(ph)
    return y_model


def compute_residual(npz_prep_path: str, fit_npz_path: str, data_source: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Build residuals using the fit file from Module 5, guaranteeing detrend consistency.
    Returns: time_in, resid, meta dict (includes flux_in used, freqs_uHz, detrend_deg, suffix).
    """
    # select data segment
    db = select_data_segment(npz_prep_path, data_source)
    time_in, flux_raw, suffix = db.time, db.flux_raw, db.suffix

    fit = np.load(fit_npz_path)
    freqs = fit["freqs_uHz"].astype(np.float64)
    C = fit["C"].astype(np.float64)
    S = fit["S"].astype(np.float64)
    detrend_deg = int(fit.get("meta_detrend_deg", 0))

    # detrend consistent with fit
    flux_in, coef = apply_detrend(time_in, flux_raw, detrend_deg)

    # model & residual
    y_model = rebuild_model(time_in, freqs, C, S)
    resid = flux_in - y_model

    meta = {
        "suffix": suffix,
        "freqs_uHz": freqs,
        "detrend_deg": detrend_deg,
        "coef": coef,
        "flux_in": flux_in,
    }
    return time_in, resid, meta


def summarize_timebase(time: np.ndarray) -> Tuple[float, float]:
    """Return (T_sec, rayleigh_uHz=1/T * 1e6)."""
    T = float(np.max(time) - np.min(time))
    rayleigh_uHz = (1.0 / T) * 1e6 if T > 0 else np.inf
    return T, rayleigh_uHz
