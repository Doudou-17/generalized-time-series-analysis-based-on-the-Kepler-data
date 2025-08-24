# src/utils_phase.py
# -*- coding: utf-8 -*-
"""Phase folding utilities (time in sec, freq in μHz)."""
from __future__ import annotations
import numpy as np

def period_from_muHz(nu_muHz: float) -> float:
    """μHz → 周期（秒）"""
    nu_Hz = float(nu_muHz) * 1e-6
    if nu_Hz <= 0:
        raise ValueError("Frequency must be > 0.")
    return 1.0 / nu_Hz

def fold_phase(time_sec: np.ndarray, nu_muHz: float, t0_sec: float | None = None,
               wrap_center: float = 0.0) -> np.ndarray:
    """
    Compute phase: φ ∈ [0,1), or shifted to [-0.5, 0.5) if wrap_center=0.5.
    time_sec: time in seconds
    nu_muHz: frequency in μHz
    t0_sec: phase zero-point (default: time.min())
    """
    t = np.asarray(time_sec, dtype=np.float64)
    if t0_sec is None:
        t0_sec = float(t.min())
    nu_Hz = float(nu_muHz) * 1e-6
    phi = ((t - t0_sec) * nu_Hz) % 1.0
    if wrap_center == 0.5:  # Shift phase to [-0.5, 0.5)
        phi = ((phi + 0.5) % 1.0) - 0.5
    return phi

def auto_t0_min_flux(time_sec: np.ndarray, flux: np.ndarray, nu_muHz: float,
                     nbins: int = 200) -> float:
    """
    Determine phase zero-point as the "minimum flux" epoch:
    - Fold the light curve roughly with t0 = time.min().
    - Bin the folded curve and compute median flux in each bin.
    - Locate the bin with minimum flux and take its center as reference phase.
    - Find the observation time closest to this phase and adjust to make it phase zero.
    Suitable for eclipsing binaries / non-sinusoidal shapes.
    """
    t = np.asarray(time_sec, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64)
    nu_Hz = float(nu_muHz) * 1e-6
    # First fold with t0 = t.min()
    phi = ((t - t.min()) * nu_Hz) % 1.0
    # Bin the folded curve
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.digitize(phi, edges) - 1
    idx = np.clip(idx, 0, nbins - 1)
    med = np.full(nbins, np.nan)
    for b in range(nbins):
        m = (idx == b)
        if np.any(m):
            med[b] = np.median(f[m])
    bmin = int(np.nanargmin(med))
    phi0 = (edges[bmin] + edges[bmin + 1]) / 2.0
    phi0 = (edges[bmin] + edges[bmin + 1]) / 2.0
    # Force this phase to be zero: (t0) such that (t - t0)*nu_Hz % 1 == 0
    # Approximate using the closest observed time to that bin center
    cand = t[np.argmin(np.abs(phi - phi0))]
    # Ensure cand is phase zero: (cand - t0)*nu_Hz is integer → round to nearest integer
    k = np.rint((cand - t.min()) * nu_Hz - phi0)  
    t0 = cand - k / nu_Hz
    return float(t0)

def bin_folded(phi: np.ndarray, y: np.ndarray, nbins: int = 100):
    """
    Bin folded data in phase space.
    Returns: (bin_center, y_median, y_err)
    y_err is approximated using MAD/1.4826/sqrt(n).
    """
    phi = np.asarray(phi); y = np.asarray(y)
    edges = np.linspace(phi.min(), phi.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y_med = np.full(nbins, np.nan)
    y_err = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (phi >= edges[i]) & (phi < edges[i + 1])
        if np.any(m):
            vals = y[m]
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            n = m.sum()
            y_med[i] = med
            y_err[i] = (mad / 1.4826) / max(np.sqrt(n), 1.0)
    return centers, y_med, y_err
