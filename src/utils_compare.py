"""
Utilities for before/after cross‑matching and direct DFT diagnostics.
Units: time=sec, freq=μHz.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

# By your preference: full has NO suffix
SUFFIX_MAP = {"before": "_before", "after": "_after", "full": ""}


def read_peaks_txt(path: str) -> np.ndarray:
    """Read a peaks txt (either original or residual). Returns refined μHz as 1D float array.
    Accepts formats with columns like: i  f_bin_uHz  f_refined_uHz  ...
    Fallback to 2nd column if needed.
    """
    freqs: List[float] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 3:
                try:
                    freqs.append(float(parts[2]))
                    continue
                except ValueError:
                    pass
            if len(parts) >= 2:
                try:
                    freqs.append(float(parts[1]))
                except ValueError:
                    pass
    return np.array(freqs, dtype=np.float64)


def direct_fr_fi(time_sec: np.ndarray, flux: np.ndarray, freq_uHz: float, weights: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Direct (non‑recursive) FR/FI at a single frequency (μHz).
    FR(ν)=Σ w f cos(2πνt), FI(ν)=Σ w f sin(2πνt)  — Kurtz/DFT definition.
    """
    t = np.asarray(time_sec, dtype=np.float64)
    y = np.asarray(flux, dtype=np.float64)
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=np.float64)
    twopi = 2.0 * np.pi
    f_Hz = freq_uHz * 1e-6
    ph = (twopi * f_Hz * t) % (2.0 * np.pi)
    c = np.cos(ph)
    s = np.sin(ph)
    FR = float(np.sum(w * y * c))
    FI = float(np.sum(w * y * s))
    return FR, FI


def amp_phase_from_frfi(FR: float, FI: float, w_sum: float | None = None) -> Tuple[float, float]:
    """
    Amplitude & phase from FR/FI. Amplitude normalized like in our Kurtz scan:
      A = (2/Σw) * sqrt(FR^2 + FI^2) if w_sum provided, else A = 2*sqrt(...) / N
    Phase: φ = atan2(-FI, FR)
    """
    R = np.hypot(FR, FI)
    if w_sum is None:
        A = 2.0 * R  # caller can divide by N outside if needed
    else:
        A = (2.0 / max(w_sum, 1e-30)) * R
    phi = float(np.arctan2(-FI, FR))
    return A, phi

