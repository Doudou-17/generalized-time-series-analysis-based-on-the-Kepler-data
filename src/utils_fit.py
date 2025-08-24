# =========================
# Multi-sine weighted least squares (time in sec, freq in μHz)
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


TWOPI = 2.0 * np.pi


def fit_multi_sine(time_sec: np.ndarray,
                   flux: np.ndarray,
                   freqs_uHz: np.ndarray,
                   flux_err: Optional[np.ndarray] = None) -> dict:
    """
    Weighted LS fit: y ~ sum_m [ C_m cos(2π f_m t) + S_m sin(2π f_m t) ]
    Returns dict with A, phi, C, S, cov, chi2, dof.
    """
    t = np.asarray(time_sec, dtype=np.float64)
    y = np.asarray(flux, dtype=np.float64)
    freqs_uHz = np.asarray(freqs_uHz, dtype=np.float64)
    M = freqs_uHz.size
    N = t.size

    # Design matrix
    Xcols = []
    for f_uHz in freqs_uHz:
        f_Hz = f_uHz * 1e-6
        phi = (TWOPI * f_Hz * t) % (TWOPI)
        Xcols.append(np.cos(phi))
        Xcols.append(np.sin(phi))
    X = np.column_stack(Xcols)  # (N, 2M)

    if flux_err is not None and np.any(np.isfinite(flux_err)):
        w = 1.0 / np.clip(flux_err, 1e-30, np.inf)
        W = np.diag(w)
        Xw = X * w[:, None]
        yw = y * w
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        # Covariance ~ (X^T W^2 X)^-1 (approx)
        XtWX = Xw.T @ Xw
        cov = np.linalg.pinv(XtWX)
        resid = y - X @ beta
        chi2 = float((w * resid**2).sum())
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        cov = np.linalg.pinv(X.T @ X)
        resid = y - X @ beta
        chi2 = float(np.sum(resid**2))

    dof = max(N - 2*M, 1)

    # Convert (C,S) to (A,phi)
    C = beta[0::2]
    S = beta[1::2]
    A = 2.0 * np.sqrt(C**2 + S**2) / N  # amplitude scaling similar to periodogram
    phi = np.arctan2(-S, C)  # match DFT convention

    return {
        "C": C, "S": S, "A": A, "phi": phi,
        "cov": cov, "resid": resid, "chi2": chi2, "dof": dof
    }

