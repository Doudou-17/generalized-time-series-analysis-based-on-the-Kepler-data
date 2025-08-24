# ==============================
"""
Uncertainty utilities: propagate (C,S) covariance to (A,phi); frequency error formulas.
Units: time=sec, freq=μHz.
"""
from __future__ import annotations
import numpy as np


def amp_phi_errors_from_cov(C: float, S: float, cov2x2: np.ndarray, A_stored: float | None = None) -> tuple[float, float]:
    """
    Propagate 2x2 cov(C,S) to (sigma_A, sigma_phi).
    If A_stored is provided and differs by a scale k from sqrt(C^2+S^2), use k=A_stored/R.
    Formulas (Jacobian):
      R = sqrt(C^2+S^2);  k = A_stored/R if given else 1
      ∂A/∂C = k * C/R,  ∂A/∂S = k * S/R
      φ = atan2(-S, C)
      ∂φ/∂C =  S / R^2,  ∂φ/∂S = -C / R^2
    """
    R = float(np.hypot(C, S))
    if R <= 0:
        return float("nan"), float("nan")
    k = (A_stored / R) if (A_stored is not None) else 1.0
    dA_dC = k * C / R
    dA_dS = k * S / R
    dphi_dC =  S / (R * R)
    dphi_dS = -C / (R * R)
    J_A   = np.array([dA_dC,  dA_dS],  dtype=float)
    J_phi = np.array([dphi_dC, dphi_dS], dtype=float)
    sigma_A2   = float(J_A @ cov2x2 @ J_A)
    sigma_phi2 = float(J_phi @ cov2x2 @ J_phi)
    return float(np.sqrt(max(sigma_A2, 0.0))), float(np.sqrt(max(sigma_phi2, 0.0)))


def sigma_nu_rayleigh_snr(T_sec: float, SNR: float) -> float:
    """Frequency uncertainty (μHz) via σν ≈ sqrt(6) / (π T SNR)."""
    if T_sec <= 0 or not np.isfinite(SNR) or SNR <= 0:
        return float("nan")
    return (np.sqrt(6.0) / (np.pi * T_sec * SNR)) * 1e6

