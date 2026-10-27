"""Parametric one-day portfolio VaR and diversification diagnostics.

The input covariance matrix is ANNUALIZED (252 trading sessions per year).
The calculation uses the optimizer's theoretical long-only portfolio weights.
Expected daily returns are deliberately excluded from this volatility-only VaR.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def diversification_var(weights, covariance, capital, confidence=0.99, trading_days=252):
    """Return one-day diversified/undiversified Gaussian VaR in portfolio currency."""
    if not 0.5 < confidence < 1.0:
        raise ValueError("Confidence must be between 0.5 and 1.")
    if capital <= 0 or trading_days <= 0:
        raise ValueError("Capital and trading_days must be positive.")
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("Covariance must be a labeled pandas DataFrame.")
    w = pd.Series(weights, dtype=float)
    if not w.index.is_unique or not covariance.index.is_unique or not covariance.columns.is_unique:
        raise ValueError("Asset labels must be unique.")
    missing = w.index.difference(covariance.index).union(w.index.difference(covariance.columns))
    if len(missing):
        raise ValueError(f"Covariance is missing assets: {list(missing)}")
    w = w.reindex(covariance.index)
    if w.isna().any():
        # Covariance may contain assets not held by the portfolio.
        w = w.fillna(0.0)
    if not np.isfinite(w.to_numpy()).all() or (w < -1e-8).any():
        raise ValueError("Long-only weights must be finite and nonnegative.")
    if not np.isclose(w.sum(), 1.0, atol=1e-4):
        raise ValueError(f"Portfolio weights must sum to 1 (got {w.sum():.8f}).")
    cov = covariance.loc[w.index, w.index].to_numpy(dtype=float)
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError("Covariance must be finite and symmetric.")
    daily_cov = cov / trading_days
    diagonal = np.diag(daily_cov)
    if (diagonal < -1e-12).any():
        raise ValueError("Covariance diagonal contains negative variances.")
    weighted_var = float(w.to_numpy() @ daily_cov @ w.to_numpy())
    if weighted_var < -1e-12:
        raise ValueError("Portfolio variance is negative.")
    z = float(norm.ppf(confidence))
    diversified = z * capital * np.sqrt(max(0.0, weighted_var))
    undiversified = z * capital * float(np.dot(w.to_numpy(), np.sqrt(np.maximum(diagonal, 0.0))))
    benefit = 100.0 * (1.0 - diversified / undiversified) if undiversified > 0 else 0.0
    return {"diversified": float(diversified), "undiversified": float(undiversified),
            "benefit_pct": float(benefit), "confidence": confidence, "horizon_days": 1}
