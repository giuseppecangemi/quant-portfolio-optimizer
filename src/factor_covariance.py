"""Point-in-time factor-implied annualized covariance in the asset native currency.

EUR equity excess returns and converted EUR factor returns are used for both expected return and risk estimation.
"""
import numpy as np
import pandas as pd
from src.factor_currency import factor_sample

FACTOR_NAMES = {"ff3": ("MKT-RF", "SMB", "HML"),
                "ff5": ("MKT-RF", "SMB", "HML", "RMW", "CMA")}


def estimate_factor_covariance(prices: pd.DataFrame, factors: pd.DataFrame,
                               *, model: str, annualization: int = 252,
                               min_observations: int = 80,
                               eurusd_prices=None, market="USA") -> pd.DataFrame:
    if model not in FACTOR_NAMES:
        raise ValueError(f"Unknown factor model {model}")
    if annualization <= 0:
        raise ValueError("Annualization must be positive")
    cols = list(FACTOR_NAMES[model])
    returns, available = factor_sample(prices, factors, eurusd_prices, market=market)
    aligned = returns.join(available[[*cols, "RF"]], how="inner").dropna()
    if len(aligned) < max(min_observations, len(cols) + 3):
        raise ValueError("Insufficient aligned observations for factor covariance")
    f = aligned[cols].to_numpy(float)
    y = aligned[returns.columns].to_numpy(float) - aligned["RF"].to_numpy(float)[:, None]
    x = np.column_stack([np.ones(len(f)), f])
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Factor design matrix is rank deficient")
    beta = coefficients[1:, :].T
    residuals = y - x @ coefficients
    factor_cov = np.cov(f, rowvar=False, ddof=1)
    residual_variance = np.sum(residuals ** 2, axis=0) / (len(f) - x.shape[1])
    cov = beta @ factor_cov @ beta.T + np.diag(residual_variance)
    cov = (cov + cov.T) / 2 * annualization
    if not np.isfinite(cov).all():
        raise ValueError("Non-finite factor covariance")
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
