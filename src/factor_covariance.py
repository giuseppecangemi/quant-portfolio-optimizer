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
                               eurusd_prices=None, market="USA", return_components=False):
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
    covariance = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    if not return_components:
        return covariance
    return covariance, {
        "loadings": pd.DataFrame(beta, index=returns.columns, columns=cols),
        "factor_covariance": pd.DataFrame(factor_cov * annualization, index=cols, columns=cols),
        "residual_covariance": pd.DataFrame(np.diag(residual_variance * annualization),
                                            index=returns.columns, columns=returns.columns),
        "observations": len(aligned), "annualization": annualization,
        "sample_start": aligned.index[0].isoformat(), "sample_end": aligned.index[-1].isoformat(),
        "factor_cutoff": factors.index.max().isoformat(),
    }


def decompose_factor_risk(weights, components):
    """Signed Euler contributions to annual variance using post-trade target weights."""
    b = components["loadings"]
    f = components["factor_covariance"]
    o = components["residual_covariance"]
    if (not weights.index.is_unique or not np.isfinite(weights).all()
            or not np.isclose(weights.sum(), 1., atol=1e-7)
            or (weights.drop(index=b.index, errors="ignore").abs() > 1e-10).any()
            or len(b.index.difference(weights.index)) or not b.index.is_unique
            or not b.columns.is_unique or min(b.shape) < 1
            or not all(axis.is_unique for axis in (f.index, f.columns, o.index, o.columns))
            or set(f.index) != set(b.columns)
            or set(f.columns) != set(b.columns) or set(o.index) != set(b.index)
            or set(o.columns) != set(b.index) or not np.isfinite(b.to_numpy()).all()):
        raise ValueError("Factor risk inputs are invalid or contain inconsistent asset or factor labels.")
    w = weights.loc[b.index].to_numpy(float)
    f = f.loc[b.columns, b.columns].to_numpy(float)
    o = o.loc[b.index, b.index].to_numpy(float)
    for matrix in (f, o):
        scale = max(np.max(np.abs(matrix)), np.finfo(float).tiny)
        if (not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-10*scale, rtol=0)
                or np.linalg.eigvalsh(matrix)[0] < -1e-10*scale):
            raise ValueError("Invalid factor or residual covariance: expected symmetric PSD matrices.")
    beta = b.to_numpy().T @ w
    contributions = beta * (f @ beta)
    systematic = float(contributions.sum())
    residual = float(w @ o @ w)
    total = float(w @ (b.to_numpy() @ f @ b.to_numpy().T + o) @ w)
    difference = total - systematic - residual
    if not np.isclose(total, systematic+residual, rtol=1e-9, atol=1e-13) or total < -1e-13:
        raise ValueError("Risk contributions do not reconcile to a nonnegative total variance.")
    return {**{k: components[k] for k in ("observations", "annualization", "sample_start", "sample_end", "factor_cutoff")
               if k in components},
            "assets": len(w), "volatility": float(np.sqrt(max(total, 0.))),
            "systematic": systematic, "residual": residual, "total": total, "difference": difference,
            "exposures": dict(zip(b.columns, map(float, beta))),
            "contributions": dict(zip([*b.columns, "Residual"], map(float, [*contributions, residual])))}


def factor_risk_record(weights, components, optimizer_covariance, cutoff, uses_factor_cov):
    """Capture a compact diagnostic alongside the actual optimizer risk."""
    record = decompose_factor_risk(weights, components)
    w = weights.reindex(optimizer_covariance.index).fillna(0.)
    variance = float(w @ optimizer_covariance @ w)
    return {**record, "optimizer_variance": variance, "decision_cutoff": cutoff.isoformat(),
            "uses_factor_cov": uses_factor_cov}
