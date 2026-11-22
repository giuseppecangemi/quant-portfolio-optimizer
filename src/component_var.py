"""One-session, zero-mean Gaussian VaR and Euler attribution."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def component_var(exposure, daily_covariance, confidence=.99):
    if not .5 < confidence < 1:
        raise ValueError("Invalid VaR confidence.")
    if (exposure.empty or not exposure.index.is_unique
            or not daily_covariance.index.is_unique or not daily_covariance.columns.is_unique):
        raise ValueError("Expected nonempty, uniquely labeled exposures and covariance.")
    cov = daily_covariance.reindex(index=exposure.index, columns=exposure.index).to_numpy(dtype=float)
    x = exposure.to_numpy(dtype=float)
    if not np.isfinite(cov).all() or not np.isfinite(x).all():
        raise ValueError("Missing or nonfinite exposures or covariance.")
    scale = max(np.abs(cov).max(), np.finfo(float).tiny)
    tolerance = 1e-10 * scale
    if not np.allclose(cov, cov.T, rtol=0, atol=tolerance):
        raise ValueError("Covariance must be symmetric.")
    cov = (cov + cov.T) / 2
    if np.linalg.eigvalsh(cov)[0] < -tolerance:
        raise ValueError("Covariance must be positive semidefinite.")
    product = cov @ x
    variance = float(x @ product)
    if variance <= np.finfo(float).eps * scale * (x @ x):
        marginal = np.full(len(x), np.nan)
        components = np.zeros(len(x))
        total = 0.
    else:
        sigma = np.sqrt(variance)
        z = norm.ppf(confidence)
        marginal = z * product / sigma
        components = x * marginal
        total = float(z * sigma)
    return pd.DataFrame({
        "Invested Amount (EUR)": x,
        "Marginal VaR (%)": marginal * 100,
        "Component VaR (EUR)": components,
        "Component VaR Contribution (%)": components / total * 100 if total else np.nan,
    }, index=exposure.index), total, float(components.sum() - total)


def allocation_var(table, annual_covariance):
    result = table.copy()
    values, total, _ = component_var(result["Invested"], annual_covariance / 252)
    result["Marginal VaR (%)"] = values["Marginal VaR (%)"]
    result["Component VaR (€)"] = values["Component VaR (EUR)"]
    return result, total


def dynamic_component_var(exposures, prices, window=252):
    """Use trailing complete daily returns through each close and actual marks."""
    if window < 2:
        raise ValueError("At least two returns are needed.")
    returns = prices.sort_index().pct_change(fill_method=None)
    returns = returns.where(np.isfinite(returns))
    records = []
    for date, exposure in exposures.iterrows():
        held = exposure[exposure.ne(0)]
        if held.empty:
            continue
        sample = returns.loc[:date].reindex(columns=held.index).tail(window)
        if len(sample) != window or sample.isna().any().any():
            continue
        values, total, _ = component_var(held, sample.cov())
        values.insert(0, "Asset", values.index)
        values.insert(0, "Date", date)
        values["Total Portfolio VaR (EUR)"] = total
        records.append(values)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
