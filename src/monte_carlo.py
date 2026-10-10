"""Forward scenario simulation from aligned, historical out-of-sample portfolio returns.

This module is deliberately separate from the portfolio construction/backtest engines.
Simulations are conditional on the empirical return distribution; not price forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def extract_equity(result: dict, key: str) -> pd.Series | None:
    raw = result.get(key)
    if isinstance(raw, pd.DataFrame):
        if "Portfolio Value" not in raw:
            return None
        raw = raw["Portfolio Value"]
    if not isinstance(raw, pd.Series):
        return None
    equity = pd.to_numeric(raw, errors="coerce").copy()
    equity.index = pd.to_datetime(equity.index, errors="coerce")
    equity = equity.loc[~equity.index.isna()]
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    if len(equity) < 3 or not np.isfinite(equity).all() or (equity <= 0).any():
        return None
    return equity


def aligned_returns(equities: dict[str, pd.Series]) -> pd.DataFrame:
    if not equities:
        raise ValueError("Nessuna curva storica valida disponibile.")
    # Calculate returns BEFORE matching calendars to avoid spanning missing days.
    daily = pd.concat({name: s.pct_change(fill_method=None) for name, s in equities.items()}, axis=1)
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(daily) < 40:
        raise ValueError(f"Solo {len(daily)} rendimenti giornalieri comuni: servono almeno 40 sedute.")
    if (daily <= -1).any().any():
        raise ValueError("La serie contiene rendimenti giornalieri non validi (<= -100%).")
    return daily


def simulate(
    historical_returns: pd.DataFrame,
    *,
    years: int = 1,
    n_paths: int = 1000,
    initial_capital: float = 10000.0,
    method: str = "Block bootstrap",
    block_size: int = 5,
    seed: int = 42,
    annual_drift_adjustment: float = 0.0,
    return_paths: bool = False,
) -> tuple:
    """Return (metrics, percentile_paths, final_values) for all selected strategies.

    Same sampled historical days / multivariate shocks across all strategies.
    The empirical mean is retained; optional drift adjustment is in annual decimal units.
    """
    if not (1 <= years <= 10 and 100 <= n_paths <= 10000 and initial_capital > 0):
        raise ValueError("Parametri di simulazione non validi.")
    hist = historical_returns.to_numpy(dtype=float)
    if hist.ndim != 2 or len(hist) < 40 or not np.isfinite(hist).all():
        raise ValueError("Serie storiche insufficienti o non finite.")
    n_days = years * TRADING_DAYS
    rng = np.random.default_rng(seed)
    if method == "Block bootstrap":
        block_size = max(1, min(int(block_size), len(hist)))
        n_blocks = (n_days + block_size - 1) // block_size
        starts = rng.integers(0, len(hist) - block_size + 1, size=(n_paths, n_blocks))
        indices = (starts[:, :, None] + np.arange(block_size)).reshape(n_paths, -1)[:, :n_days]
        shocks = hist[indices]
    elif method == "IID bootstrap":
        indices = rng.integers(0, len(hist), size=(n_paths, n_days))
        shocks = hist[indices]
    elif method == "Multivariate Gaussian":
        mu = hist.mean(axis=0)
        cov = np.atleast_2d(np.cov(hist, rowvar=False))
        cov = (cov + cov.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        cov = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        shocks = rng.multivariate_normal(mu, cov, size=(n_paths, n_days), check_valid="ignore")
    else:
        raise ValueError(f"Metodo sconosciuto: {method}")
    # Approximate additive adjustment to expected daily arithmetic returns.
    shocks = shocks + float(annual_drift_adjustment) / TRADING_DAYS
    if np.any(shocks <= -1):
        raise ValueError("Uno scenario genera una perdita giornaliera >=100%; riduci lo stress sulla media.")
    # Include initial capital at t=0.
    growth = np.concatenate([np.ones((n_paths, 1, hist.shape[1])), np.cumprod(1 + shocks, axis=1)], axis=1)
    values = growth * initial_capital
    final = values[:, -1, :]
    running_peak = np.maximum.accumulate(values, axis=1)
    max_dd = (values / running_peak - 1).min(axis=1)
    metrics = []
    pct_paths = {}
    final_values = {}
    for j, name in enumerate(historical_returns.columns):
        terminal = final[:, j]
        pnl = terminal - initial_capital
        cutoff = float(np.quantile(pnl, 0.05))
        tail = pnl[pnl <= cutoff]
        metrics.append({
            "Strategy": name,
            "Median final (€)": float(np.median(terminal)),
            "Mean final (€)": float(np.mean(terminal)),
            "P5 final (€)": float(np.quantile(terminal, 0.05)),
            "P95 final (€)": float(np.quantile(terminal, 0.95)),
            "Probability of loss (%)": float(np.mean(terminal < initial_capital) * 100),
            "VaR 95% (€)": float(max(0, -cutoff)),
            "ES 95% (€)": float(max(0, -tail.mean())) if len(tail) else 0.0,
            "Probability MDD ≤ -20% (%)": float(np.mean(max_dd[:, j] <= -0.20) * 100),
        })
        p = np.percentile(values[:, :, j], [5, 25, 50, 75, 95], axis=0)
        pct_paths[name] = pd.DataFrame(p.T, columns=["P5", "P25", "Median", "P75", "P95"], index=pd.RangeIndex(n_days + 1, name="Trading day"))
        final_values[name] = terminal
    result = (pd.DataFrame(metrics).set_index("Strategy"), pct_paths, pd.DataFrame(final_values))
    if return_paths:
        # Shape: (simulation, trading_day, strategy); store compactly for plotting.
        return (*result, {name: values[:, :, j].astype(np.float32) for j, name in enumerate(historical_returns.columns)})
    return result
