"""Point-in-time input preparation and portfolio volatility analytics."""
from dataclasses import dataclass
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd

from src.forward_mc import covariance, nav_paths, validate_covariance
from src.volatility_forecasting import ANNUALIZATION, arch_model

# Registry of current optimization outputs. Historical backtest state is excluded.
STRATEGY_SOURCES = {
    "Maximum Sharpe": ("optimization_results", "max_weights"),
    "Minimum Volatility": ("optimization_results", "min_weights"),
    "Risk Parity": ("optimization_results", "risk_parity_weights"),
    "HRP": ("optimization_results", "hrp_weights"),
    "CAPM Maximum Sharpe": ("factor_model_results", "weights"),
    "FF3 Maximum Sharpe": ("ff3_results", "weights"),
    "FF3 Factor Cov Maximum Sharpe": ("ff3_factor_results", "weights"),
    "FF5 Maximum Sharpe": ("ff5_results", "weights"),
    "FF5 Factor Cov Maximum Sharpe": ("ff5_factor_results", "weights"),
    "Equal Weight": ("optimization_results", None),
}


@dataclass
class Portfolio:
    weights: pd.Series
    source: str
    as_of: pd.Timestamp
    optimized_at: str | None
    max_weight: float | None
    configuration: dict | None = None


def available_portfolios(state):
    portfolios = {}
    for name, (source, field) in STRATEGY_SOURCES.items():
        stored = state.get(source)
        if not stored:
            continue
        prices = stored.get("prices")
        weights = stored.get(field) if field else None
        if field is None and isinstance(prices, pd.DataFrame) and len(prices.columns):
            weights = pd.Series(1/len(prices.columns), index=prices.columns)
        if not isinstance(weights, pd.Series) or weights.empty:
            continue
        as_of = prices.index.max() if isinstance(prices, pd.DataFrame) else pd.NaT
        portfolios[name] = Portfolio(weights, source, as_of, stored.get("optimized_at"),
                                     stored.get("max_weight") if field else None,
                                     (stored.get("selection_configuration") or {}).get("effective_universe"))
    ml = state.get("factor_ml_compare")
    if ml:
        table = ml[0]["weights"]
        for label, rows in table.groupby("Strategy"):
            portfolios[f"ML comparison · {label}"] = Portfolio(
                rows.set_index("Ticker")["Weight"], "factor_ml_compare", pd.Timestamp(ml[0]["as_of"]), None, None,
            )
    return portfolios


def valid_weights(weights, prices):
    if (not weights.index.is_unique or not np.isfinite(weights).all()
            or (weights < -1e-10).any() or not np.isclose(weights.sum(), 1, atol=1e-6)):
        raise ValueError("Expected unique, finite long-only target weights summing to one.")
    weights = weights[weights > 1e-10]
    if not weights.index.isin(prices.columns).all():
        raise ValueError("Missing prices for one or more held assets.")
    return weights / weights.sum()


def dated_returns(prices, date):
    sample = prices.sort_index().loc[:pd.Timestamp(date)]
    if sample.empty or not sample.index.is_unique:
        raise ValueError("No prices or duplicate dates at the analysis cutoff.")
    sample = sample.where(np.isfinite(sample) & sample.gt(0))
    return sample.pct_change(fill_method=None).iloc[1:]


def complete_suffix(returns):
    gaps = np.flatnonzero(returns.isna().any(axis=1))
    return returns.iloc[gaps[-1]+1:] if len(gaps) else returns


def reconstruct_returns(returns, weights, rebalance):
    values = nav_paths(returns.to_numpy()[None, :, :], weights.to_numpy()[:, None], 1., rebalance)[0,:,0]
    return pd.Series(values[1:]/values[:-1]-1, index=returns.index, name="Reconstructed return")


def target_risk(returns, weights, method, decay):
    matrix = validate_covariance(covariance(returns.to_numpy(), method, decay)) * ANNUALIZATION
    cov = pd.DataFrame(matrix, index=returns.columns, columns=returns.columns)
    weights = weights.reindex(cov.index)
    product = cov @ weights
    sigma = np.sqrt(max(0., float(weights @ product)))
    marginal = product / sigma if sigma else pd.Series(np.nan, index=weights.index)
    component = weights*marginal if sigma else weights*0
    return cov, sigma, pd.DataFrame({
        "Portfolio Weight":weights, "Marginal Risk Contribution":marginal,
        "Risk Contribution":component, "Percentage Risk Contribution":component/sigma if sigma else np.nan,
    })


def estimate(returns, model, horizon, window, decay):
    """Daily variances and diagnostic parameters. EWMA trace uses prior returns."""
    values = returns.to_numpy()
    if len(values) < window or not np.isfinite(values).all():
        raise ValueError(f"Need {window} finite daily returns.")
    if model in ("Historical", "Rolling"):
        sample = returns if model == "Historical" else returns.tail(window)
        return None, np.repeat(sample.var(), horizon), {}
    if model == "EWMA":
        forecast = values[0]**2
        filtered = [np.nan]
        for value in values[1:]:
            filtered.append(forecast)
            forecast = decay*forecast + (1-decay)*value**2
        return pd.Series(filtered,index=returns.index), np.repeat(forecast,horizon), {"lambda":decay}
    if arch_model is None or len(values) < 150 or np.std(values) <= 0:
        raise ValueError("GARCH requires arch and at least 150 nonconstant observations.")
    with catch_warnings():
        simplefilter("ignore")
        fit = arch_model(values*100, mean="Zero", vol="EGARCH" if model=="EGARCH" else "GARCH",
                         p=1, o=int(model=="GJR-GARCH"), q=1, dist="t", rescale=False).fit(disp="off",show_warning=False)
    p = fit.params
    if fit.convergence_flag or not np.isfinite(p).all() or p["nu"] <= 2:
        raise ValueError(f"{model} did not converge to valid Student-t parameters.")
    if model=="EGARCH":
        valid = abs(p["beta[1]"]) < 1
    else:
        alpha, beta, gamma = p["alpha[1]"], p["beta[1]"], p.get("gamma[1]",0.)
        valid = p["omega"] > 0 and min(alpha,beta,alpha+gamma) >= 0 and alpha+beta+gamma/2 < 1
    if not valid:
        raise ValueError(f"{model} parameters violate stationarity or positivity.")
    kwargs = {}
    if model=="EGARCH" and horizon>1:
        rng = np.random.default_rng(42)
        nu = p["nu"]
        kwargs = dict(method="simulation", simulations=1000,
                      rng=lambda size: rng.standard_t(nu,size)*np.sqrt((nu-2)/nu))
    path = fit.forecast(horizon=horizon,reindex=False,**kwargs).variance.iloc[-1].to_numpy()/10000
    if not np.isfinite(path).all() or (path <= 0).any():
        raise ValueError(f"{model} returned invalid forecast variance.")
    return pd.Series(np.asarray(fit.conditional_volatility)**2/10000,index=returns.index), path, p.to_dict()


def validate_models(returns, models, horizon, train, stride, window, decay):
    rows = []
    skipped = {}
    for end in range(train,len(returns)-horizon+1,stride):
        future = returns.iloc[end:end+horizon]
        row = {"Origin":returns.index[end-1], "Observed variance":(future**2).mean()}
        for model in models:
            try:
                row[model] = estimate(returns.iloc[end-train:end],model,horizon,window,decay)[1].mean()
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                skipped[model] = str(exc)
        rows.append(row)
    forecasts = pd.DataFrame(rows)
    scores = []
    if len(rows):
        for model in models:
            if model in forecasts and forecasts[model].notna().all():
                variance = forecasts[model]
                y = forecasts["Observed variance"]
                q = variance.clip(lower=1e-12)
                scores.append(dict(Model=model, QLIKE=(np.log(q)+y/q).mean(),
                                   MSE=((variance-y)**2).mean(), Windows=len(y)))
    return forecasts,pd.DataFrame(scores),skipped
