"""Asset-level, common-scenario forward Monte Carlo."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.factor_covariance import FACTOR_NAMES
from src.factor_currency import configured_factor_inputs, factor_sample, factors_for_backtest_signal
from src.portfolio_context import check_minimum, portfolio_settings
from src.monte_carlo import empirical_tail_mean
from src.optimization import (
    optimize_hrp, optimize_maximum_sharpe, optimize_minimum_volatility, optimize_risk_parity,
)
from src.transaction_costs import fund_orders


def calibration(prices, date, window, benchmark=None, factor_inputs=None, market="Borsa Italiana", rf_daily=0.):
    date = pd.Timestamp(date)
    prices = prices.sort_index().loc[:date]
    if portfolio_settings():
        prices = prices.dropna()
    prices = prices.tail(window+1)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if len(returns) < window or not np.isfinite(returns.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError(f"Need {window} consecutive complete returns for every selected asset.")
    date = prices.index[-1]
    fits = {}
    if benchmark is not None:
        market_return = benchmark.reindex(prices.index).pct_change(fill_method=None).rename("Market")
        joined = returns.join(market_return).dropna()
        fits["CAPM"] = (joined[returns.columns].to_numpy(),
                        joined[["Market"]].to_numpy()-rf_daily,
                        np.full(len(joined), rf_daily), ["MKT-RF"])
    for name, (factors, fx) in (factor_inputs or {}).items():
        factors = factors_for_backtest_signal(factors, date)
        fx = fx.sort_index().loc[:date] if fx is not None else None
        if portfolio_settings():
            group, factors, fx, base_market, eur_rf = next(configured_factor_inputs(prices, factors, name, fx))
            assets, converted = factor_sample(group, factors, fx, market=base_market, eur_rf_mode=eur_rf)
        else:
            assets, converted = factor_sample(prices, factors, fx, market=market)
        names = list(FACTOR_NAMES[name.lower()])
        joined = assets.join(converted[[*names, "RF"]]).dropna()
        fits[name] = (joined[returns.columns].to_numpy(), joined[names].to_numpy(),
                      joined["RF"].to_numpy(), names)
    return returns, fits


def covariance(x, method, decay):
    if method == "Ledoit–Wolf":
        return LedoitWolf().fit(x).covariance_
    if method == "EWMA":
        w = decay ** np.arange(len(x)-1, -1, -1, dtype=float)
        w /= w.sum()
        centered = x - w @ x
        return (centered.T * w) @ centered
    return np.atleast_2d(np.cov(x, rowvar=False))


def validate_covariance(matrix):
    matrix = (matrix + matrix.T)/2
    if not np.isfinite(matrix).all():
        raise ValueError("Covariance is not finite.")
    eig, vectors = np.linalg.eigh(matrix)
    if eig.min() < -1e-10 * max(np.abs(eig).max(), np.finfo(float).tiny):
        raise ValueError("Covariance is not positive semidefinite.")
    return (vectors * np.maximum(eig, 0)) @ vectors.T if eig.min() < 0 else matrix


def fit_factor(data, indices=None):
    y, f, rf, names = data
    if indices is not None:
        y, f, rf = y[indices], f[indices], rf[indices]
    if len(f) < max(80, f.shape[1]+3):
        raise ValueError("Need at least 80 aligned observations for the factor model.")
    design = np.column_stack([np.ones(len(f)), f])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y-rf[:, None], rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Factor design is rank deficient.")
    beta = coefficients[1:].T
    residual = y-rf[:, None]-design @ coefficients
    return beta, f, rf.mean(), residual, names


def portfolio_weights(returns, fits, strategies, rf, cap, sessions):
    cov = returns.cov()*sessions
    labels = returns.columns
    historical = returns.mean()*sessions
    weights = {}
    unavailable = {}
    for strategy in strategies:
        try:
            if strategy == "Equal Weight":
                w = pd.Series(1/len(labels), index=labels)
            elif strategy == "Minimum Volatility":
                w = optimize_minimum_volatility(historical, cov, max_weight=cap)
            elif strategy == "Risk Parity":
                w = optimize_risk_parity(cov, max_weight=cap)
            elif strategy == "HRP":
                w = optimize_hrp(cov, max_weight=cap)
            else:
                name = strategy.split()[0]
                mu, risk = historical, cov
                if name != "Maximum":
                    beta, factors, rate, residual, _ = fit_factor(fits[name])
                    mu = pd.Series((rate + beta @ factors.mean(axis=0))*sessions, index=labels)
                    if "Factor Cov" in strategy:
                        matrix = beta @ covariance(factors, "Sample", .94) @ beta.T
                        matrix += np.diag((residual**2).sum(axis=0)/(len(factors)-factors.shape[1]-1))
                        risk = pd.DataFrame(matrix*sessions, index=labels, columns=labels)
                w = optimize_maximum_sharpe(mu, risk, risk_free_rate=rf, max_weight=cap)
            if not np.isfinite(w).all() or (w < -1e-8).any() or not np.isclose(w.sum(), 1):
                raise ValueError("Optimizer returned invalid weights.")
            check_minimum(w)
            weights[strategy] = w.clip(lower=0)/w.clip(lower=0).sum()
        except (ValueError, KeyError, RuntimeError) as exc:
            unavailable[strategy] = str(exc)
    return pd.DataFrame(weights).reindex(labels), unavailable


def block_indices(rng, n, paths, steps, length):
    starts = rng.integers(n-length+1, size=(paths, (steps+length-1)//length, 1))
    return (starts + np.arange(length)).reshape(paths, -1)[:, :steps]


def nav_paths(shocks, weights, capital, rebalance=0, costs=None):
    if np.any(shocks <= -1) or not np.isfinite(shocks).all():
        raise ValueError("A simulated asset return is nonfinite or ≤ −100%.")
    paths, steps, _ = shocks.shape
    result = np.full((paths, steps+1, weights.shape[1]), float(capital))
    for j in range(weights.shape[1]):
        w = weights[:, j]
        budget = np.full(paths, float(capital))
        previous = np.zeros((paths, len(w)))
        series = pd.Series(w)
        unit_prices = pd.Series(1., index=series.index)
        stride = rebalance or steps
        for start in range(0, steps, stride):
            if costs is not None:
                budget = np.array([
                    fund_orders(amount, series, pd.Series(held), unit_prices, costs)[0]
                    for amount, held in zip(budget, previous)
                ])
            stop = min(steps, start+stride)
            holdings = np.cumprod(1+shocks[:, start:stop], axis=1) * (budget[:, None, None]*w)
            result[:, start+1:stop+1, j] = holdings.sum(axis=2)
            previous = holdings[:, -1]
            budget = result[:, stop, j]
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Simulated NAV is outside numerical bounds.")
    return result


def calibrate_scenarios(returns, factor_data, covariance_method, decay, method):
    cov = validate_covariance(covariance(returns, covariance_method, decay))
    if factor_data is None:
        return returns.mean(axis=0), cov, None
    beta, factors, rf, residual, names = fit_factor(factor_data)
    mean = rf + beta @ factors.mean(axis=0)
    factor_cov = validate_covariance(covariance(factors, covariance_method, decay))
    residual_cov = validate_covariance(covariance(residual, covariance_method, decay))
    model = dict(beta=beta, premiums=factors.mean(axis=0), rf=rf, names=names,
                 factor_cov=factor_cov, residual_cov=residual_cov)
    return mean, beta @ factor_cov @ beta.T + residual_cov if method == "Factor Monte Carlo" else cov, model


def simulate_forward(returns, weights, *, model_data=None, method="Multivariate Normal",
                     covariance_method="Sample", decay=.94, block=5, paths=5000,
                     steps=252, sessions=252, capital=10000., seed=42, rf=0.,
                     rebalance=0, costs=None, parameter_draws=0):
    history = returns.to_numpy(float)
    n, assets = history.shape
    if (not 1 <= block <= n or len(weights.columns) == 0 or not weights.index.equals(returns.columns)
            or not np.isfinite(history).all() or not .0 < decay < 1 or not 0 < capital
            or not np.isfinite(capital) or min(paths, steps, sessions) <= 0):
        raise ValueError("Invalid simulation inputs.")
    if paths*(steps+1)*len(weights.columns) > 38_000_000:
        raise ValueError("This combination exceeds the memory budget. Reduce paths, horizon or strategies.")
    if costs is not None and paths*len(weights.columns)*((steps-1)//(rebalance or steps)+1) > 12_000:
        raise ValueError("Too many simulated cost calculations. Reduce paths, strategies or rebalancing frequency.")
    if method == "Factor Monte Carlo" and model_data is None:
        raise ValueError("Factor simulation requires CAPM, FF3 or FF5.")
    if method not in ("Multivariate Normal", "Historical Block Bootstrap", "Factor Monte Carlo"):
        raise ValueError("Unknown simulation method.")
    if (not np.isfinite(weights.to_numpy()).all() or (weights < 0).any().any()
            or not np.allclose(weights.sum(), 1)):
        raise ValueError("Weights must be finite, nonnegative and sum to one.")
    rng = np.random.default_rng(seed)
    baseline = calibrate_scenarios(history, model_data, covariance_method, decay, method)
    values = np.empty((paths, steps+1, len(weights.columns)), dtype=float)
    batch_limit = max(1, min(256, 2_000_000//(steps*assets)))
    for group in np.array_split(np.arange(paths), parameter_draws or 1):
        if not len(group):
            continue
        if parameter_draws:
            idx = block_indices(rng, n, 1, n, block)[0]
            sampled_factor = None
            if model_data is not None:
                size = len(model_data[0])
                idx_f = block_indices(rng, size, 1, size, min(block, size))[0]
                sampled_factor = tuple(x[idx_f] for x in model_data[:3]) + (model_data[3],)
            fitted = calibrate_scenarios(history[idx], sampled_factor, covariance_method, decay, method)
        else:
            fitted = baseline
        mu, cov, factors = fitted
        for offset in range(0, len(group), batch_limit):
            ids = group[offset:offset+batch_limit]
            size = (len(ids), steps)
            if method == "Historical Block Bootstrap":
                idx = block_indices(rng, n, *size, block)
                # Edge-weighted mean for each block position, including the truncated final block.
                starts = np.arange(n-block+1)
                means = history[starts[:, None]+np.arange(block)].mean(axis=0)
                shocks = history[idx] - means[np.arange(steps)%block] + mu
            elif method == "Factor Monte Carlo":
                factor = rng.multivariate_normal(factors["premiums"], factors["factor_cov"], size=size, check_valid="raise")
                residual = rng.multivariate_normal(np.zeros(assets), factors["residual_cov"], size=size, check_valid="raise")
                shocks = factors["rf"] + factor @ factors["beta"].T + residual
            else:
                shocks = rng.multivariate_normal(mu, cov, size=size, check_valid="raise")
            values[ids] = nav_paths(shocks, weights.to_numpy(), capital, rebalance, costs)
    return summarize(values, weights.columns, capital, sessions, rf), baseline


def summarize(values, labels, capital, sessions, rf):
    final = pd.DataFrame(values[:, -1]/capital-1, columns=labels)
    rows, fans = [], {}
    for j, label in enumerate(labels):
        nav = values[:, :, j]
        terminal = final[label].to_numpy()
        daily = nav[:, 1:]/nav[:, :-1]-1
        vol = daily.std(ddof=1)*np.sqrt(sessions)
        pnl = terminal*capital
        dd = (nav/np.maximum.accumulate(nav, axis=1)-1).min(axis=1)
        rows.append([terminal.mean(), np.median(terminal), vol,
                     (daily.mean()*sessions-rf)/vol if vol > 1e-12 else np.nan,
                     (terminal < 0).mean(), max(0., -np.quantile(pnl, .05)),
                     max(0., -empirical_tail_mean(pnl, .05)), np.median(dd),
                     *np.quantile(nav[:, -1], [.05, .95])])
        fans[label] = np.quantile(nav, [.05,.25,.5,.75,.95], axis=0)
    metrics = pd.DataFrame(rows, index=labels, columns=[
        "Mean Return", "Median Return", "Annualized Volatility", "Simulated Sharpe",
        "Probability of Loss", "Horizon VaR 95%", "Horizon ES 95%", "Median Max Drawdown",
        "P5 Final Value", "P95 Final Value",
    ])
    pair = pd.DataFrame([[(final[i]>final[j]).mean() for j in labels] for i in labels],
                        index=labels, columns=labels)
    best = final.eq(final.max(axis=1), axis=0)
    winners = best.div(best.sum(axis=1), axis=0).mean()
    return dict(metrics=metrics, final=final, fans=fans, pair=pair, best=winners,
                paths=values.astype(np.float32))
