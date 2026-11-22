"""Execute exposure variants from historical base targets and closing prices."""
import json

import numpy as np
import pandas as pd

from src.backtest import calculate_backtest_metrics
from src.backtest_registry import backtest_strategies
from src.dynamic_risk_backtest import exposure_signal, pretest_portfolio_returns
from src.transaction_costs import CostAssumptions, estimate_order_costs, fund_orders


def execute_overlay(curve, targets, prices, model, *, capital, costs=None,
                    train=252, horizon=21, update_every=5, target_vol=15.,
                    trade_band_pp=10., review_band_pp=5., reentry_band_pp=5.,
                    cash_annual_yield=0., window=63, decay=.94):
    dates = curve.index
    if len(dates) < 2 or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("At least two ordered, unique out-of-sample dates are required.")
    if targets is None or targets.empty:
        raise ValueError("Historical target weights are unavailable.")
    targets = targets.sort_index().loc[:dates[-1]].astype(float)
    if (targets.empty or not targets.index.is_unique or targets.index[0] != dates[0]
            or not targets.index.isin(dates).all() or not targets.columns.is_unique
            or not np.isfinite(targets).all().all() or (targets < -1e-10).any().any()
            or not np.allclose(targets.sum(axis=1), 1., atol=1e-7, rtol=0)):
        raise ValueError("Overlay requires dated, long-only, fully invested base targets.")
    targets = targets.clip(lower=0)
    if (not 0 < target_vol <= 100 or not 0 <= review_band_pp < trade_band_pp <= 100
            or not 0 < reentry_band_pp <= trade_band_pp or not 0 <= cash_annual_yield < 1
            or min(train, horizon, update_every) < 1 or not np.isfinite(capital) or capital <= 0):
        raise ValueError("Invalid overlay parameters.")
    if not targets.columns.isin(prices.columns).all():
        raise ValueError("Execution prices are missing for one or more target assets.")
    raw = prices.reindex(index=dates, columns=targets.columns)
    raw = raw.where(np.isfinite(raw) & raw.gt(0))
    marks = raw.ffill()
    pre = pretest_portfolio_returns(prices, targets, dates[0], train)
    base_returns = curve["Daily Return"].iloc[1:]
    if not np.isfinite(base_returns).all() or (base_returns <= -1).any():
        raise ValueError("Invalid base returns for volatility estimation.")
    history = pd.concat([np.log1p(pre), np.log1p(base_returns)])
    initial_signal_date = prices.index[prices.index < dates[0]].max()
    quantities = pd.Series(0., index=targets.columns)
    reference = quantities.copy()
    cash, exposure = float(capital), 1.
    cash_rate = (1 + cash_annual_yield) ** (1 / 252) - 1
    forecast = np.nan
    rows, orders, signals = [], [], []
    free = CostAssumptions(0., 0., 0., 0.)
    for i, date in enumerate(dates):
        signal_date = initial_signal_date if i == 0 else dates[i - 1]
        resize = False
        if i == 0 or (i - 1) % update_every == 0:
            try:
                forecast, proposed, action = exposure_signal(
                    history.loc[:signal_date], model, exposure, target_vol,
                    horizon, window, decay, trade_band_pp, review_band_pp, reentry_band_pp,
                )
                resize = action == "REBALANCE"
                if resize:
                    exposure = proposed
                signals.append((signal_date, date, forecast, action, None))
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                forecast = np.nan
                signals.append((signal_date, date, forecast, "FORECAST ERROR", str(exc)))
        if i:
            cash *= 1 + cash_rate
        price = marks.loc[date]
        pretrade = float((quantities * price).sum() + cash)
        rebalance = date in targets.index
        cost = 0.
        if rebalance or resize:
            if rebalance:
                base_weights = targets.loc[date]
            else:
                market_values = (reference * price).fillna(0.)
                base_weights = market_values / market_values.sum()
            target = exposure * base_weights
            executable = target.gt(0) | quantities.ne(0)
            if raw.loc[date, executable].isna().any():
                raise ValueError(f"Missing closing execution price on {date:%Y-%m-%d}.")
            if rebalance:
                reference = base_weights.div(price).fillna(0.)
            budget = pretrade
            if costs is not None:
                budget, cost, detail = fund_orders(pretrade, target, quantities, price, costs)
            else:
                _, detail = estimate_order_costs(budget * target.div(price).fillna(0.) - quantities, price, free)
            quantities = (budget * target).div(price).fillna(0.)
            cash = budget * (1 - target.sum())
            orders.append({"Rebalance Date": date, "Base Rebalance": rebalance,
                           "Exposure Change": resize, **detail, "Total Cost": cost,
                           "Executed Turnover": detail["Traded Notional"] / pretrade})
        invested = float((quantities * price).sum())
        nav = invested + cash
        rows.append((nav, invested, cash, exposure, invested / nav, forecast, cost))

    out = pd.DataFrame(rows, index=dates, columns=[
        "Portfolio Value", "Invested Value", "Cash", "Exposure Multiplier",
        "Market Exposure", "Forecast Volatility (%)", "Transaction Cost",
    ])
    out["Daily Return"] = out["Portfolio Value"].pct_change()
    out.loc[dates[0], "Daily Return"] = out["Portfolio Value"].iloc[0] / capital - 1
    out.attrs.update(initial_capital=capital, transaction_log=pd.DataFrame(orders),
                     exposure_log=pd.DataFrame(signals, columns=[
                         "Signal Date", "Execution Date", "Forecast Volatility (%)", "Action", "Error"]))
    return out


def overlay_metrics(curve, risk_free_rate, target_vol):
    metrics = calculate_backtest_metrics(curve, risk_free_rate)
    exposure = curve["Market Exposure"]
    log = curve.attrs["transaction_log"]
    losses = -curve["Daily Return"]
    var = losses.quantile(.95)
    drawdown = abs(metrics["Max Drawdown"])
    return {**metrics, "Annualized Return": curve["Daily Return"].mean() * 252,
            "Calmar Ratio": metrics["CAGR"] / drawdown if drawdown else np.nan,
            "Daily VaR 95": var, "Daily ES 95": losses[losses >= var].mean(),
            "Executed Turnover": log["Executed Turnover"].sum(),
            "Transaction Costs (€)": log["Total Cost"].sum(),
            "Average Exposure": exposure.mean(), "Minimum Exposure": exposure.min(),
            "Maximum Exposure": exposure.max(), "Time Below Full Exposure": (exposure < 1 - 1e-8).mean(),
            "Target Volatility": target_vol / 100,
            "Volatility minus Target": metrics["Volatility"] - target_vol / 100}


def build_overlay_variants(result, selected, models, risk_free_rate, *, prices=None, **settings):
    variants, errors = {}, {}
    if not selected or not models:
        return variants, errors
    strategies = backtest_strategies(result)
    if prices is None:
        prices = result.get("risk_prices")
    for name in dict.fromkeys(selected):
        if name not in strategies:
            errors[name] = result.get("ml_errors", {}).get(name, "Base backtest unavailable.")
            continue
        base = strategies[name]
        for model in dict.fromkeys(models):
            label = f"{name} + {model} Overlay"
            try:
                if prices is None:
                    raise ValueError("Historical execution prices are unavailable.")
                curve = execute_overlay(base["curve"], base["weights"], prices, model,
                                        capital=result["capital"], costs=result.get("transaction_cost_preset"),
                                        **settings)
                metrics = overlay_metrics(curve, risk_free_rate, settings.get("target_vol", 15.))
                baseline = calculate_backtest_metrics(base["curve"], risk_free_rate)
                delta = {key: metrics[key] - baseline[key]
                         for key in ("Total Return", "Volatility", "Sharpe Ratio", "Max Drawdown")}
                base_log = base["curve"].attrs.get("transaction_log")
                delta["Transaction Costs (€)"] = metrics["Transaction Costs (€)"] - (
                    base_log.loc[base_log["Rebalance Date"].isin(base["curve"].index), "Total Cost"].sum()
                    if base_log is not None else 0.)
                variant_id = json.dumps(["exposure", base["strategy_id"], model], separators=(",", ":"))
                variants[variant_id] = dict(label=label, base=name, model=model, curve=curve,
                                            metrics=metrics, delta=delta, strategy_id=variant_id)
            except (ValueError, ImportError, RuntimeError, np.linalg.LinAlgError) as exc:
                errors[label] = str(exc)
    return variants, errors
