"""Risk-exposure overlay on an EXISTING strategy net backtest curve.

This is an explicitly labelled approximation, not an execution-level replay.
The underlying strategy series already reflects its own transaction costs when enabled.
Only ADDITIONAL exposure changes are charged here.
"""
import warnings
import numpy as np
import pandas as pd
from src.volatility_forecasting import _predict_variance, ARCH_AVAILABLE

MODELS = ("EWMA", "GARCH(1,1)", "GJR-GARCH", "EGARCH")


def run_exposure_overlay(curve, models=MODELS, train=252, horizon=21, pretest_returns=None,
                         target_vol=15.0, trade_band_pp=10.0, review_band_pp=5.0,
                         window=63, decay=0.94, update_every=5, capital=10000.,
                         commission_rate=0.0005, min_commission=3.0,
                         half_spread_bps=5.0, slippage_bps=5.0,
                         estimated_holdings=20, cash_annual_yield=0.0, reentry_band_pp=None):
    """Generate strictly lagged risk decisions using net strategy returns.

    Signals at close t are applied starting at t+1. Forecast is fitted to data
    available up to t. The benchmark base curve is never altered. Since only
    aggregate portfolio returns are available, this overlay approximates cash
    transitions and proportional trading costs; it cannot reconstruct actual
    constituent-level orders or intraday fills.
    """
    if isinstance(curve, pd.DataFrame):
        values = curve["Portfolio Value"].astype(float)
    else:
        values = pd.Series(curve, dtype=float)
    values = values.sort_index().dropna()
    if len(values) < 25 or (values <= 0).any():
        raise ValueError(f"Invalid/short OOS equity curve: {len(values)} observations; at least 25 positive values required")
    if pretest_returns is None:
        raise ValueError("Pre-test portfolio returns are required. Rerun the standard backtest with the updated app.py.")
    pretest_returns = pd.Series(pretest_returns, dtype=float).sort_index().replace([np.inf, -np.inf], np.nan).dropna()
    pretest_returns = pretest_returns.loc[pretest_returns.index < values.index[0]]
    if len(pretest_returns) < train:
        raise ValueError(f"Pre-test volatility history: {len(pretest_returns)} daily returns, need {train}. "
                         "Increase the historical dataset or reduce the initial volatility estimation window.")
    if (pretest_returns <= -1).any():
        raise ValueError("Invalid pre-test portfolio returns (<= -100%).")
    if reentry_band_pp is None:
        reentry_band_pp = trade_band_pp
    if not (0 < target_vol <= 100 and 0 <= review_band_pp < trade_band_pp <= 100
            and 0 < reentry_band_pp <= trade_band_pp):
        raise ValueError("Invalid target or exposure bands")
    if not (0 <= cash_annual_yield < 1 and 0 <= commission_rate < 1):
        raise ValueError("Invalid cash yield or commission")
    if update_every < 1 or horizon < 1 or train < 252:
        raise ValueError("Invalid training / forecast settings")
    models = tuple(dict.fromkeys(models))
    if any(m not in MODELS for m in models):
        raise ValueError("Unknown volatility model")
    if not ARCH_AVAILABLE and any(m != "EWMA" for m in models):
        raise ImportError("arch is required for GARCH risk overlays")

    daily = values.pct_change().fillna(0.0)
    # Pre-test proxy is estimated using the FIRST point-in-time portfolio weights.
    # No out-of-sample sessions are discarded for volatility warm-up.
    logret = pd.concat([np.log1p(pretest_returns.tail(train)), np.log1p(daily.iloc[1:])])
    test_index = values.index
    base = values / values.iloc[0] * capital
    cash_daily = (1 + cash_annual_yield) ** (1/252) - 1
    curves = pd.DataFrame({"Standard": base}, index=test_index)
    logs = []
    summary = []
    for model in models:
        nav = float(capital)
        exposure = 1.0
        nav_history = []
        exposure_history = []
        forecast_history = []
        costs_total = 0.0
        turnover = 0.0
        trades = 0
        latest_forecast = np.nan
        latest_target = 1.0
        # Initial allocation decision uses only pre-test observations.
        initial_past = np.log1p(pretest_returns.tail(train))
        try:
            initial_var = _predict_variance(initial_past, horizon, model, window, decay)
            initial_forecast = float(np.sqrt(initial_var * 252) * 100)
            initial_target = min(1.0, target_vol / initial_forecast)
            initial_gap = abs(initial_target - exposure) * 100
            if initial_gap >= trade_band_pp:
                exposure = initial_target
                # Approximate opening exposure transaction cost.
                delta = abs(initial_target - 1.0)
                notional = delta * nav
                cost = min(nav * 0.99, max(commission_rate * notional, max(1, int(estimated_holdings)) * min_commission)
                           + notional * (half_spread_bps + slippage_bps) / 10000)
                nav -= cost
                costs_total += cost
                turnover += delta
                trades += 1
                initial_action = "REBALANCE"
            else:
                cost = 0.0
                initial_action = "REVIEW" if initial_gap >= review_band_pp else "HOLD"
            latest_forecast = initial_forecast
            latest_target = initial_target
            logs.append({"Model": model, "Date": pretest_returns.index[-1], "Action": initial_action,
                         "Forecast volatility (%)": initial_forecast, "Target exposure (%)": initial_target * 100,
                         "Exposure after decision (%)": exposure * 100, "Exposure gap (pp)": initial_gap,
                         "Execution threshold (pp)": trade_band_pp, "Incremental cost (€)": cost})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            logs.append({"Model": model, "Date": pretest_returns.index[-1],
                         "Action": "FORECAST ERROR", "Error": str(exc)})
        for j, date in enumerate(test_index):
            # Daily holdings exposure set at previous close; no same-day lookahead.
            r = float(daily.loc[date])
            if j > 0:
                nav *= 1.0 + exposure * r + (1.0 - exposure) * cash_daily
            # End-of-day decision for the FOLLOWING session only.
            if j % update_every == 0 and j < len(test_index) - 1:
                past = logret.loc[logret.index <= date]
                try:
                    var = _predict_variance(past, horizon, model, window, decay)
                    latest_forecast = float(np.sqrt(var * 252) * 100)
                    latest_target = min(1.0, target_vol / latest_forecast)
                    gap_pp = abs(latest_target - exposure) * 100
                    # Asymmetric hysteresis: a smaller gap is enough to restore exposure.
                    threshold = reentry_band_pp if latest_target > exposure else trade_band_pp
                    action = "REBALANCE" if gap_pp >= threshold else ("REVIEW" if gap_pp >= review_band_pp else "HOLD")
                    cost = 0.0
                    if action == "REBALANCE":
                        delta = abs(latest_target - exposure)
                        notional = delta * nav
                        # Approximate per-holding commissions for proportional scaling.
                        n_orders = max(1, int(estimated_holdings))
                        commission = max(commission_rate * notional, n_orders * min_commission)
                        variable = notional * (half_spread_bps + slippage_bps) / 10000
                        cost = min(nav * 0.99, commission + variable)
                        nav -= cost
                        exposure = latest_target
                        turnover += delta
                        trades += 1
                        costs_total += cost
                    logs.append({"Model": model, "Date": date, "Forecast volatility (%)": latest_forecast,
                                 "Target exposure (%)": latest_target * 100,
                                 "Exposure after decision (%)": exposure * 100,
                                 "Exposure gap (pp)": gap_pp, "Execution threshold (pp)": threshold, "Action": action,
                                 "Incremental cost (€)": cost})
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    logs.append({"Model": model, "Date": date, "Action": "FORECAST ERROR",
                                 "Error": str(exc)})
            nav_history.append(nav)
            exposure_history.append(exposure)
            forecast_history.append(latest_forecast)
        out = pd.Series(nav_history, index=test_index, name=model)
        curves[model] = out
        ret = out.pct_change().dropna()
        annual_return = float((out.iloc[-1] / capital) ** (252 / len(out)) - 1)
        annual_vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else np.nan
        sharpe = (annual_return - cash_annual_yield) / annual_vol if annual_vol > 0 else np.nan
        drawdown = float((out.to_numpy() / np.maximum.accumulate(np.r_[capital, out.to_numpy()])[1:] - 1).min())
        summary.append({"Strategy": model, "CAGR": annual_return, "Sharpe (approx.)": sharpe,
                        "Max Drawdown": drawdown, "Annual volatility": annual_vol,
                        "Total return": float(out.iloc[-1]/capital - 1),
                        "Avg invested (%)": float(np.mean(exposure_history)*100),
                        "Trades": trades, "Turnover (one-way)": turnover,
                        "Incremental costs (€)": costs_total})
    stdret = base.pct_change().dropna()
    stdcagr = float((base.iloc[-1] / capital)**(252/len(base))-1)
    stdvol = float(stdret.std()*np.sqrt(252))
    stdd = float((base.to_numpy()/np.maximum.accumulate(np.r_[capital,base.to_numpy()])[1:]-1).min())
    summary.insert(0,{"Strategy":"Standard", "CAGR":stdcagr,
                      "Sharpe (approx.)":(stdcagr-cash_annual_yield)/stdvol if stdvol>0 else np.nan,
                      "Max Drawdown":stdd,"Annual volatility":stdvol,
                      "Total return":float(base.iloc[-1]/capital-1),
                      "Avg invested (%)":100.,"Trades":0,"Turnover (one-way)":0.,"Incremental costs (€)":0.})
    return pd.DataFrame(summary).set_index("Strategy"), curves, pd.DataFrame(logs)
