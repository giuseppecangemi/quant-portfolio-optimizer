"""Rolling out-of-sample robustness, delegating portfolio logic to src.backtest.

No changes to the existing financial models. Price downloads preserve missing data.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf

from src.backtest import backtest_portfolio, calculate_backtest_metrics
from src.transaction_costs import CostAssumptions, apply_transaction_costs
from src.fama_french import load_europe_ff3
from src.fama_french_5 import load_europe_ff5

STRATEGIES = {
    "CAPM": "capm_max_sharpe",
    "FF3": "ff3_max_sharpe",
    "FF5": "ff5_max_sharpe",
    "Equal Weight": "equal_weight",
}


def _download_close(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download adjusted prices by absolute dates, preserving partial ticker histories."""
    tickers = list(dict.fromkeys(str(t).strip() for t in tickers if str(t).strip()))
    if not tickers:
        return pd.DataFrame()
    pieces = []
    # Small batches are more tolerant of failures in large market universes.
    for offset in range(0, len(tickers), 25):
        batch = tickers[offset:offset + 25]
        try:
            raw = yf.download(
                tickers=batch, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"), auto_adjust=True,
                progress=False, threads=False, group_by="column",
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" not in raw.columns.get_level_values(0):
                    continue
                close = raw["Close"]
            else:
                if "Close" not in raw:
                    continue
                close = raw[["Close"]].rename(columns={"Close": batch[0]})
            if isinstance(close, pd.Series):
                close = close.to_frame(name=batch[0])
            close.columns = [str(c) for c in close.columns]
            pieces.append(close)
        except Exception:
            continue
    if not pieces:
        return pd.DataFrame()
    frame = pd.concat(pieces, axis=1).sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")


def run_rolling_robustness(
    tickers: list[str], benchmark_ticker: str, start_year: int = 2010,
    end_year: int = 2025, estimation_years: int = 5,
    horizon_months: int = 12, step_months: int = 12,
    top_n: int | None = 20, selection_policy: str = "fixed",
    rebalance_days: int = 63, capital: float = 10000.0,
    risk_free_rate: float = 0.02, max_weight: float = 0.20,
    progress=None, transaction_costs: CostAssumptions | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (metrics, skipped_windows, download_info).

    The window starts on the first trading date on/after each calendar anchor.
    Estimation is the preceding calendar years; OOS ends at the horizon boundary.
    Every model is re-estimated from historical data by the existing backtest engine.
    """
    if not 2010 <= start_year <= end_year:
        raise ValueError("Invalid historical range (start year must be >= 2010).")
    if estimation_years < 1 or horizon_months < 1 or step_months < 1:
        raise ValueError("Estimation, horizon and step must be positive.")
    if selection_policy not in {"fixed", "dynamic"}:
        raise ValueError("Unknown selection policy.")
    if top_n is not None and max_weight * top_n < 1 - 1e-10:
        raise ValueError("Max asset weight is incompatible with the requested Top N.")

    first_anchor = pd.Timestamp(year=start_year, month=1, day=1)
    last_boundary = pd.Timestamp(year=end_year + 1, month=1, day=1)
    download_start = first_anchor - pd.DateOffset(years=estimation_years, days=12)
    download_end = last_boundary + pd.Timedelta(days=7)
    prices = _download_close(tickers, download_start, download_end)
    benchmark = _download_close([benchmark_ticker], download_start, download_end)
    fx = _download_close(["EURUSD=X"], download_start, download_end)
    if prices.empty:
        raise ValueError("No historical asset prices were downloaded.")
    if benchmark.empty:
        raise ValueError("Benchmark historical prices are unavailable.")

    ff3, ff5 = None, None
    factor_errors = {}
    try:
        ff3 = load_europe_ff3()
    except Exception as exc:
        factor_errors["FF3"] = str(exc)
    try:
        ff5 = load_europe_ff5()
    except Exception as exc:
        factor_errors["FF5"] = str(exc)

    records, skipped = [], []
    anchors = []
    anchor = first_anchor
    while anchor + pd.DateOffset(months=horizon_months) <= last_boundary:
        anchors.append(anchor)
        anchor += pd.DateOffset(months=step_months)
    if not anchors:
        raise ValueError("No complete OOS windows fit in the chosen period.")

    for window_i, anchor in enumerate(anchors):
        horizon_end = anchor + pd.DateOffset(months=horizon_months)
        estimation_start = anchor - pd.DateOffset(years=estimation_years)
        # Calendar window, no leakage from later periods.
        window = prices.loc[(prices.index >= estimation_start) & (prices.index < horizon_end)].copy()
        before = window.index[window.index < anchor]
        after = window.index[(window.index >= anchor) & (window.index < horizon_end)]
        # Do not silently shorten the estimation period because the provider lacks history.
        expected_first = estimation_start + pd.Timedelta(days=14)
        if before.empty or before[0] > expected_first or len(before) < estimation_years * 180:
            skipped.append({"OOS Start": anchor.date().isoformat(), "Reason": "Insufficient estimation history"})
            continue
        if len(after) < max(20, int(horizon_months * 15)):
            skipped.append({"OOS Start": anchor.date().isoformat(), "Reason": "Incomplete OOS prices"})
            continue
        # The engine expects lookback observations immediately preceding the first OOS day.
        lookback = len(before)
        # Extra terminal date is needed for the engine's loop boundary.
        next_trading = prices.index[prices.index >= horizon_end]
        if len(next_trading) == 0:
            skipped.append({"OOS Start": anchor.date().isoformat(), "Reason": "Missing post-horizon boundary"})
            continue
        window = prices.loc[(prices.index >= estimation_start) & (prices.index <= next_trading[0])].copy()
        if window.shape[1] < 2:
            skipped.append({"OOS Start": anchor.date().isoformat(), "Reason": "Fewer than two assets"})
            continue
        # Independent buy-and-hold benchmark for this OOS window.
        # Uses the first and last dates available in the asset trading calendar.
        benchmark_window = benchmark.iloc[:, 0].reindex(after).ffill().dropna()
        if len(benchmark_window) >= 2 and benchmark_window.iloc[0] > 0:
            benchmark_curve = pd.DataFrame({
                "Portfolio Value": capital * benchmark_window / benchmark_window.iloc[0],
            })
            benchmark_curve["Daily Return"] = benchmark_curve["Portfolio Value"].pct_change().fillna(0.0)
            benchmark_metrics = calculate_backtest_metrics(benchmark_curve, risk_free_rate)
            benchmark_total_return = float(benchmark_metrics["Total Return"])
            records.append({
                "OOS Start": anchor.date().isoformat(),
                "OOS End": (horizon_end - pd.Timedelta(days=1)).date().isoformat(),
                "Model": "FTSE MIB",
                "Total Return": benchmark_total_return,
                "Sharpe Ratio": float(benchmark_metrics["Sharpe Ratio"]),
                "Max Drawdown": float(benchmark_metrics["Max Drawdown"]),
                "Benchmark Return": benchmark_total_return,
                "Excess Return": 0.0,
                "Beat Benchmark": False,
                "Initial Selected Assets": np.nan,
                "Requested Estimation Days": len(before),
                "Effective Common Estimation Days": np.nan,
                "Estimation Coverage": np.nan,
                "Effective Estimation Start": None,
                "Estimation Coverage Warning": False,
                "Net Total Return": benchmark_total_return,
                "Net Sharpe Ratio": float(benchmark_metrics["Sharpe Ratio"]),
                "Net Max Drawdown": float(benchmark_metrics["Max Drawdown"]),
                "Transaction Costs EUR": 0.0,
                "Traded Notional EUR": 0.0,
                "Orders": 0,
                "Rebalances": 0,
            })
        else:
            skipped.append({"OOS Start": anchor.date().isoformat(), "Model": "FTSE MIB", "Reason": "Insufficient benchmark OOS history"})

        for label, strategy in STRATEGIES.items():
            if label in factor_errors:
                skipped.append({"OOS Start": anchor.date().isoformat(), "Model": label, "Reason": factor_errors[label]})
                continue
            try:
                curve, weights = backtest_portfolio(
                    window, strategy=strategy, initial_capital=capital,
                    lookback_days=lookback, rebalance_days=rebalance_days,
                    risk_free_rate=risk_free_rate, max_weight=max_weight,
                    market_prices=benchmark, selection_top_n=top_n,
                    selection_policy=selection_policy, shared_window_cache={},
                    ff3_factors=ff3, ff5_factors=ff5, eurusd_prices=fx,
                )
                curve = curve.loc[(curve.index >= anchor) & (curve.index < horizon_end)]
                if len(curve) < 2:
                    raise ValueError("No OOS portfolio curve")
                # Metric function works on the original engine's portfolio curve.
                metrics = calculate_backtest_metrics(curve, risk_free_rate)
                b = benchmark.iloc[:, 0].loc[(benchmark.index >= curve.index[0]) & (benchmark.index <= curve.index[-1])].dropna()
                if len(b) < 2:
                    raise ValueError("Missing benchmark in the OOS window")
                benchmark_return = float(b.iloc[-1] / b.iloc[0] - 1)

                net_metrics = None
                trade_log = pd.DataFrame()
                if transaction_costs is not None:
                    net_metrics, trade_log, _ = apply_transaction_costs(
                        curve, weights.loc[(weights.index >= curve.index[0]) & (weights.index <= curve.index[-1])],
                        window, capital, transaction_costs, risk_free_rate,
                    )

                # Diagnostics only: reproduce the first rebalance's complete-case
                # history without changing selection, optimization or returns.
                first_weights = weights.iloc[0] if not weights.empty else pd.Series(dtype=float)
                initial_assets = [
                    ticker for ticker, weight in first_weights.items()
                    if pd.notna(weight) and float(weight) > 1e-10
                ]
                estimation_prices = window.loc[window.index < anchor, initial_assets]
                common_history = estimation_prices.dropna()
                common_days = len(common_history)
                requested_days = len(before)
                coverage = common_days / requested_days if requested_days else 0.0
                earliest_common = (
                    common_history.index[0].date().isoformat()
                    if common_days else None
                )
                # Informative flag, not an exclusion criterion: preserve every
                # existing backtest result and financial calculation.
                coverage_warning = coverage < 0.80
                records.append({
                    "OOS Start": anchor.date().isoformat(),
                    "OOS End": (horizon_end - pd.Timedelta(days=1)).date().isoformat(),
                    "Model": label,
                    "Total Return": float(metrics["Total Return"]),
                    "Sharpe Ratio": float(metrics["Sharpe Ratio"]),
                    "Max Drawdown": float(metrics["Max Drawdown"]),
                    "Benchmark Return": benchmark_return,
                    "Excess Return": float(metrics["Total Return"]) - benchmark_return,
                    "Beat Benchmark": float(metrics["Total Return"]) > benchmark_return,
                    "Initial Selected Assets": len(initial_assets),
                    "Requested Estimation Days": requested_days,
                    "Effective Common Estimation Days": common_days,
                    "Estimation Coverage": round(coverage, 4),
                    "Effective Estimation Start": earliest_common,
                    "Estimation Coverage Warning": coverage_warning,
                    "Net Total Return": float(net_metrics["Total Return"]) if net_metrics else np.nan,
                    "Net Sharpe Ratio": float(net_metrics["Sharpe Ratio"]) if net_metrics else np.nan,
                    "Net Max Drawdown": float(net_metrics["Max Drawdown"]) if net_metrics else np.nan,
                    "Transaction Costs EUR": float(trade_log["Total Cost"].sum()) if not trade_log.empty else 0.0,
                    "Traded Notional EUR": float(trade_log["Traded Notional"].sum()) if not trade_log.empty else 0.0,
                    "Orders": int(trade_log["Orders"].sum()) if not trade_log.empty else 0,
                    "Rebalances": len(trade_log),
                })
            except Exception as exc:
                skipped.append({"OOS Start": anchor.date().isoformat(), "Model": label, "Reason": str(exc)[:260]})
        if progress is not None:
            progress((window_i + 1) / len(anchors))
    info = {
        "requested_start": download_start.date().isoformat(),
        "first_price": prices.index.min().date().isoformat(),
        "last_price": prices.index.max().date().isoformat(),
        "asset_columns": len(prices.columns),
        "candidate_windows": len(anchors),
        "factor_errors": factor_errors,
    }
    return pd.DataFrame(records), pd.DataFrame(skipped), info
