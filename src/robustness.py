"""Analisi rolling della robustezza fuori campione (out-of-sample), 
delegando la logica di costruzione e gestione del portafoglio al modulo src.backtest.
Nessuna modifica ai modelli finanziari esistenti. Il download dei prezzi mantiene i dati mancanti 
senza effettuare imputazioni o riempimenti automatici.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from src.eur_risk_free import EONIA_ESTR, EONIA_KEY, ESTR_KEY, ESTR_ONLY, validate_mode
from src.data import download_adjusted_close_range as _download_close

from src.backtest import backtest_portfolio, calculate_backtest_metrics
from src.backtest_inputs import MarketInputs, benchmark_curve, prepare_market_inputs, prepare_walk_forward_window
from src.transaction_costs import CostAssumptions
from src.benchmarks import ETFS, LEGACY_BENCHMARK, daily_relative_metrics, endpoint_comparison, etf_coverage, index_coverage, load_etf_data, matched_portfolio_metrics
from src.fama_french import load_europe_ff3, load_us_ff3
from src.fama_french_5 import load_europe_ff5, load_us_ff5

STRATEGIES = {
    "Maximum Sharpe": "max_sharpe",
    "CAPM": "capm_max_sharpe",
    "FF3": "ff3_max_sharpe",
    "FF5": "ff5_max_sharpe",
    "FF3 Factor Cov": "ff3_factor_cov_max_sharpe",
    "FF5 Factor Cov": "ff5_factor_cov_max_sharpe",
    "Minimum Volatility": "min_volatility",
    "Risk Parity": "risk_parity",
    "HRP": "hrp",
    "Equal Weight": "equal_weight",
}


def run_rolling_robustness(
    tickers: list[str], benchmark_ticker: str, start_year: int = 2010,
    end_year: int = 2025, estimation_years: int = 5,
    horizon_months: int = 12, step_months: int = 12,
    top_n: int | None = 20, selection_policy: str = "fixed",
    rebalance_days: int = 63, capital: float = 10000.0,
    risk_free_rate: float = 0.02, max_weight: float = 0.20,
    progress=None, transaction_costs: CostAssumptions | None = None,
    *, market: str = "Borsa Italiana", models: list[str] | None = None,
    market_inputs: MarketInputs | None = None, active_factors=("Momentum", "Risk"),
    benchmarks=None, etf_inputs=None, eur_rf_mode=ESTR_ONLY,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (metrics, skipped_windows, download_info).

    The window starts on the first trading date on/after each calendar anchor.
    Estimation is the preceding calendar years; OOS ends at the horizon boundary.
    Every model is re-estimated from historical data by the existing backtest engine.
    """
    validate_mode(eur_rf_mode)
    if not 2010 <= start_year <= end_year:
        raise ValueError("Invalid historical range (start year must be >= 2010).")
    if estimation_years < 1 or horizon_months < 1 or step_months < 1:
        raise ValueError("Estimation, horizon and step must be positive.")
    if selection_policy not in {"fixed", "dynamic"}:
        raise ValueError("Unknown selection policy.")
    if top_n is not None and max_weight * top_n < 1 - 1e-10:
        raise ValueError("Max asset weight is incompatible with the requested Top N.")

    models = list(STRATEGIES) if models is None else list(dict.fromkeys(models))
    if not models or any(model not in STRATEGIES for model in models):
        raise ValueError("Select at least one valid portfolio model.")
    if market not in {"Borsa Italiana", "USA"}:
        raise ValueError("Unsupported market.")

    first_anchor = pd.Timestamp(year=start_year, month=1, day=1)
    last_boundary = pd.Timestamp(year=end_year + 1, month=1, day=1)
    download_start = first_anchor - pd.DateOffset(years=estimation_years, days=12)
    factor_models = [model for model in models if model.startswith(("FF3", "FF5"))]
    families = tuple(dict.fromkeys(model[:3] for model in factor_models))
    if market_inputs is not None and market_inputs.market != market:
        raise ValueError("Prepared inputs must use the requested market.")
    if market_inputs is None:
        market_inputs, errors = prepare_market_inputs(
            lambda: _download_close(tickers, download_start, last_boundary),
            lambda: _download_close([benchmark_ticker], download_start, last_boundary),
            market=market, families=families, require_benchmark=False,
            fetch_fx=lambda: _download_close(["EURUSD=X"], download_start, last_boundary),
            factor_loaders={
                "FF3": load_us_ff3 if market == "USA" else load_europe_ff3,
                "FF5": load_us_ff5 if market == "USA" else load_europe_ff5,
            },
        )
    else:
        market_inputs, errors = prepare_market_inputs(
            market_inputs.prices, market_inputs.benchmark, market=market, families=families,
            ff3=market_inputs.ff3, ff5=market_inputs.ff5, fx=market_inputs.fx,
            fundamentals=market_inputs.fundamentals,
            allow_download=False, require_benchmark=False,
        )
    prices, benchmark = market_inputs.prices, market_inputs.benchmark
    factor_errors = {model: errors[model[:3]] for model in factor_models if model[:3] in errors}

    selected_benchmarks = tuple(dict.fromkeys((LEGACY_BENCHMARK,) if benchmarks is None else benchmarks))
    if not selected_benchmarks or any(name not in {LEGACY_BENCHMARK, *ETFS} for name in selected_benchmarks):
        raise ValueError("Select at least one supported benchmark.")
    currency = "USD" if market == "USA" else "EUR"
    if etf_inputs is None:
        etf_inputs = load_etf_data(selected_benchmarks, first_anchor, last_boundary - pd.Timedelta(days=1), currency)
    etf_quotes, etf_fx, etf_errors = etf_inputs
    matched_metrics = {}
    portfolio_nav, benchmark_nav = {}, {}
    coverage_reports, etf_diagnostics, endpoints, evaluated_bounds = [], [], [], []
    benchmark_coverage = {name: (series.index[0].date().isoformat(), series.index[-1].date().isoformat())
                for name, series in etf_quotes.items() if not series.empty}
    if LEGACY_BENCHMARK in selected_benchmarks and not benchmark.empty:
        close = benchmark.iloc[:, 0].dropna()
        if not close.empty:
            benchmark_coverage[LEGACY_BENCHMARK] = (close.index[0].date().isoformat(), close.index[-1].date().isoformat())

    records, skipped, sample_coverage = [], [], {}
    failed = set()
    anchors = []
    anchor = first_anchor
    while anchor + pd.DateOffset(months=horizon_months) <= last_boundary:
        anchors.append(anchor)
        anchor += pd.DateOffset(months=step_months)
    if not anchors:
        raise ValueError("No complete OOS windows fit in the chosen period.")

    import exchange_calendars

    calendar = exchange_calendars.get_calendar(
        {"Borsa Italiana": "XMIL", "USA": "XNYS"}[market],
        start=first_anchor - pd.Timedelta(days=7), end=last_boundary + pd.Timedelta(days=7),
    )
    for window_i, anchor in enumerate(anchors):
        if progress is not None:
            progress(window_i / len(anchors))
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
        sessions = calendar.sessions_in_range(anchor, horizon_end - pd.Timedelta(days=1))
        if sessions.empty:
            skipped.append({"OOS Start": anchor.date().isoformat(), "Reason": "No scheduled OOS sessions"})
            continue
        valid_prices = (np.isfinite(window) & window.gt(0)).any(axis=1)
        observed = valid_prices.index[valid_prices]
        missing_sessions = sessions.difference(observed)
        if len(missing_sessions):
            if sessions[-1] in missing_sessions:
                reason = f"Missing asset prices on final scheduled OOS session {sessions[-1]:%Y-%m-%d}"
            else:
                reason = f"Missing asset prices on {len(missing_sessions)} scheduled OOS sessions, first {missing_sessions[0]:%Y-%m-%d}"
            skipped.append({"OOS Start": anchor.date().isoformat(), "Reason": reason})
            continue
        if window.shape[1] < 2:
            skipped.append({"OOS Start": anchor.date().isoformat(), "Reason": "Fewer than two assets"})
            continue
        prepared = prepare_walk_forward_window(
            window, lookback, selection_policy=selection_policy, models=models,
            rebalance_days=rebalance_days, transaction_costs=transaction_costs,
            risk_free_rate=risk_free_rate, max_weight=max_weight, top_n=top_n, capital=capital,
        )
        evaluated_bounds.append((str(prepared.oos_start.date()), str(prepared.oos_end.date())))
        benchmark_return = np.nan
        etf_runs, window_coverage = {}, {}
        for benchmark_name in selected_benchmarks:
            if benchmark_name == LEGACY_BENCHMARK:
                report = index_coverage(benchmark, prepared.oos_dates, market, ticker=benchmark_ticker,
                                        requested=(str(download_start.date()), str((last_boundary - pd.Timedelta(days=1)).date())))
                report.update(Window=str(anchor.date()), Benchmark=benchmark_name)
                window_coverage[benchmark_name] = report
                coverage_reports.append(report)
            try:
                if benchmark_name == LEGACY_BENCHMARK:
                    reference_curve = benchmark_curve(
                        benchmark, prepared.oos_start, prepared.oos_end, capital,
                        required_dates=prepared.oos_dates,
                    )
                    benchmark_metrics = calculate_backtest_metrics(reference_curve, risk_free_rate)
                    benchmark_return = benchmark_metrics["Total Return"]
                else:
                    run, report, detail = etf_coverage(
                        ETFS[benchmark_name], etf_quotes.get(benchmark_name), prepared.oos_dates,
                        capital, risk_free_rate, currency, etf_fx,
                        requested=(str(first_anchor.date()), str((last_boundary - pd.Timedelta(days=1)).date())),
                        error=etf_errors.get(benchmark_name),
                    )
                    window_coverage[benchmark_name] = report
                    coverage_reports.append({"Window": str(anchor.date()), "Benchmark": benchmark_name, **report})
                    etf_diagnostics.extend(detail)
                    if run is None:
                        raise ValueError(report["Reason"])
                    etf_runs[benchmark_name] = run
                    reference_curve, benchmark_metrics = run["curve"], run["metrics"]
            except ValueError as exc:
                if benchmark_name == LEGACY_BENCHMARK:
                    report["Reason"] = str(exc)
                skipped.append({"OOS Start": anchor.date().isoformat(), "Model": benchmark_name, "Reason": str(exc)})
                continue
            benchmark_nav.setdefault(str(anchor.date()), {})[benchmark_name] = reference_curve["Portfolio Value"].copy()
            benchmark_total_return = float(benchmark_metrics["Total Return"])
            records.append({
                "OOS Start": anchor.date().isoformat(),
                "OOS End": (horizon_end - pd.Timedelta(days=1)).date().isoformat(),
                "Model": benchmark_name,
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
                "Transaction Costs": 0.0,
                "Traded Notional": 0.0,
                "Orders": 0,
                "Rebalances": 0,
            })
            if benchmark_name in ETFS:
                records[-1].update({
                    "Coverage Status": run["status"],
                    "Execution Start": reference_curve.index[0].date().isoformat(),
                    "Execution End": reference_curve.index[-1].date().isoformat(),
                    "CAGR": benchmark_metrics["CAGR"],
                    "Volatility": benchmark_metrics["Volatility"],
                })

        window_cache = {}
        for label in models:
            strategy = STRATEGIES[label]
            if label in factor_errors:
                skipped.append({"OOS Start": anchor.date().isoformat(), "Model": label, "Reason": factor_errors[label]})
                continue
            try:
                if strategy == "capm_max_sharpe" and benchmark.empty:
                    raise ValueError("CAPM market-index observations are unavailable")
                curve, weights = backtest_portfolio(
                    prepared.prices, strategy=strategy, initial_capital=prepared.capital,
                    lookback_days=len(prepared.training_dates), rebalance_days=prepared.rebalance_days,
                    risk_free_rate=prepared.risk_free_rate, max_weight=prepared.max_weight,
                    market_prices=benchmark, selection_top_n=prepared.top_n,
                    selection_policy=prepared.selection_policy, shared_window_cache=window_cache,
                    market=market, transaction_costs=prepared.transaction_costs, capture_initial_ex_ante=True,
                    ff3_factors=market_inputs.ff3, ff5_factors=market_inputs.ff5, eurusd_prices=market_inputs.fx,
                    historical_fundamentals=market_inputs.fundamentals, active_factors=active_factors,
                    eur_rf_mode=eur_rf_mode,
                )
                net_metrics = calculate_backtest_metrics(curve, risk_free_rate)
                gross_nav = curve.get("Gross Portfolio Value", curve["Portfolio Value"])
                gross_curve = gross_nav.to_frame("Portfolio Value")
                gross_curve["Daily Return"] = gross_nav.pct_change().fillna(0.0)
                gross_curve.attrs["initial_capital"] = capital
                metrics = calculate_backtest_metrics(gross_curve, risk_free_rate)
                if not curve.index.equals(prepared.oos_dates):
                    raise ValueError("Strategy did not complete the requested OOS calendar")
                nav = pd.DataFrame({"Portfolio Value": curve["Portfolio Value"],
                                    "Gross Portfolio Value": gross_nav})
                nav.attrs["initial_capital"] = capital
                portfolio_nav.setdefault(str(anchor.date()), {})[label] = nav
                trade_log = curve.attrs.get("transaction_log", pd.DataFrame())
                ex_ante = curve.attrs["initial_ex_ante_kpis"]

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
                risk_history = curve.attrs.get("factor_risk_history", {})
                samples = [sample for sample in risk_history.values() if "observations" in sample]
                if samples:
                    sample_coverage.setdefault(str(anchor.date()), {})[label] = {
                        "initial_observations": samples[0]["observations"],
                        "min_observations": min(sample["observations"] for sample in samples),
                        "from": min(sample["sample_start"][:10] for sample in samples),
                        "through": max(sample["sample_end"][:10] for sample in samples),
                    }
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
                    "Execution Start": curve.index[0].date().isoformat(),
                    "Execution End": curve.index[-1].date().isoformat(),
                    "Estimation End": ex_ante["estimation_end"],
                    "Initial Expected Return": ex_ante["expected_return"],
                    "Initial Volatility": ex_ante["volatility"],
                    "Initial Sharpe": ex_ante["sharpe"],
                    "Requested Estimation Days": requested_days,
                    "Effective Common Estimation Days": common_days,
                    "Estimation Coverage": round(coverage, 4),
                    "Effective Estimation Start": earliest_common,
                    "Estimation Coverage Warning": coverage_warning,
                    "Net Total Return": float(net_metrics["Total Return"]),
                    "Net Sharpe Ratio": float(net_metrics["Sharpe Ratio"]),
                    "Net Max Drawdown": float(net_metrics["Max Drawdown"]),
                    "Transaction Costs": float(trade_log["Total Cost"].sum()) if not trade_log.empty else 0.0,
                    "Traded Notional": float(trade_log["Traded Notional"].sum()) if not trade_log.empty else 0.0,
                    "Orders": int(trade_log["Orders"].sum()) if not trade_log.empty else 0,
                    "Rebalances": max(0, len(weights) - 1),
                })
                for name, report in window_coverage.items():
                    for basis in ("Net", "Gross"):
                        value = endpoint_comparison(curve, report, gross=basis == "Gross")
                        if value is not None:
                            endpoints.append({"Window": str(anchor.date()), "Benchmark": name, "Model": label,
                                              "Basis": basis, "From": report["From"], "Through": report["Through"],
                                              "Scope": report["Endpoint Scope"], "Portfolio Return": value,
                                              "Benchmark Return": report["Benchmark Return"],
                                              "Excess Return": value - report["Benchmark Return"]})
                for name, run in etf_runs.items():
                    dates = run["curve"].index
                    try:
                        comparison = {
                            basis: {**matched_portfolio_metrics(curve, dates, risk_free_rate, gross=basis == "Gross"),
                                    **daily_relative_metrics(curve, run["curve"], dates, gross=basis == "Gross")}
                            for basis in ("Net", "Gross")
                        }
                    except ValueError as exc:
                        skipped.append({"OOS Start": anchor.date().isoformat(), "Model": name,
                                        "Reason": f"{label} matched metrics: {exc}"})
                    else:
                        matched_metrics.setdefault(name, {}).setdefault(anchor.date().isoformat(), {})[label] = comparison
            except Exception as exc:
                if not isinstance(exc, (ValueError, OSError)):
                    failed.add(str(anchor.date()))
                skipped.append({"OOS Start": anchor.date().isoformat(), "Model": label, "Reason": str(exc)[:260]})
    if progress is not None:
        progress(1.0)
    reported = {(row["Window"], row["Benchmark"]) for row in coverage_reports}
    window_errors = {row["OOS Start"]: row["Reason"] for row in skipped if "Model" not in row}
    for anchor in anchors:
        window_id = str(anchor.date())
        for name in selected_benchmarks:
            if (window_id, name) not in reported:
                coverage_reports.append({"Window": window_id, "Benchmark": name, "Status": "UNAVAILABLE",
                                         "OOS Start": window_id,
                                         "OOS End": str((anchor + pd.DateOffset(months=horizon_months, days=-1)).date()),
                                         "Benchmark Return": None,
                                         "Reason": window_errors.get(window_id, "Window not evaluated")})
    info = {
        "requested_start": download_start.date().isoformat(),
        "first_price": prices.index.min().date().isoformat(),
        "last_price": prices.index.max().date().isoformat(),
        "asset_columns": len(prices.columns),
        "candidate_windows": len(anchors),
        "factor_errors": factor_errors,
        "models": models,
        "market": market,
        "benchmark": benchmark_ticker,
        "currency": currency,
        "selected_benchmarks": list(selected_benchmarks),
        "benchmark_coverage": benchmark_coverage,
        "benchmark_comparisons": matched_metrics,
        "portfolio_nav": portfolio_nav, "benchmark_nav": benchmark_nav, "risk_free_rate": risk_free_rate,
        "provider_start": str(first_anchor.date()),
        "provider_end": str((last_boundary - pd.Timedelta(days=1)).date()),
        "oos_start": min((start for start, _ in evaluated_bounds), default=None),
        "oos_end": max((end for _, end in evaluated_bounds), default=None),
        "coverage_reports": coverage_reports, "benchmark_diagnostics": etf_diagnostics,
        "benchmark_endpoints": endpoints,
    }
    completed = {(row["OOS Start"], row["Model"]) for row in records}
    window_status = []
    for anchor in anchors:
        window_id = str(anchor.date())
        successful = [model for model in models if (window_id, model) in completed]
        reasons = [f"{row['Model']}: {row['Reason']}" if row.get("Model") else row["Reason"]
                   for row in skipped if row["OOS Start"] == window_id]
        sample = sample_coverage.get(window_id, {})
        window_status.append({
            "OOS Start": window_id, "Status": ("FAILED" if window_id in failed else
                       "PASS" if len(successful) == len(models) else "SKIPPED"),
            "Completed": ", ".join(successful),
            "Initial factor observations": "; ".join(
                f"{label}: {coverage['initial_observations']}" for label, coverage in sample.items()
            ),
            "Reason": " | ".join(reasons),
        })
    info["window_status"] = window_status
    info["factor_sample_coverage"] = sample_coverage
    info["eur_rf_mode"] = eur_rf_mode if market == "Borsa Italiana" else None
    if info["eur_rf_mode"] and factor_models:
        info["eur_rf_sources"] = [EONIA_KEY, ESTR_KEY] if eur_rf_mode == EONIA_ESTR else [ESTR_KEY]
    results = pd.DataFrame(records)
    if not results.empty:
        results["Currency"] = info["currency"]
    return results, pd.DataFrame(skipped), info
