"""Shared benchmark controls and Standard/Fixed Horizon result presentation."""
import pandas as pd
import streamlit as st

from src.backtest_registry import backtest_strategies
from src.benchmarks import (
    ETFS, LEGACY_BENCHMARK, attach_etf_benchmarks, benchmark_label, benchmark_run_label,
    daily_relative_metrics, download_adjusted, endpoint_comparison, load_etf_data, matched_portfolio_metrics,
)


def select_benchmarks(market, key, *, default=LEGACY_BENCHMARK):
    return st.multiselect(
        "Benchmarks", [LEGACY_BENCHMARK, *ETFS], default=[default],
        format_func=lambda name: (
            "IUSQ.DE — iShares MSCI ACWI UCITS ETF"
            if default == name == "iShares MSCI ACWI ETF" else benchmark_label(name, market)
        ), key=key,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_adjusted(symbol, currency, start, end, *, listing=None):
    return download_adjusted(symbol, currency, start, end, listing=listing)


def prepare_result_benchmarks(result, selected, market, risk_free_rate):
    dates = result["max_sharpe_bt"].index
    currency = result.get("portfolio_currency", "USD" if market == "USA" else "EUR")
    if "run_provenance" in result:
        start, end = str(dates[0].date()), str(dates[-1].date())
        result["run_provenance"].update(oos_start=start, oos_end=end,
                                        provider_start=start, provider_end=end)
    quotes, fx, errors = load_etf_data(selected, dates[0], dates[-1], currency, fetch=cached_adjusted)
    attach_etf_benchmarks(result, selected, market, risk_free_rate, quotes, fx, errors)


def benchmark_details(name):
    if name in ETFS:
        etf = ETFS[name]
        return f"{etf.ticker} · {etf.isin} · {etf.venue} · {etf.quote_currency} · {etf.income}"
    return "Price index · dividends excluded"


def render_benchmark_details(result, rate, selected_portfolios, *, observed_only=False):
    market = result.get("benchmark_market", "Borsa Italiana")
    selected = result.get("selected_benchmarks", [LEGACY_BENCHMARK])
    runs = {name: run for name, run in result.get("etf_benchmarks", {}).items()
            if not observed_only or run.get("status") != "COMPLETE_WITH_IMPUTATION"}
    if LEGACY_BENCHMARK in selected:
        runs[LEGACY_BENCHMARK] = {
            "curve": result["ftse_mib_bt"], "metrics": result["ftse_bt_metrics"],
            "currency": "USD" if market == "USA" else "EUR",
        }
    for name, error in result.get("benchmark_failures", {}).items():
        st.warning(f"{name} · {error}")
    rows = []
    for name in selected:
        if name not in ETFS or name in runs:
            continue
        first, last = result.get("benchmark_quote_coverage", {}).get(name, (None, None))
        rows.append({"Benchmark": name, "Instrument": benchmark_details(name),
                     "Return Currency": "USD" if market == "USA" else "EUR",
                     "Available From": first, "Available Through": last,
                     "OOS Start": None, "OOS End": None})
    if not runs and not rows:
        return
    deltas = []
    portfolios = {name: run["curve"] for name, run in backtest_strategies(result).items()
                  if name != LEGACY_BENCHMARK and (selected_portfolios is None or name in selected_portfolios)}
    portfolios.update({run["label"]: run["curve"] for run in result.get("overlay_runs", {}).values()})
    endpoints = []
    for report in result.get("coverage_reports", []):
        for portfolio, curve in portfolios.items():
            value = endpoint_comparison(curve, report)
            if value is not None:
                endpoints.append({"Portfolio": portfolio, "Benchmark": report["Benchmark"],
                                  "From": report["From"], "Through": report["Through"], "Scope": report["Endpoint Scope"],
                                  "Portfolio Return": value, "Benchmark Return": report["Benchmark Return"],
                                  "Excess Return": value - report["Benchmark Return"]})
    render_coverage(result.get("coverage_reports", []), result.get("benchmark_diagnostics", []),
                    endpoints, key=result.get("run_provenance", {}).get("run_id", "legacy") + "_"
                    + str(result.get("is_fixed_horizon", False)) + str(result.get("selection_policy", "")),
                    provenance=result.get("run_provenance"))
    for name, run in runs.items():
        dates = run["curve"].index
        label = benchmark_run_label(benchmark_label(name, market), run)
        rows.append({
            "Benchmark": label, "Instrument": benchmark_details(name), "Return Currency": run["currency"],
            "Available From": run.get("history_start", dates[0].date().isoformat()),
            "Available Through": run.get("history_end", dates[-1].date().isoformat()),
            "OOS Start": dates[0].date().isoformat(), "OOS End": dates[-1].date().isoformat(),
        })
        for portfolio, curve in portfolios.items():
            common = dates.intersection(curve.index)
            if len(common) < 2:
                continue
            model_return = matched_portfolio_metrics(curve, common, rate)["Total Return"]
            benchmark_return = matched_portfolio_metrics(run["curve"], common, rate)["Total Return"]
            deltas.append({"Portfolio": portfolio, "Benchmark": label, "Sessions": len(common),
                           **daily_relative_metrics(curve, run["curve"], common),
                           "From": common[0].date().isoformat(), "Through": common[-1].date().isoformat(),
                           "Excess Return": model_return - benchmark_return})
    with st.expander("Benchmark coverage and matched excess returns"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if deltas:
            st.dataframe(pd.DataFrame(deltas).style.format({"Excess Return": "{:+.2%}", "Tracking Error": "{:.2%}", "Information Ratio": "{:.2f}"}, na_rep="—"),
                         hide_index=True, width="stretch")
        st.caption("Excess returns use common observed sessions. ETF adjusted prices reflect fund expenses "
                   "and distributions. Benchmark returns exclude investor commissions, spread, slippage "
                   "and FX charges.")


def render_run_provenance(result, snapshot, attempt):
    from src.run_provenance import run_status

    if attempt and attempt.get("error"):
        st.error(f"FAILED · {attempt['section']} · {attempt['run_id']} · {attempt['timestamp']} · {attempt['error']}")
        with st.expander("Failed attempt configuration"):
            st.code(attempt["configuration"], language="json")
            st.write({key: attempt.get(key) for key in ("oos_start", "oos_end", "provider_start", "provider_end")})
    if result is None:
        return
    status = run_status(result, snapshot, attempt)
    if status == "STALE":
        st.warning("STALE · This result belongs to a previous configuration or attempt.")
    st.caption(f"{status} · {result['section']} · {result['run_id']} · {result['timestamp']}")
    with st.expander("Run configuration and date ranges"):
        st.code(result["configuration"], language="json")
        st.write({key: result.get(key) for key in ("oos_start", "oos_end", "provider_start", "provider_end")})


def render_coverage(reports, diagnostics, endpoints, key, provenance=None):
    imputed = [row for row in reports if row.get("Imputed Count", 0)]
    if imputed:
        count = len({(row["Ticker"], day) for row in imputed
                     for day in row["Imputed Dates"].split(", ")})
        st.warning(f"Benchmark ETF — {count} prezzi interpolati. "
                   "Le metriche giornaliere calcolate con stime sono contrassegnate. "
                   "Il rendimento agli estremi usa prezzi osservati.")
    if reports:
        with st.expander("Benchmark coverage by window"):
            frame = pd.DataFrame(reports)
            st.write(frame.groupby(["Benchmark", "Status"], sort=False).size().rename("Windows").reset_index())
            st.dataframe(frame.style.format({"Benchmark Return": "{:.2%}"}, na_rep="—"),
                         hide_index=True, width="stretch")
    if diagnostics:
        with st.expander("Daily benchmark observations"):
            data = pd.DataFrame(diagnostics)
            if provenance:
                data = data.assign(Run_ID=provenance["run_id"], Section=provenance["section"])
            st.dataframe(data.loc[data.Type.ne("VALID_OBSERVATION")], hide_index=True, width="stretch")
            st.download_button("Download benchmark diagnostics", data.to_csv(index=False),
                               f"{key}_benchmark_diagnostics.csv", "text/csv", key=f"{key}_coverage_csv")
    if endpoints:
        with st.expander("Endpoint cumulative return"):
            st.caption("Return between the two observed dates. PARTIAL denotes a shorter period. "
                       "Daily risk metrics are reported separately.")
            data = pd.DataFrame(endpoints)
            st.dataframe(data.style.format({column: "{:+.2%}" for column in
                                           ("Portfolio Return", "Benchmark Return", "Excess Return")}),
                         hide_index=True, width="stretch")
