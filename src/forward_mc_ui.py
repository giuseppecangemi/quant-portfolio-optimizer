"""Forward scenario configuration and diagnostics."""
import io
import json
import zipfile
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forward_mc import calibration, portfolio_weights, simulate_forward
from src.transaction_costs import PRESETS
from src.portfolio_context import portfolio_settings
from src.factor_currency import factor_covariance_available, factor_regions


STRATEGIES = [
    "Maximum Sharpe", "Minimum Volatility", "Risk Parity", "HRP", "Equal Weight",
    "CAPM Maximum Sharpe", "FF3 Maximum Sharpe", "FF5 Maximum Sharpe",
    "FF3 Factor Cov Maximum Sharpe", "FF5 Factor Cov Maximum Sharpe",
]


def load_prices(tickers):
    return _cached_prices(tickers, portfolio_settings())


@st.cache_data(ttl=3600, show_spinner=False, max_entries=8)
def _cached_prices(tickers, configuration):
    from src.data import get_prices
    return get_prices(list(tickers), period="max", preserve_missing=True)


def render_forward_mc(tickers, market, benchmark, risk_free, max_weight, theme):
    if not tickers:
        st.info("Select an asset universe in Asset Selection.")
        return
    if market not in ("Borsa Italiana", "USA"):
        st.info("Forward simulation currently supports Italian and US equities.")
        return
    configuration = portfolio_settings()
    currency = configuration["base_currency"] if configuration else ("EUR" if market == "Borsa Italiana" else "USD")
    try:
        prices = load_prices(tuple(tickers))
        if not set(tickers).issubset(prices.columns):
            raise ValueError("Missing market history for one or more selected assets.")
        prices = prices.reindex(columns=tickers)
        # Exclude today's possibly incomplete daily bar.
        zone = "Europe/Rome" if currency == "EUR" else "America/New_York"
        today = pd.Timestamp.now(tz=zone).date()
        prices = prices.loc[prices.index.date < today]
        complete = prices.dropna(how="any")
        if complete.empty:
            raise ValueError("No common completed market date is available.")
    except Exception as exc:
        st.warning(f"Market history unavailable: {exc}")
        return
    st.caption(f"Fixed candidate universe from Asset Selection · {'base currency' if configuration else 'native currency'} {currency}. "
               "Historical analysis is conditional on this universe and on retrospective price data.")
    with st.container(key="kangemi_controls_forward_mc"):
        left, right = st.columns(2)
        latest = complete.index[-1].date()
        date = left.date_input("Valuation Date", latest, min_value=prices.index[0].date(), max_value=latest)
        strategies = right.multiselect("Portfolio Strategies", STRATEGIES, default=STRATEGIES[:5])
        left, middle, right = st.columns(3)
        years = left.selectbox("Simulation Horizon (Years)", [.25,.5,1,2,3,5], index=2)
        paths = middle.selectbox("Number of Simulations", [1000,5000,10000,20000], index=1)
        capital = right.number_input(f"Initial Capital ({currency})", min_value=1., value=10000.)
        left, middle, right = st.columns(3)
        model = left.selectbox("Expected Return Model", ["FF3","Historical Mean","CAPM","FF5"])
        method = middle.selectbox("Simulation Method", ["Multivariate Normal","Historical Block Bootstrap","Factor Monte Carlo"])
        cov = right.selectbox("Covariance Estimation", ["Sample","EWMA","Ledoit–Wolf"], disabled=method=="Historical Block Bootstrap")
        left, middle, right = st.columns(3)
        window = left.selectbox("Estimation Window (Sessions)", [252,504,756,1260], index=2)
        decay = middle.number_input("EWMA lambda", min_value=.01,max_value=.999,value=.94, disabled=cov!="EWMA")
        block = right.number_input("Bootstrap Block Length", min_value=1,max_value=window,value=5)
        left, middle, right = st.columns(3)
        seed = left.number_input("Random Seed", min_value=0,value=42)
        sessions = middle.number_input("Trading Sessions per Year", min_value=1,max_value=366,value=252)
        management = right.selectbox("Portfolio Management", ["Buy & Hold","Constant Weight Rebalancing"])
        left, middle, right = st.columns(3)
        rebalance = left.selectbox("Rebalance frequency (sessions)", [21,63,126,252]) if management != "Buy & Hold" else 0
        cost = middle.selectbox("Transaction costs", ["None",*PRESETS])
        uncertainty = right.checkbox("Include Parameter Uncertainty")
        draws = right.number_input("Parameter draws", min_value=2,max_value=100,value=20) if uncertainty else 0
        run = st.button("Run Forward-Looking Simulation", type="primary", disabled=not strategies)
    config = dict(date=str(date), years=years, paths=paths, capital=capital, seed=seed,
                  model=model, method=method, covariance_method=cov, window=window,
                  decay=decay, block=block, sessions=sessions, rebalance=rebalance,
                  costs=cost, parameter_draws=draws, rf=risk_free, max_weight=max_weight,
                  market=market, currency=currency, tickers=list(tickers), strategies=strategies,
                  effective_universe=configuration)
    if run:
        st.session_state.pop("forward_mc_result", None)
        try:
            with st.spinner("Calibrating and simulating common market scenarios…"):
                from src.fama_french import load_us_ff3, load_europe_ff3
                from src.fama_french_5 import load_us_ff5, load_europe_ff5
                required = {model, *(name.split()[0] for name in strategies)}
                benchmark_prices = None
                loaders = {"FF3": (load_europe_ff3, load_us_ff3), "FF5": (load_europe_ff5, load_us_ff5)}
                factor_inputs = {}
                requires_factors = bool(required.intersection({"CAPM", "FF3", "FF5"}))
                common_factors = not requires_factors or factor_covariance_available(prices.columns)
                if not common_factors:
                    st.warning("A joint factor simulation requires a common factor set. For this regional configuration, use Historical Mean and historical strategies.")
                if common_factors and "CAPM" in required:
                    if configuration:
                        from src.hybrid_portfolio import BENCHMARKS

                        benchmark = BENCHMARKS[factor_regions(prices.columns).iloc[0]]
                    benchmark_prices = load_prices((benchmark,))[benchmark]
                for name in required.intersection(loaders) if common_factors else ():
                    try:
                        fx = load_prices(("EURUSD=X",)).iloc[:,0] if currency == "EUR" else None
                        factor_inputs[name] = (pd.DataFrame(index=prices.index) if configuration else loaders[name][currency == "USD"]()), fx
                    except Exception as exc:
                        st.warning(f"{name} data unavailable: {exc}")
                history, fits = calibration(prices, date, window, benchmark_prices, factor_inputs,
                                            market, risk_free/sessions)
                weights, skipped = portfolio_weights(history, fits, strategies, risk_free, max_weight, sessions)
                model_data = fits.get(model)
                if model != "Historical Mean" and model_data is None:
                    raise ValueError(f"No point-in-time input sample for {model}.")
                result, baseline = simulate_forward(
                    history, weights, model_data=model_data, method=method, covariance_method=cov,
                    decay=decay, block=block, paths=paths, steps=max(1, round(years*sessions)),
                    sessions=sessions, capital=capital, seed=seed, rf=risk_free,
                    rebalance=rebalance, costs=PRESETS.get(cost), parameter_draws=draws,
                )
                st.session_state["forward_mc_result"] = (
                    config, result, weights, baseline, history, model_data, skipped,
                )
        except (ValueError, KeyError, RuntimeError, np.linalg.LinAlgError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("forward_mc_result")
    if saved and saved[0] == config:
        render_results(saved, theme)
    elif saved:
        st.info("Run the simulation to apply the new settings.")
    with st.expander("Methodology — Forward-Looking Monte Carlo"):
        st.write("Calibrates on asset returns through the valuation date. The Backtest Robustness tab uses "
                 "realized portfolio returns from a backtest. All forward portfolios share identical asset shocks.")
        st.write("Risk-free and expected return conventions are arithmetic. Factor estimates omit regression alpha "
                 "from the drift and use a two-calendar-month publication buffer for FF3/FF5, a proxy for unavailable "
                 "historical release vintages. Factor simulation uses Gaussian factors and independent Gaussian "
                 "residual vectors with estimated cross-asset residual covariance.")
        st.write("Sample and shrinkage covariance use daily simple returns. EWMA uses normalized exponential weights "
                 "around the weighted mean. Bootstrap draws common overlapping blocks and recenters by block "
                 "position to match the target daily mean and uses empirical, rather than covariance-estimator, shocks. Parameter uncertainty uses repeated block resamples "
                 "of calibration returns and aligned factor observations, then refits within each draw.")
        st.write("Positions use fractional shares and full initial allocation. Buy & Hold allows weight drift. "
                 "Rebalancing restores target weights at the chosen interval. Adjusted-price returns include "
                 "distribution and split adjustments. Cost presets use the existing self-financing order model.")
        st.write("VaR and ES refer to terminal loss. Volatility uses pooled daily simulated returns scaled by "
                 "the square root of sessions per year. Sharpe uses annualized arithmetic daily excess return. "
                 "Outperformance uses strict wins and splits ties for best-strategy probability. "
                 "Assumptions and the selected calibration window determine the scenario distribution. "
                 "Multi-year ML drift is unavailable because the existing forecasts are short horizon.")


def render_results(saved, theme):
    config, result, weights, baseline, history, factor_data, skipped = saved
    currency = config["currency"]
    for name, reason in skipped.items():
        st.warning(f"{name} unavailable: {reason}")
    st.caption(f"Valuation close: {history.index[-1].date()} · {len(history)} daily returns · {currency}")
    st.subheader("Risk–Return Comparison")
    percentages = ["Mean Return","Median Return","Annualized Volatility","Probability of Loss","Median Max Drawdown"]
    monetary = ["Horizon VaR 95%","Horizon ES 95%","P5 Final Value","P95 Final Value"]
    st.dataframe(result["metrics"].style.format({
        **dict.fromkeys(percentages, "{:.2%}"), **dict.fromkeys(monetary, "{:,.2f}"),
        "Simulated Sharpe":"{:.2f}",
    }, na_rep="—"))
    names = list(weights.columns)
    strategy = st.selectbox("Portfolio Growth Fan Chart", names)
    chart = go.Figure()
    if st.checkbox("Show all simulated paths"):
        paths = result["paths"][:,:,names.index(strategy)]
        x = np.tile(np.r_[np.arange(paths.shape[1], dtype=float),np.nan], len(paths))
        y = np.column_stack([paths,np.full(len(paths),np.nan)]).ravel()
        chart.add_scattergl(x=x,y=y,mode="lines",line=dict(color="rgba(123,202,255,.08)"),name="Paths",hoverinfo="skip")
    for lo,hi,label in ((0,4,"P5–P95"),(1,3,"P25–P75")):
        chart.add_scatter(y=result["fans"][strategy][hi],line=dict(width=0),showlegend=False)
        chart.add_scatter(y=result["fans"][strategy][lo],line=dict(width=0),fill="tonexty",
                          fillcolor="rgba(123,202,255,.15)",name=label)
    chart.add_scatter(y=result["fans"][strategy][2],name="Median",line=dict(color="#ffbf69"))
    chart.add_hline(y=config["capital"],line_dash="dash",annotation_text="Initial capital")
    chart.update_layout(title="Portfolio Growth Fan Chart",xaxis_title="Session",yaxis_title=f"NAV ({currency})")
    theme(chart)
    st.plotly_chart(chart)
    dist = go.Figure()
    for name in names:
        dist.add_histogram(x=(1+result["final"][name])*config["capital"],name=name,histnorm="probability density",opacity=.50)
    dist.update_layout(title="Final Portfolio Value Distribution",barmode="overlay",xaxis_title=currency)
    theme(dist)
    st.plotly_chart(dist)
    pair = result["pair"]
    heat = go.Figure(go.Heatmap(x=names,y=names,z=pair.to_numpy(),zmin=0,zmax=1,colorbar=dict(tickformat=".0%")))
    heat.update_layout(title="Probability of Outperformance · row > column")
    theme(heat)
    st.plotly_chart(heat)
    st.subheader("Probability of Being the Best")
    st.bar_chart(result["best"])
    if len(names)>1:
        left,right = st.columns(2)
        i = left.selectbox("Strategy A",names)
        j = right.selectbox("Strategy B",names,index=1)
        diff = result["final"][i]-result["final"][j]
        low, high = diff.quantile([.025,.975])
        st.caption(f"Mean {diff.mean():+.2%} · Median {diff.median():+.2%} · Scenario P2.5–P97.5 "
                   f"{low:+.2%} to {high:+.2%} · P(A > B) {(diff>0).mean():.2%}")
        chart = go.Figure(go.Histogram(x=diff,histnorm="probability density"))
        chart.update_layout(title="Distribution of Return Differences",xaxis_tickformat=".0%")
        theme(chart)
        st.plotly_chart(chart)
    with st.expander("Research & Validation — Export Audit Data"):
        if st.button("Prepare Forward Audit Data"):
            archive = build_archive(config, result, weights, baseline, history, factor_data)
            st.download_button("Download Forward Audit Data",archive,"forward_mc_audit.zip","application/zip")


def build_archive(config, result, weights, baseline, history, factor_data):
    mu, cov, factors = baseline
    payload = io.BytesIO()
    with zipfile.ZipFile(payload,"w",compression=zipfile.ZIP_DEFLATED) as z:
        meta = dict(config, effective_valuation_date=str(history.index[-1].date()),
                    cost_assumptions=asdict(PRESETS[config["costs"]]) if config["costs"] in PRESETS else None,
                    numpy_version=np.__version__)
        z.writestr("parameters.json",json.dumps(meta,indent=2))
        for name, table in dict(weights=weights, asset_returns=history,
                                expected_daily=pd.Series(mu,index=weights.index),
                                covariance_daily=pd.DataFrame(cov,index=weights.index,columns=weights.index),
                                terminal_returns=result["final"], metrics=result["metrics"],
                                outperformance=result["pair"], probability_best=result["best"]).items():
            z.writestr(f"{name}.csv",table.to_csv())
        if factors is not None:
            for name, data in factors.items():
                z.writestr(f"factor_{name}.json",json.dumps(np.asarray(data).tolist()))
            y, f, rates, names = factor_data
            z.writestr("factor_sample_returns.csv",pd.DataFrame(y,columns=weights.index).to_csv(index=False))
            z.writestr("factor_sample_inputs.csv",pd.DataFrame(f,columns=names).assign(RF=rates).to_csv(index=False))
    return payload.getvalue()
