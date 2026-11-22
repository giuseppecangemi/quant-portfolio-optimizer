"""Streamlit workspace for equity scenario construction."""
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
import streamlit as st

from src.mc_coverage import ADAPTIVE, MANUAL, MIN_SESSIONS, STRICT, assess_coverage, download_history

cached_coverage_history = st.cache_data(ttl=3600, show_spinner=False, max_entries=4)(download_history)
from src.mc_construction import (
    STATES, STRATEGIES, calibrate_ffr, cvar_contributions, draw_paths, equity_logs, scenario_allocation, state_variables,
)


def _chart(fig, title):
    fig.update_layout(template="plotly_dark", title=dict(text=title, font_color="#ffbf69"))
    st.plotly_chart(fig)


def _stationarity(prices, logs):
    asset = st.selectbox("Asset", list(logs))
    series = logs[asset]
    _chart(px.line(prices[asset]), "Adjusted price")
    _chart(px.line(series), "Daily log return")
    _chart(px.histogram(series, histnorm="probability density", nbins=35, marginal="box"), "Empirical density")
    theoretical, observed = stats.probplot(series, fit=False)
    _chart(px.scatter(x=theoretical, y=observed, labels={"x": "Normal quantile", "y": "Observed quantile"}), "Normal Q–Q")
    lags = min(30, len(series)//3)
    serial = pd.DataFrame({"ACF": acf(series, nlags=lags), "PACF": pacf(series, nlags=lags),
                           "Absolute ACF": acf(series.abs(), nlags=lags),
                           "Squared ACF": acf(series**2, nlags=lags)})
    _chart(px.line(serial, labels={"index": "Lag"}), "Serial dependence")
    _chart(px.line(pd.DataFrame({"Mean": series.rolling(21).mean(), "Volatility": series.rolling(21).std()})),
           "Rolling daily moments · 21 sessions")
    _chart(px.imshow(logs.corr(), zmin=-1, zmax=1, color_continuous_scale="RdBu_r"), "Log-return correlations")
    st.subheader("Stationarity diagnostics · laboratory extensions")
    adf = adfuller(series, autolag="AIC")
    level = kpss(series, regression="c", nlags="auto")
    st.dataframe(pd.DataFrame([
        ("ADF", adf[0], adf[1], adf[2], "Unit root"),
        ("KPSS", level[0], level[1], level[2], "Level stationary"),
    ], columns=["Test", "Statistic", "p-value", "Lags", "Null hypothesis"]))
    st.caption("A small p-value is evidence against the corresponding null. Failure to reject provides no proof of stationarity. KPSS p-values may be bounded by its reference table.")
    st.subheader("Serial dependence and distribution")
    lag = min(10, lags)
    st.dataframe(pd.concat({"Returns": acorr_ljungbox(series, lags=[lag]),
                             "Squared returns": acorr_ljungbox(series**2, lags=[lag])}, names=["Series", "Lag"]))
    lm, lm_p, _, _ = het_arch(series, nlags=lag)
    jb = stats.jarque_bera(series) if len(series) > 2000 else stats.normaltest(series)
    st.write({"ARCH-LM statistic": lm, "ARCH-LM p-value": lm_p, "Skewness": series.skew(),
              "Excess kurtosis": series.kurt(), "Normality test": "Jarque–Bera" if len(series)>2000 else "D’Agostino–Pearson",
              "Normality p-value": jb.pvalue})


def _validation(logs, paths, state_paths, labels, calibration):
    wealth = 1 + paths
    daily = np.diff(np.log(wealth), axis=2, prepend=0)
    simulated = pd.DataFrame(daily.transpose(0, 2, 1).reshape(-1, len(labels)), columns=labels)
    st.caption("In-sample calibration diagnostics. Simulated moments are conditional on the last observed state.")
    def moments(frame):
        return pd.DataFrame({"Mean": frame.mean(), "Volatility": frame.std(),
                             "Skewness": frame.skew(), "Excess kurtosis": frame.kurt()})
    st.dataframe(pd.concat({"Historical": moments(logs), "Simulated": moments(simulated)}, axis=1))
    _chart(px.imshow(simulated.corr()-logs.corr(), color_continuous_scale="RdBu_r", color_continuous_midpoint=0),
           "Change in cross-asset correlations")
    asset = st.selectbox("Validation asset", labels)
    h = min(10, paths.shape[2]-1)
    if h:
        sim = daily[:, labels.index(asset), :]
        def mean_path_acf(power):
            series = np.abs(sim)**power
            valid = np.ptp(series, axis=1) > 0
            return np.mean([acf(row, nlags=h) for row in series[valid]], axis=0) if valid.any() else np.full(h+1, np.nan)
        _chart(px.line(pd.DataFrame({"Historical |r|": acf(logs[asset].abs(), nlags=h),
                                    "Simulated |r|": mean_path_acf(1),
                                    "Historical r²": acf(logs[asset]**2, nlags=h),
                                    "Simulated r²": mean_path_acf(2)})), "Autocorrelation within paths")
    state_comparison = pd.concat({
        "Observed": calibration["logs"][asset].groupby(calibration["membership"]).mean(),
        "Simulated": pd.Series(daily[:, labels.index(asset), :].ravel()).groupby(state_paths.ravel()).mean(),
    }, axis=1)
    st.dataframe(state_comparison)
    if len(labels) > 1:
        _, _, vectors = np.linalg.svd(logs.to_numpy()-logs.mean().to_numpy(), full_matrices=False)
        pairs = []
        for name, frame in (("Historical", logs), ("Simulated", simulated.iloc[:3000])):
            scores = (frame.to_numpy()-logs.mean().to_numpy()) @ vectors[:2].T
            pairs.append(pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "Sample": name}))
        _chart(px.scatter(pd.concat(pairs), x="PC1", y="PC2", color="Sample", opacity=.45), "Common historical PCA basis")


def render_mc_construction(tickers, market, rf, cap, theme, benchmark="FTSEMIB.MI"):
    st.header("Monte Carlo Portfolio Construction")
    theme()
    tabs = st.tabs(["Data & Stationarity Lab", "Monte Carlo Simulation Lab", "Simulation Validation",
                    "Scenario-Based Portfolio Optimization", "Portfolio Analytics & Export"])
    with tabs[0]:
        lookback = st.selectbox("Calibration history", [252, 504, 756, 1260], index=1)
        policy = st.selectbox("Universe selection policy", [STRICT, ADAPTIVE, MANUAL])
        minimum = st.number_input("Minimum calibration sessions", MIN_SESSIONS, lookback, min(252, lookback)) if policy == ADAPTIVE else MIN_SESSIONS
        key = (market, benchmark, tuple(dict.fromkeys(tickers)))
        if st.button("Load daily prices", disabled=not tickers):
            st.session_state.pop("mc_coverage_raw", None)
            st.session_state.pop("mc_coverage_settings", None)
            try:
                with st.spinner("Checking historical coverage by asset…"):
                    st.session_state["mc_coverage_raw"] = (key, *cached_coverage_history(key[2], benchmark, market))
            except ValueError as exc:
                st.error(str(exc))
        raw = st.session_state.get("mc_coverage_raw")
        if raw is None or raw[0] != key:
            st.info("Load the currently selected equity universe to start.")
            return
        _, universe, calendar, errors = raw
        if policy == MANUAL:
            _, strict_report = assess_coverage(universe, calendar, lookback, download_errors=errors)
            default = strict_report.loc[strict_report["Eligible"], "Asset"].tolist()
            selected = st.multiselect("Assets for calibration", list(universe), default=default,
                                      key=f"mc_manual_{market}_{benchmark}_{lookback}_{','.join(key[2])}")
        else:
            selected = list(universe)
        prices, report = assess_coverage(universe, calendar, lookback, policy, minimum,
                                         selected=selected, download_errors=errors)
        st.subheader("Asset Coverage & Eligibility")
        a, b, c = st.columns(3)
        a.metric("Selected", len(selected))
        b.metric("Eligible", prices.shape[1])
        c.metric("Excluded from calibration", len(universe) - prices.shape[1])
        st.caption(f"Reference sessions: {benchmark} · Coverage measured against {lookback + 1} closing prices "
                   f"({lookback} daily returns). Data gaps are left unfilled.")
        st.dataframe(report, hide_index=True,
                     column_config={"Coverage": st.column_config.NumberColumn(format="percent")})
        st.download_button("Download eligibility report (CSV)", report.to_csv(index=False),
                           "mc_asset_eligibility.csv", "text/csv")
        spans = report.dropna(subset=["First available", "Last available"])
        if not spans.empty:
            spans = spans.assign(Status=np.where(spans["Eligible"], "Eligible", "Excluded"))
            _chart(px.timeline(spans, x_start="First available", x_end="Last available",
                               y="Asset", color="Status"), "Observed history by asset")
            st.caption("Bars show first and last valid closes. Internal gaps are listed in the table.")
        settings = (key, lookback, policy, minimum, tuple(selected), tuple(prices.columns),
                    tuple(prices.index), cap)
        if st.session_state.get("mc_coverage_settings") != settings:
            st.session_state["mc_coverage_settings"] = settings
            st.session_state.pop("mc_construction_run", None)
            st.session_state.pop("mc_construction_portfolios", None)
        if prices.empty:
            st.warning(f"No assets meet {policy} for the requested window. See the exclusion reasons above.")
            return
        logs = equity_logs(prices)
        st.caption(f"Effective calibration: {prices.index[0]:%Y-%m-%d} → {prices.index[-1]:%Y-%m-%d} "
                   f"· {len(logs)} return sessions · {lookback-len(logs)} requested sessions lost")
        if prices.shape[1] * cap < 1 - 1e-12:
            st.warning("Calibration is available. The eligible asset count and current weight cap make portfolio optimization infeasible.")
        try:
            _stationarity(prices, logs)
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.warning(f"Stationarity diagnostics unavailable: {exc}")
    with tabs[1]:
        a, b, c, d = st.columns(4)
        simulations = a.number_input("Paths", 100, 5000, 500, 100)
        horizon = b.number_input("Horizon (sessions)", 1, 63, 21)
        half_life = c.number_input("Prior half-life (sessions)", 10, 2520, 252)
        seed = d.number_input("Seed", 0, 2**31-1, 42)
        states = st.multiselect("Observable state variables", STATES, default=[STATES[0]])
        lo, hi = st.slider("Quantile boundaries", .05, .95, (.25, .75), .05)
        st.caption("States are trailing, equally weighted universe statistics. Paths start from the latest observed state. Prior weights decay exponentially.")
        if st.button("Run FFR", disabled=not states):
            try:
                calibration = calibrate_ffr(logs, state_variables(logs, states), half_life, (lo, hi))
                paths, state_paths = draw_paths(calibration, simulations, horizon, seed)
                config = dict(states=states, half_life=half_life, cuts=(lo, hi), simulations=simulations,
                              horizon=horizon, seed=seed, market=market, family="MC Construction",
                              version="Vorobets 2026-10-22 · equity",
                              coverage_policy=policy, coverage_minimum=minimum,
                              coverage_selection=selected if policy == MANUAL else None)
                st.session_state["mc_construction_run"] = (calibration, paths, state_paths, config)
                st.session_state["mc_construction_portfolios"] = {}
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
        run = st.session_state.get("mc_construction_run")
        if run is None:
            return
        calibrated, paths, state_paths, config = run
        st.caption(f"Saved run · {paths.shape[0]} paths × {paths.shape[1]} assets × {paths.shape[2]} sessions")
        asset = st.selectbox("Simulated asset", list(logs), key="construction_asset")
        asset_paths = paths[:, logs.columns.get_loc(asset), :]
        fan = pd.DataFrame(np.quantile(asset_paths, [.05,.25,.5,.75,.95], axis=0).T,
                           columns=["5%", "25%", "50%", "75%", "95%"], index=range(1, paths.shape[2]+1))
        _chart(px.line(fan), "Fan chart · cumulative return quantiles")
        _chart(px.line(asset_paths[:15].T), "Sample equity paths")
        day = st.slider("Distribution horizon", 1, paths.shape[2], paths.shape[2]) if paths.shape[2] > 1 else 1
        _chart(px.histogram(asset_paths[:, day-1], nbins=35), "Cumulative return distribution")
        _chart(px.imshow(calibrated["transition"], zmin=0, zmax=1), "Implied state transition matrix")
        _chart(px.imshow(calibrated["q"], aspect="auto"), "Posterior probability by state and historical observation")
        occupancy = pd.DataFrame({j: (state_paths == j).mean(axis=0) for j in range(len(calibrated["q"]))})
        _chart(px.line(occupancy), "Simulated state occupancy")
    with tabs[2]:
        _validation(logs, paths, state_paths, list(logs), calibrated)
        st.subheader("Out-of-sample holdout")
        st.caption("Single historical origin. Calibration stops before the held-out path.")
        if st.button("Validate held-out path"):
            h = paths.shape[2]
            train, held = logs.iloc[:-h], logs.iloc[-h:]
            try:
                history_states = state_variables(train, config["states"])
                hold_fit = calibrate_ffr(train, history_states, config["half_life"], config["cuts"])
                hold_paths, _ = draw_paths(hold_fit, config["simulations"], h, config["seed"])
                realized = np.expm1(held.sum())
                lower, median, upper = np.quantile(hold_paths[:, :, -1], [.05, .5, .95], axis=0)
                st.dataframe(pd.DataFrame({"Realized": realized, "Forecast P05": lower,
                                           "Forecast median": median, "Forecast P95": upper,
                                           "Inside interval": (realized >= lower) & (realized <= upper)}))
                st.caption(f"Decision {train.index[-1]:%Y-%m-%d} · Held-out end {held.index[-1]:%Y-%m-%d}. "
                           "One held-out path cannot establish interval coverage.")
            except ValueError as exc:
                st.warning(f"Holdout unavailable: {exc}")
    with tabs[3]:
        strategy = st.selectbox("Objective", STRATEGIES)
        alpha = st.slider("Confidence", .80, .99, .95, .01)
        budget = st.number_input("Horizon CVaR budget (%)", value=10.) / 100 if strategy == STRATEGIES[2] else None
        st.caption("Risk budgeting and Resampled Portfolio Stacking are outside the implemented equity workflow.")
        if st.button("Optimize scenario portfolio"):
            try:
                w, metrics = scenario_allocation(paths, strategy, alpha, cap, budget)
                if config["simulations"] * (1-alpha) < 10:
                    st.warning("Fewer than ten effective observations in the empirical CVaR tail.")
                saved_config = {**config, "strategy": strategy, "alpha": alpha, "budget": budget}
                st.session_state.setdefault("mc_construction_portfolios", {})[strategy] = (w, metrics, saved_config)
            except ValueError as exc:
                st.error(str(exc))
    with tabs[4]:
        portfolios = st.session_state.get("mc_construction_portfolios", {})
        if not portfolios:
            st.info("Optimize at least one strategy to view and export its allocation.")
            return
        name = st.selectbox("Portfolio", list(portfolios))
        w, metrics, config = portfolios[name]
        portfolio = paths.transpose(0, 2, 1) @ w
        h_rf = (1+rf)**(paths.shape[2]/252)-1
        st.dataframe(pd.DataFrame([{**metrics, "Horizon Sharpe": (metrics["expected_return"]-h_rf)/metrics["volatility"]
                                    if metrics["volatility"] else np.nan}]))
        allocation = pd.Series(w, index=logs.columns, name="Weight")
        _chart(px.bar(allocation), "Scenario-optimized weights")
        _chart(px.histogram(-portfolio[:, -1]), "Terminal loss distribution")
        wealth = np.c_[np.ones(len(portfolio)), 1+portfolio]
        _chart(px.histogram((1-wealth/np.maximum.accumulate(wealth, axis=1)).max(axis=1)), "Maximum drawdown by path")
        st.caption("CML measures loss relative to initial capital. Maximum drawdown measures decline relative to the running peak.")
        marginal, component = cvar_contributions(paths, w, config["alpha"])
        st.dataframe(pd.DataFrame({"Weight": allocation, "Marginal terminal CVaR": marginal,
                                   "Component terminal CVaR": component}))
        st.caption("Terminal CVaR contributions use a fractional tail with tied observations weighted equally.")
        st.download_button("Download weights", allocation.to_csv(index_label="Ticker"), "mc_weights.csv")
        st.download_button("Download configuration and solver metadata",
                           pd.Series({**config, **metrics, "as_of": str(prices.index[-1]), "weight_cap": cap}).to_json(),
                           "mc_configuration.json")
        if st.button("Register for Backtesting"):
            st.session_state.setdefault("mc_construction_registry", {})[(market, name)] = config
            st.success("Registered. Select this configuration in Backtesting.")
