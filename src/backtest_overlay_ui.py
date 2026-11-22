"""Streamlit controls and comparison details for exposure variants."""
import pandas as pd
import streamlit as st

from src.backtest_registry import strategy_options
from src.dynamic_risk_backtest import MODELS
from src.volatility_forecasting import ARCH_AVAILABLE


def overlay_controls(configs):
    classic = strategy_options({})
    selected = strategy_options(configs)
    with st.container(border=True, key="backtest_overlay_panel"):
        st.subheader("Dynamic Risk Overlay — Strategy Variants")
        if not st.checkbox("Enable Dynamic Risk Overlay comparison", key="bt_enable_overlay"):
            return selected, [], [], {}
        originals = st.multiselect("Classic and factor portfolios for comparison", classic,
                                   default=classic, key="bt_original_strategies")
        selected = [name for name in selected if name in originals or name in configs]
        models = list(MODELS if ARCH_AVAILABLE else ("EWMA",))
        if not ARCH_AVAILABLE:
            st.info("GARCH models require the arch package. EWMA is available.")
        prior = st.session_state.get("bt_overlay_portfolios")
        if prior is not None:
            st.session_state["bt_overlay_portfolios"] = [name for name in prior if name in selected]
        portfolios = st.multiselect("Select portfolios for overlay", selected, key="bt_overlay_portfolios")
        models = st.multiselect("Overlay volatility models", models, default=models[:2], key="bt_overlay_models")
        left, center, right = st.columns(3)
        target = left.number_input("Annual volatility target (%)", 5., 50., 15.)
        horizon = center.selectbox("Forecast horizon (sessions)", [5, 21, 63], index=1)
        refresh = right.selectbox("Forecast refresh (sessions)", [1, 5, 21], index=1)
        train = left.number_input("Pre-test estimation (sessions)", 252, 1000, 252)
        trade = center.number_input("Reduce exposure from (pp)", 5., 50., 10.)
        review = right.number_input("Review from (pp)", 0., trade - 1., min(5., trade - 1.))
        reentry = center.number_input("Restore exposure from (pp)", 1., trade, min(5., trade))
        cash = left.number_input("Annual cash yield (%)", 0., 15., 0., .5)
        st.caption("Maximum exposure 100%. Orders use the Backtesting cost scenario and execute "
                   "at the next session’s close after the signal. The forecast starts from "
                   "a pre-test return proxy using the initial portfolio weights.")
        if not portfolios or not models:
            st.info("Select at least one portfolio and one model to add variants.")
        return selected, portfolios, models, dict(
            target_vol=target, horizon=horizon, update_every=refresh, train=train,
            trade_band_pp=trade, review_band_pp=review, reentry_band_pp=reentry,
            cash_annual_yield=cash / 100,
        )


def render_overlay_details(runs, result_key, render_table):
    st.caption("Differences are overlay minus original on the same out-of-sample dates. "
               "Cost differences compare total order costs across the two runs.")
    deltas = pd.DataFrame({run["label"]: run["delta"] for run in runs.values()}).T
    render_table(deltas, {"Total Return": "{:+.2%}", "Volatility": "{:+.2%}",
                          "Max Drawdown": "{:+.2%}", "Sharpe Ratio": "{:+.2f}",
                          "Transaction Costs (€)": "€{:+,.2f}"}, index_label="Overlay")
    with st.expander("Exposure signals and orders"):
        variant = st.selectbox("Overlay variant", list(runs), format_func=lambda key: runs[key]["label"],
                               key=f"overlay_log_{result_key}")
        attrs = runs[variant]["curve"].attrs
        log = attrs.get("exposure_log")
        if log is not None and (log["Action"] == "FORECAST ERROR").any():
            st.warning("Some forecasts failed. Exposure was held at its last setting on those dates.")
        for name in ("exposure_log", "transaction_log"):
            if name in attrs:
                st.dataframe(attrs[name], width="stretch")
