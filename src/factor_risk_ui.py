"""Presentation of historical ex-ante Fama–French variance estimates."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.backtest_registry import LEGACY

STRATEGIES = {
    "FF3": "FF3 Maximum Sharpe",
    "FF5": "FF5 Maximum Sharpe",
    "FF3 Factor Cov": "FF3 Factor Cov Maximum Sharpe",
    "FF5 Factor Cov": "FF5 Factor Cov Maximum Sharpe",
}


def render_factor_risk(result, key, metric):
    available = {}
    for label, name in STRATEGIES.items():
        curve_key, history_key = LEGACY[name]
        curve, weights = result.get(curve_key), result.get(history_key)
        if isinstance(curve, pd.DataFrame) and not curve.empty and isinstance(weights, pd.DataFrame) and not weights.empty:
            dates = pd.DatetimeIndex(weights.index).intersection(curve.index)
            available[label] = {d.date().isoformat(): curve.attrs.get("factor_risk_history", {}).get(d.date().isoformat())
                                for d in dates}
    if not available:
        st.info("No Fama–French strategies are available in this backtest.")
        return
    strategy = st.selectbox("Strategy", list(available), key=f"factor_risk_strategy_{key}")
    history = available[strategy]
    valid = {date: row for date, row in history.items() if row and "error" not in row}
    missing = {date: row.get("error", "Estimate unavailable") if row else "Estimate unavailable"
               for date, row in history.items() if date not in valid}
    if missing:
        with st.expander(f"Unavailable estimates · {len(missing)}"):
            st.dataframe(pd.Series(missing, name="Reason"))
    if not valid:
        st.info("Rerun the backtest to collect the historical risk decomposition.")
        return
    dates = sorted(valid, reverse=True)
    date = st.selectbox("Rebalance date", dates, key=f"factor_risk_date_{key}") if len(dates) > 1 else dates[0]
    row = valid[date]
    st.caption(f"Post-rebalance target weights · {date} · information cutoff {row['decision_cutoff'][:10]}")
    total = row["total"]

    def share(variance):
        return f"{variance/total:.2%}" if total > 1e-14 else "—"

    for column, (label, value, color) in zip(st.columns(4), (
        ("Total Estimated Volatility", f"{row['volatility']:.2%}", "blue"),
        ("Systematic Risk Share", share(row["systematic"]), "green"),
        ("Residual Risk Share", share(row["residual"]), "purple"),
        ("Reconciliation Difference", f"{row['difference']:.2e}", "yellow"),
    )):
        with column:
            metric(label, value, color)
    values = pd.Series(row["contributions"]).rename(index={"MKT-RF": "Market"})
    table = values.rename("Variance Contribution").to_frame()
    exposure = pd.Series(row["exposures"]).rename(index={"MKT-RF": "Market"})
    table.insert(0, "Portfolio Exposure", exposure)
    table.loc["Total"] = [np.nan, total]
    table["Share of Total Variance (%)"] = 100*table["Variance Contribution"]/total if total > 1e-14 else np.nan
    st.dataframe(table.rename_axis("Factor").reset_index(), hide_index=True,
                 column_config={"Variance Contribution": st.column_config.NumberColumn(format="%.8f"),
                                "Share of Total Variance (%)": st.column_config.NumberColumn(format="%+.2f")})
    bar = px.bar(values.rename_axis("Factor").reset_index(name="Annual variance contribution"),
                 x="Annual variance contribution", y="Factor", orientation="h",
                 title="Factor Risk Contributions", template="plotly_dark")
    bar.add_vline(x=0, line_color="#ffbf69", line_dash="dash")
    st.plotly_chart(bar)
    timeline = pd.DataFrame.from_dict({d: r["contributions"] for d, r in valid.items()}, orient="index")
    timeline = timeline.rename(columns={"MKT-RF": "Market"}).reindex(sorted(history))
    if len(history) > 1:
        trend = px.line(timeline, markers=True, title="Risk Decomposition Over Time", template="plotly_dark",
                        labels={"index": "Rebalance date", "value": "Annual variance contribution", "variable": "Factor"})
        trend.add_hline(y=0, line_color="#ffbf69", line_dash="dash")
        st.plotly_chart(trend)
    components = pd.DataFrame.from_dict(
        {d: {"Systematic": r["systematic"], "Residual": r["residual"]} for d, r in valid.items()},
        orient="index",
    ).reindex(sorted(history))
    plot = px.line if len(history) > 1 else px.bar
    st.plotly_chart(plot(components, title="Systematic vs Residual Risk", template="plotly_dark",
                        labels={"index": "Rebalance date", "value": "Annual variance", "variable": "Component"}))
    with st.expander("Estimation diagnostics"):
        st.write(f"Aligned daily observations: {row['observations']} · Assets: {row['assets']} · "
                 f"Sample: {row['sample_start'][:10]} to {row['sample_end'][:10]} · Factor archive cutoff: "
                 f"{row['factor_cutoff'][:10]} · Annualization: {row['annualization']} sessions.")
        st.write("Joint OLS on excess returns with intercept. Factor covariance is sample covariance. "
                 "Residual covariance is diagonal with asset-specific OLS variance (residual degrees of freedom). "
                 "Numerical regularization: none.")
        if row["uses_factor_cov"]:
            st.write("The diagnostic uses the factor covariance supplied to the optimizer.")
        else:
            st.write("The optimizer uses historical asset covariance. This decomposition uses a separate factor-implied estimate.")
        st.write(f"Optimizer annual variance {row['optimizer_variance']:.8f} · "
                 f"Factor-implied minus optimizer variance {total-row['optimizer_variance']:+.3e}.")
