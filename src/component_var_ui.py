"""Streamlit presentation for backtest component VaR."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.component_var import dynamic_component_var
from src.backtest_registry import backtest_strategies
from src.return_distribution_ui import render_distributions


@st.cache_data(show_spinner=False, max_entries=24)
def cached_dynamic_var(exposures, prices, window):
    return dynamic_component_var(exposures, prices, window)


def render_dynamic_component_var(result, key, company_name):
    strategies = backtest_strategies(result)
    options = [label for label, run in strategies.items() if run.get("weights") is not None]
    if not options:
        st.info("No strategy curves available.")
        return
    cols = st.columns(3)
    strategy = cols[0].selectbox("Strategy", options, key=f"component_var_strategy_{key}")
    window = cols[1].number_input("Covariance window (sessions)", min_value=2, value=252,
                                  step=1, key=f"component_var_window_{key}")
    frequency = cols[2].selectbox("Display frequency", ["Daily", "Weekly", "Monthly"],
                                  key=f"component_var_frequency_{key}")
    curve = strategies[strategy]["curve"]
    prices = result.get("risk_prices")
    render_component_chart(curve, prices, window, frequency, strategy, key, company_name)
    render_distributions(curve, prices, strategies[strategy]["weights"], strategy, key, company_name)


def render_component_chart(curve, prices, window, frequency, strategy, key, company_name):
    marks = curve.attrs.get("risk_position_marks")
    if marks is None or prices is None:
        st.info("Rerun this backtest to capture executed position values and estimation prices.")
        return
    try:
        daily = cached_dynamic_var(marks.loc[marks.index.intersection(curve.index)], prices, window)
    except ValueError as exc:
        st.warning(f"Component VaR unavailable: {exc}")
        return
    if daily.empty:
        st.info(f"No dates have {window} complete daily returns for the held assets.")
        return
    unavailable = len(curve.index.difference(daily["Date"]))
    if unavailable:
        st.caption(f"{unavailable} valuation dates have insufficient history or no invested exposure.")
    dates = pd.DatetimeIndex(daily["Date"].unique()).sort_values()
    if frequency != "Daily":
        periods = dates.to_period("W" if frequency == "Weekly" else "M")
        dates = dates[~periods.duplicated(keep="last")]
    fig = go.Figure()
    totals = daily.groupby("Date")["Total Portfolio VaR (EUR)"].first().reindex(dates)
    for asset, rows in daily.groupby("Asset"):
        # The estimator emits rows only for held assets. Align the two plotted
        # measures to valid portfolio dates, using zero only for unheld assets.
        shown = rows.set_index("Date")[
            ["Component VaR (EUR)", "Component VaR Contribution (%)"]
        ].reindex(dates, fill_value=0.)
        percent = shown["Component VaR Contribution (%)"].where(totals.notna() & totals.ne(0))
        fig.add_scatter(
            x=dates, y=percent, name=f"{company_name(asset)} ({asset})", mode="lines+markers",
            marker=dict(size=3), customdata=shown["Component VaR (EUR)"],
            hovertemplate="%{x|%Y-%m-%d}<br>Contribution: %{y:.2f}%<br>Component VaR: €%{customdata:,.2f}",
        )
    fig.add_hline(y=0, line_color="#ffbf69", line_dash="dash")
    fig.update_layout(template="plotly_dark", xaxis_title="Date",
                      yaxis_title="Component VaR Contribution (%)")
    st.plotly_chart(fig, key=f"component_var_chart_{key}")
    export = daily.assign(Strategy=strategy)[[
        "Date", "Strategy", "Asset", "Invested Amount (EUR)", "Component VaR (EUR)",
        "Component VaR Contribution (%)", "Total Portfolio VaR (EUR)",
    ]]
    st.download_button("Download Dynamic Component VaR (CSV)", export.to_csv(index=False),
                       "dynamic_component_var.csv", "text/csv", key=f"component_var_download_{key}")
    with st.expander("Methodology — Dynamic Component VaR"):
        st.write(f"One-day 99% Gaussian VaR with zero drift and a rolling sample covariance of {window} "
                 "complete daily returns through the valuation close. The resulting estimate is forward-looking "
                 "as of that close. Historical covariance is used for every strategy.")
        st.write("Each signed contribution is the marked position value times its marginal VaR. The contributions "
                 "sum to total VaR by Euler decomposition. Negative contributions offset portfolio risk. Position "
                 "values, volatilities and correlations change over time. VaR estimates risk and does not measure "
                 "actual losses. Percent contributions are undefined at zero VaR. Weekly and monthly views select "
                 "the last available daily estimate; the CSV retains daily estimates.")
