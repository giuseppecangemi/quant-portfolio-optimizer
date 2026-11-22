"""Interactive market volatility dashboard."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.vix_analytics import PERIODS, compare_markets, date_bounds, rolling_statistics
from src.vix_data import SOURCES, load_history

COLORS = {"VIX": "#ffb454", "VSTOXX": "#55d9f5"}
LEVEL_COLORS = ["#79b8a6", "#a6b97a", "#e9be60", "#ff9f55", "#fa7669", "#f25566"]
BAND_COLORS = ["#8297ad", "#cfad69", "#fa7669"]


def initial_levels(index, close):
    if index == "VIX":
        return [(v, label) for v, label in zip(
            [15., 20., 25., 30., 35., 40.],
            ["Low volatility", "Moderate", "Rising tension", "High fear", "Marked stress", "Extreme stress"],
        )]
    return [(round(float(close.quantile(q)), 2), f"Historical P{q:.0%}".replace("%", ""))
            for q in (.25, .5, .75, .9, .95)]


def period_controls(dates, key):
    period = st.selectbox("Display period", list(PERIODS), index=3, key=f"{key}_period")
    custom = None
    if period == "Custom":
        custom = st.date_input("Start and end", (dates[0].date(), dates[-1].date()),
                               min_value=dates[0].date(), max_value=dates[-1].date(), key=f"{key}_dates")
        if len(custom) != 2:
            st.info("Choose both dates.")
            return None
    return date_bounds(dates, period, custom)


def terminal_chart(fig, start, end, ylabel, slider=False, selector=True):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#111722", plot_bgcolor="#111722",
                      font=dict(color="#cbd5e1"), margin=dict(l=12, r=12, t=55, b=18),
                      height=390, hovermode="x unified", dragmode="zoom",
                      legend=dict(orientation="h", y=-.22, x=0))
    fig.update_yaxes(title=ylabel, gridcolor="#263140", fixedrange=False)
    fig.update_xaxes(title="Date", range=[start, end], gridcolor="#263140",
                     rangeslider=dict(visible=slider))
    if selector:
        fig.update_xaxes(rangeselector=dict(
            bgcolor="#202b39", activecolor="#3e4c60", font=dict(color="#e2e8f0"),
            buttons=[dict(count=n, step=step, stepmode="backward", label=label)
                     for n, step, label in ((1, "month", "1M"), (3, "month", "3M"),
                                             (6, "month", "6M"), (1, "year", "1Y"),
                                             (3, "year", "3Y"), (5, "year", "5Y"), (10, "year", "10Y"))]
                    + [dict(step="all", label="MAX")]))
    return fig


def index_chart(index, frame, bounds, mean, bands, levels, slider):
    fig = go.Figure()
    for k in bands:
        for side in ("Lower", "Upper"):
            fig.add_scatter(x=frame.index, y=frame[f"{side} {k}"], mode="lines",
                            name=f"±{k}σ", legendgroup=f"band{k}", showlegend=side == "Upper",
                            line=dict(color=BAND_COLORS[k-1], width=1, dash="dot"), hoverinfo="skip")
    if mean:
        fig.add_scatter(x=frame.index, y=frame["Mean"], mode="lines", name="Rolling mean",
                        line=dict(color="#e2e8f0", width=1), hoverinfo="skip")
    fig.add_scatter(x=frame.index, y=frame["Close"], mode="lines", name=index,
                    line=dict(color=COLORS[index], width=2),
                    customdata=frame[["Mean", "Std", "Z", "Percentile"]].to_numpy(),
                    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f} pts<br>Mean %{customdata[0]:.2f} · σ %{customdata[1]:.2f}"
                                  "<br>Z %{customdata[2]:+.2f} · P %{customdata[3]:.1f}%<extra></extra>")
    for i, row in enumerate(levels):
        if row["Visible"] and pd.notna(row["Level"]) and float(row["Level"]) > 0:
            level = float(row["Level"])
            color = LEVEL_COLORS[min(i, len(LEVEL_COLORS)-1)]
            fig.add_hline(y=level, line_color=color, line_width=1, line_dash="dash",
                          annotation_text=f"{level:g} · {row['Label'] or ''}", annotation_font_color=color,
                          annotation_position="bottom left")
    return terminal_chart(fig, *bounds, "Index points", slider)


def show_chart(fig, key):
    st.plotly_chart(fig, width="stretch", key=key, config={
        "displaylogo": False, "scrollZoom": True,
        "toImageButtonOptions": {"format": "png", "filename": key, "scale": 2},
    })


def show_index(index, close, metric):
    bounds = period_controls(close.index, index)
    a, b = st.columns(2)
    window = a.selectbox("Rolling window", [21, 63, 126, 252, "Custom"], index=3, key=f"{index}_window")
    if window == "Custom":
        window = a.number_input("Observations", 2, 5000, 252, key=f"{index}_window_custom")
    bands = b.multiselect("Standard deviation bands", [1, 2, 3], format_func=lambda k: f"±{k}σ",
                          key=f"{index}_bands")
    with st.expander("Indicators and levels"):
        mean = st.checkbox("Rolling mean", value=True, key=f"{index}_mean")
        slider = st.checkbox("Range slider", key=f"{index}_slider")
        show_levels = st.checkbox("Show horizontal levels", value=True, key=f"{index}_show_levels")
        defaults_key = f"{index}_level_defaults"
        editor_key = f"{index}_levels"
        saved_key = f"{index}_level_saved"
        calibration_key = f"{index}_level_calibration"
        if defaults_key not in st.session_state:
            st.session_state[defaults_key] = pd.DataFrame(
                initial_levels(index, close), columns=["Level", "Label"],
            ).assign(Visible=True)
            st.session_state[calibration_key] = (close.index[0], close.index[-1])
        if index == "VSTOXX":
            start, end = st.session_state[calibration_key]
            st.caption(f"Initial levels use historical quantiles, {start:%Y-%m-%d} to {end:%Y-%m-%d}.")
        if editor_key not in st.session_state and saved_key in st.session_state:
            st.session_state[defaults_key] = st.session_state[saved_key]
        levels = st.data_editor(st.session_state[defaults_key], num_rows="dynamic", hide_index=True,
                                key=editor_key, width="stretch",
                                column_config={"Level": st.column_config.NumberColumn(min_value=.01, format="%.2f"),
                                               "Visible": st.column_config.CheckboxColumn(default=True)})
        st.session_state[saved_key] = levels
    if bounds is None:
        return
    frame = rolling_statistics(close, window)
    visible = frame.loc[bounds[0]:bounds[1]]
    if visible.empty:
        st.info("No closes in the selected interval.")
        return
    row = visible.iloc[-1]

    def fmt(value, spec=".2f", unit=""):
        return f"{value:{spec}}{unit}" if pd.notna(value) else "—"

    metrics = [
        ("Last", fmt(row["Close"], unit=" pts")),
        ("Change", f"{fmt(row['Change'], '+.2f', ' pts')} / {fmt(row['Change %'], '+.2f', '%')}"),
        ("Rolling mean", fmt(row["Mean"], unit=" pts")), ("Std dev", fmt(row["Std"], unit=" pts")),
        ("Z-score", fmt(row["Z"], "+.2f")), ("Percentile", fmt(row["Percentile"], ".1f", "%")),
        ("Rolling min", fmt(row["Min"], unit=" pts")), ("Rolling max", fmt(row["Max"], unit=" pts")),
    ]
    for start in (0, 4):
        for column, (label, value) in zip(st.columns(4), metrics[start:start+4]):
            with column:
                metric(label, value, "yellow" if index == "VIX" else "blue")
    st.caption(f"KPI as of {visible.index[-1]:%Y-%m-%d} · {window} valid closes")
    show_chart(index_chart(index, frame, bounds, mean, bands,
                           levels.sort_values("Level").to_dict("records") if show_levels else [], slider),
               f"{index}_daily")
    if pd.isna(row["Mean"]):
        st.caption("Insufficient history for the selected rolling window at this date.")
    else:
        gap = row["Close"] - row["Mean"]
        context = f"Z-score {row['Z']:+.2f}" if pd.notna(row["Z"]) else "Z-score unavailable (zero dispersion)"
        st.caption(f"{index} is {gap:+.2f} points relative to its {window}-close mean. "
                   f"{context} · trailing empirical percentile {row['Percentile']:.1f}%.")


def show_comparison(series):
    st.subheader("VIX vs VSTOXX — Cross-Market Comparison")
    controls = st.columns(3)
    with controls[0]:
        common, _, _ = compare_markets(series["VIX"], series["VSTOXX"], 21)
        if common.empty:
            st.info("The two sources have no common daily closes.")
            return
        bounds = period_controls(common.index, "vix_comparison")
    mode = controls[1].selectbox("Normalization", ["Base 100", "Rolling Z-score"])
    window = controls[2].selectbox("Common closes per rolling window", [21, 63, 126, 252], index=1)
    if bounds is None:
        return
    common, corr, spread = compare_markets(series["VIX"], series["VSTOXX"], window)
    visible = common.loc[bounds[0]:bounds[1]]
    if len(visible) < 2:
        st.info("Select an interval with at least two common closes.")
        return
    normalized = (visible / visible.iloc[0] * 100 if mode == "Base 100" else
                  pd.DataFrame({key: rolling_statistics(common[key], window)["Z"] for key in common})
                  .reindex(visible.index))
    fig = go.Figure()
    for key in normalized:
        fig.add_scatter(x=normalized.index, y=normalized[key], name=key, mode="lines",
                        line=dict(color=COLORS[key]))
    show_chart(terminal_chart(fig, *bounds, mode, selector=False), "vix_comparison_chart")
    for label, values, unit in (("Rolling correlation", corr, "Pearson r"),
                                 ("VSTOXX − VIX", spread, "Index points")):
        values = values.reindex(visible.index)
        fig = go.Figure(go.Scatter(x=values.index, y=values, name=label, mode="lines"))
        show_chart(terminal_chart(fig, *bounds, unit, selector=False), "vix_" + label)
    st.caption(f"{len(visible):,} common closes, {visible.index[0]:%Y-%m-%d} to {visible.index[-1]:%Y-%m-%d}. "
               "The indices cover different equity markets and close at different times. Correlation is descriptive.")
    st.dataframe(visible.agg(["count", "mean", "std", "min", "median", "max"]).round(2), width="stretch")


def render_vix(metric):
    st.header("VIX | VOLATILITY & MARKET SENTIMENT")
    st.caption("Implied Volatility Monitoring — US & European Equity Markets")
    layout = st.radio("Chart layout", ["Full width", "Side by side"], index=1, horizontal=True)
    panels = st.columns(2) if layout == "Side by side" else [st.container(), st.container()]
    series = {}
    for panel, (key, title) in zip(panels, [("VIX", "US Market — CBOE VIX"), ("VSTOXX", "European Market — VSTOXX")]):
        with panel:
            st.subheader(title)
            try:
                with st.spinner(f"Loading {key} daily history…"):
                    close = load_history(key)
            except ValueError as exc:
                st.warning(str(exc))
                continue
            series[key] = close
            st.caption(f"{SOURCES[key][0]} · Daily historical closes · Latest available {close.index[-1]:%Y-%m-%d}")
            show_index(key, close, metric)
    if len(series) == 2:
        show_comparison(series)
    else:
        st.info("Cross-market comparison requires both daily histories.")
    st.caption("Rolling statistics use complete trailing windows of valid closes. Percentile is the share "
               "of closes less than or equal to the current close, including that close. σ uses N−1. "
               "The bands describe dispersion and carry no Gaussian probability interpretation.")
