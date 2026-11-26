"""Distribution analysis for the selected backtest strategy."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm, t

from src.return_distribution import (
    GAUSSIAN, HISTORICAL, LOGNORMAL, MODELS, STUDENT, backtest_return_samples, distribution_statistics,
)


cached_statistics = st.cache_data(show_spinner=False, max_entries=96)(distribution_statistics)
cached_samples = st.cache_data(show_spinner=False, max_entries=24)(backtest_return_samples)


COLORS = {GAUSSIAN: "#cbd5e1", STUDENT: "#b699ff", LOGNORMAL: "#63d6ba"}
ASSUMPTIONS = {
    GAUSSIAN: "Symmetric, thin tails",
    STUDENT: "Symmetric, heavy tails",
    LOGNORMAL: "Asymmetric simple returns above −100%",
}


def distribution_chart(sample, stats, method, confidence, visible=(), tail_zoom=False):
    models = stats["models"]
    var, es = models.get(method, {}).get("risk", {}).get(confidence, (np.nan, np.nan))
    mean = stats["Mean return"]
    bins = int(np.clip(np.sqrt(len(sample)), 10, 80))
    if stats["Volatility"] == 0:
        span = max(abs(mean) * .01, .0001)
        bins = [mean - span, mean + span]
    density, edges = np.histogram(sample, bins=bins, density=True)
    fig = go.Figure(go.Bar(
        x=edges[:-1] + np.diff(edges) / 2, y=density / 100, width=np.diff(edges),
        name="Observed returns", marker_color="#70a9eb", opacity=.65,
        hovertemplate="Bin center %{x:.2%}<br>Density %{y:.3f}<extra></extra>",
    ))
    markers = (("Selected VaR", -var, "#ffab67", "solid"),
               ("Selected ES", -es, "#ff6f78", "dash"),
               ("Mean return", mean, "#8ddbaf", "dashdot"))
    observed_left, observed_right = edges[0], edges[-1]
    if tail_zoom:
        left, right = sample.quantile([.005, .20])
        comparison_var = [-row["risk"][confidence][0] for row in models.values()]
        left = min(left, *comparison_var)
        if left == right:
            left, right = observed_left, observed_right
    else:
        marks = [x for _, x, _, _ in markers if np.isfinite(x)]
        left, right = min([observed_left, *marks]), max([observed_right, *marks])
    padding = .04 * (right - left)
    left, right = left - padding, right + padding
    for family in visible:
        if family not in models:
            continue
        fit = models[family]["fit"]
        probabilities = np.linspace(.0005, .9995, 300)
        if family == STUDENT:
            central = t.ppf(probabilities, fit.df, loc=fit.location, scale=fit.scale)
        else:
            central = norm.ppf(probabilities, loc=fit.location, scale=fit.scale)
            if family == LOGNORMAL:
                with np.errstate(over="ignore"):
                    central = np.expm1(central)
        grid = np.union1d(np.linspace(left, right, 320), central[(central > left) & (central < right)])
        with np.errstate(over="ignore"):
            pdf = np.exp(fit.log_density(grid) - np.log(100))
        fig.add_scatter(x=grid, y=np.where(np.isfinite(pdf), pdf, np.nan), mode="lines",
                        name=family, line=dict(color=COLORS[family]),
                        hovertemplate=f"%{{x:.2%}}<br>Density %{{y:.3f}}<extra>{ASSUMPTIONS[family]}</extra>")
    if np.isfinite(var):
        fig.add_vrect(x0=left, x1=-var, fillcolor="#fe6269", opacity=.16, line_width=0, layer="below")
    for label, location, color, dash in markers:
        if np.isfinite(location):
            fig.add_scatter(x=[location, location], y=[0, 1], yaxis="y2",
                            mode="lines", name=label, line=dict(color=color, dash=dash),
                            hovertemplate="%{x:.2%}<extra>%{fullData.name}</extra>")
    fig.update_layout(
        template="plotly_dark", xaxis=dict(title="Daily return", tickformat=".1%", range=[left, right]),
        yaxis_title="Density per percentage point",
        yaxis2=dict(overlaying="y", range=[0, 1], visible=False, fixedrange=True),
        legend=dict(orientation="h", y=-.23), bargap=0,
        margin=dict(l=12, r=12, b=50, t=12), height=365,
    )
    return fig


def _sample_description(sample):
    return f"{sample.index[0]:%d %b %Y} – {sample.index[-1]:%d %b %Y} · {len(sample):,} daily observations"


def _risk_text(value):
    if pd.isna(value):
        return "—"
    return "+∞" if np.isposinf(value) else f"{value:.2%}"


def model_comparison(stats):
    rows = []
    for family in MODELS:
        model = stats["models"].get(family, {})
        row = {}
        for confidence in (.95, .99):
            var, es = model.get("risk", {}).get(confidence, (np.nan, np.nan))
            row.update({f"VaR {confidence:.0%}": var, f"ES {confidence:.0%}": es})
        row.update({key: model.get(key, np.nan) for key in ("Log-likelihood", "AIC", "BIC")})
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(MODELS, name="Distribution"))


def _diagnostics(sample, stats):
    with st.expander("Distribution Model Diagnostics"):
        measures = {"Sample mean": f"{stats['Mean return']:.2%}", "Sample volatility": f"{stats['Volatility']:.2%}"}
        measures.update({k: f"{stats[k]:.4g}" if pd.notna(stats[k]) else "—"
                         for k in ("Skewness", "Excess kurtosis", "JB statistic", "JB p-value")})
        st.caption(_sample_description(sample))
        st.dataframe(pd.DataFrame([measures]), hide_index=True, width="stretch")
        parameters = []
        for family in MODELS[1:]:
            if family in stats["models"]:
                fit = stats["models"][family]["fit"]
                variance = None
                if family == STUDENT:
                    variance = fit.scale**2 * fit.df / (fit.df - 2) if fit.df > 2 else np.inf
                parameters.append({"Distribution": family, "Location": fit.location, "Scale": fit.scale,
                                   "Degrees of freedom": fit.df, "Student-t variance": variance})
        if parameters:
            st.dataframe(pd.DataFrame(parameters).style.format(precision=5, na_rep="—"),
                         hide_index=True, width="stretch")
        st.caption("The parameter table reports Student-t scale separately from standard deviation. "
                   "Shifted lognormal location and scale apply to log gross returns.")
        st.caption("Gaussian assumes symmetric, thin tails. Student-t allows symmetric heavy tails. "
                   "Shifted lognormal produces asymmetric simple returns. Historical simulation uses observed returns. "
                   "Gaussian and Student-t have unbounded support and may imply losses above 100%.")
        st.caption("Likelihood, AIC and BIC use the same finite observations and simple-return density. "
                   "The shifted lognormal likelihood includes the change-of-variable Jacobian. "
                   "AIC and BIC measure relative fit and do not validate VaR forecasts.")
        st.caption("Historical VaR uses the inverse empirical loss CDF. Historical ES averages the worst "
                   "tail probability mass, with fractional weighting at its boundary. "
                   "Jarque–Bera results use the asymptotic approximation for nonconstant samples above 2,000 observations.")


def _render_model_analysis(sample, stats, method, confidence, visible, tail_zoom, key):
    st.caption(_sample_description(sample))
    st.plotly_chart(distribution_chart(sample, stats, method, confidence, visible, tail_zoom),
                    key=f"dist_chart_{key}", width="stretch")
    comparison = model_comparison(stats)
    risk_columns = [column for column in comparison if column.startswith(("VaR", "ES"))]
    st.dataframe(comparison.style.format(dict.fromkeys(risk_columns, _risk_text), precision=2, na_rep="—"),
                 width="stretch")
    if stats["Excluded"]:
        st.caption(f"{stats['Excluded']:,} missing or nonfinite returns were excluded from the common sample.")
    for family, reason in stats["unavailable"].items():
        st.caption(f"{family} unavailable · {reason}")
    student = stats["models"].get(STUDENT)
    if student:
        df = student["fit"].df
        if df <= 1:
            st.caption("Student-t theoretical ES is infinite because degrees of freedom are at most 1.")
        if df <= 2:
            st.caption("Student-t theoretical variance is not finite because degrees of freedom are at most 2.")
        if df >= 1e4:
            st.caption("Student-t degrees of freedom are weakly identified near the Gaussian limit.")
    mass = stats["Observations"] * .01
    if mass < 5:
        st.warning(f"The 99% historical tail has {mass:.2f} effective observations. "
                   "Tail estimates and fitted shape parameters can be unstable in small samples.")
    _diagnostics(sample, stats)


def render_distributions(curve, prices, weights, strategy, key, company_name):
    st.subheader("Distribution Analysis")
    st.caption("Ex-post one-day risk · Signed losses. Markers show the corresponding values on the return axis.")
    controls = st.columns(3)
    method = controls[0].selectbox("Risk markers", MODELS, key=f"dist_method_{key}")
    confidence = controls[1].selectbox("Confidence", [.95, .99], format_func="{:.0%}".format,
                                       key=f"dist_confidence_{key}")
    tail_zoom = controls[2].checkbox("Left-tail zoom", key=f"dist_zoom_{key}")
    visibility = st.columns(3)
    visible = [family for column, family in zip(visibility, MODELS[1:])
               if column.checkbox(f"Show {family} density", value=True, key=f"dist_visible_{key}_{family}",
                                  help=ASSUMPTIONS[family])]
    portfolio, assets = cached_samples(curve, prices, weights)
    start, end = curve.index.min().date(), curve.index.max().date()
    period = st.date_input("Evaluation period", (start, end), min_value=start, max_value=end,
                           key=f"dist_period_{key}_{strategy}")
    if len(period) != 2:
        st.info("Select both dates.")
        return
    start, end = map(pd.Timestamp, period)
    portfolio, assets = portfolio.loc[start:end], assets.loc[start:end]
    portfolio_tab, asset_tab = st.tabs(["Portfolio Distribution", "Asset Distribution"],
                                       key=f"dist_tab_{key}", on_change="rerun")
    if portfolio_tab.open:
        with portfolio_tab:
            sample, stats = cached_statistics(portfolio)
            if stats is None:
                st.info("At least two finite, recorded daily portfolio returns are required.")
            else:
                st.caption(f"{strategy} · Recorded backtest returns")
                var, es = stats["models"].get(method, {}).get("risk", {}).get(confidence, (np.nan, np.nan))
                metrics = [
                    ("Mean daily return", _risk_text(stats["Mean return"])),
                    ("Daily volatility", _risk_text(stats["Volatility"])),
                    (f"VaR {confidence:.0%}", _risk_text(var)),
                    ("Expected Shortfall", _risk_text(es)),
                ]
                for col, (label, value) in zip(st.columns(4), metrics):
                    col.metric(label, value)
                _render_model_analysis(sample, stats, method, confidence, visible, tail_zoom, f"{key}_portfolio")
    if asset_tab.open:
        with asset_tab:
            st.caption("Standalone Asset VaR uses daily returns from the adjusted backtest prices in their original "
                       "currency basis. Component VaR uses portfolio exposures and covariance.")
            if assets.columns.empty:
                st.info("No asset prices are available for this backtest.")
                return
            labels = {ticker: f"{company_name(ticker)} ({ticker})" for ticker in assets.columns}
            selected = st.multiselect("Assets", list(assets.columns), default=list(assets.columns[:1]),
                                      format_func=labels.__getitem__, max_selections=3,
                                      key=f"dist_assets_{key}_{strategy}")
            comparison = []
            for ticker in selected:
                sample, stats = cached_statistics(assets[ticker])
                if stats is None:
                    st.info(f"{labels[ticker]} has fewer than two finite daily returns in this period.")
                    continue
                st.markdown(f"**{labels[ticker]}**")
                _render_model_analysis(sample, stats, method, confidence, visible, tail_zoom, f"{key}_{ticker}")
                var, es = stats["models"].get(method, {}).get("risk", {}).get(confidence, (np.nan, np.nan))
                comparison.append({"Asset": labels[ticker], "Mean return": stats["Mean return"],
                                   "Volatility": stats["Volatility"], "VaR": var, "ES": es,
                                   "Skewness": stats["Skewness"], "Excess kurtosis": stats["Excess kurtosis"],
                                   "Observations": stats["Observations"]})
            if comparison:
                st.caption(f"Standalone asset comparison · {method} · {confidence:.0%}")
                formats = dict.fromkeys(("Mean return", "Volatility", "VaR", "ES"), _risk_text)
                st.dataframe(pd.DataFrame(comparison).style.format(formats, precision=3, na_rep="—"),
                             hide_index=True, width="stretch")
