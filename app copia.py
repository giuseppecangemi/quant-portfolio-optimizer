from pathlib import Path
import yfinance as yf

YFINANCE_CACHE = (
    Path.home()
    / ".cache"
    / "quant-portfolio-optimizer"
    / "yfinance"
)

YFINANCE_CACHE.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime
from pathlib import Path

from src.ff3_attribution import compute_ff3_attribution
from src.fama_french_5 import load_europe_ff5, fit_ff5
from src.data import get_prices
from src.fama_french import (
    load_europe_ff3, fit_ff3, convert_eur_prices_to_usd,
    estimate_ff3_eur_returns,
)
from src.factors import calculate_capm, capm_expected_returns
from src.returns import calculate_returns, annualized_returns
from src.risk import (
    annualized_volatility,
    covariance_matrix,
    portfolio_return,
    portfolio_volatility,
)
from src.optimization import (
    optimize_minimum_volatility,
    optimize_maximum_sharpe,
    optimize_risk_parity,
    optimize_hrp,
    hrp_cluster_order,
    risk_contributions,
    efficient_frontier,
)
from src.visualization import plot_efficient_frontier
from src.screening import (
    screen_universe,
    screen_prices_point_in_time,
    build_price_matrix,
)
from src.backtest import (
    backtest_portfolio,
    calculate_backtest_metrics,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Quant Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
)

import sys

st.sidebar.caption(f"Python: {sys.executable}")
st.sidebar.caption(f"yfinance: {yf.__version__}")

# =========================================================
# DATA
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
INSTRUMENTS_FILE = PROJECT_DIR / "data" / "instruments.csv"
EXPERIMENTS_DIR = PROJECT_DIR / "saved_experiments"

instruments = pd.read_csv(INSTRUMENTS_FILE)

FTSE_MIB_TICKER = "FTSEMIB.MI"


# =========================================================
# HELPERS
# =========================================================


def _json_safe(value):
    """
    Converte gli oggetti usati dall'app in una struttura JSON serializzabile.
    Mantiene esplicitamente DataFrame e Series così che ogni esperimento
    conservi anche tabelle, curve, matrici e storici dei pesi.

    NaN, +inf e -inf vengono convertiti in None perché non sono valori
    JSON validi quando allow_nan=False.
    """
    if value is None:
        return None

    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "data": value.tolist(),
        }

    if isinstance(value, pd.Series):
        return {
            "__type__": "series",
            "name": _json_safe(value.name),
            "index": [_json_safe(x) for x in value.index.tolist()],
            "data": [_json_safe(x) for x in value.tolist()],
        }

    if isinstance(value, pd.DataFrame):
        return {
            "__type__": "dataframe",
            "index": [_json_safe(x) for x in value.index.tolist()],
            "columns": [_json_safe(x) for x in value.columns.tolist()],
            "data": [
                [_json_safe(cell) for cell in row]
                for row in value.to_numpy(dtype=object).tolist()
            ],
        }

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if pd.isna(value):
        return None

    return str(value)


def _experiment_id():
    return datetime.now().strftime("EXP-%Y%m%d-%H%M%S-%f")


def build_experiment_snapshot(
    experiment_name,
    market,
    selection_mode,
    tickers,
    risk_free_rate,
    capital,
    use_fractional_shares,
    use_max_weight,
    max_weight,
):
    """
    Crea una fotografia immutabile del run corrente.

    Il file contiene i cinque blocchi richiesti:
    Asset Selection, Portfolio Parameters, Portfolio Optimization,
    Factor Models e Backtesting.
    """
    now = datetime.now()
    experiment_id = _experiment_id()

    screening_state = st.session_state.get("screening_results")
    optimization_state = st.session_state.get("optimization_results")
    factor_state = st.session_state.get("factor_model_results")
    ff3_state = st.session_state.get("ff3_results")
    ff5_state = st.session_state.get("ff5_results")
    standard_state = st.session_state.get("standard_backtest_results")
    fixed_state = st.session_state.get("fixed_backtest_results")

    selected_assets = []
    for ticker in tickers:
        match = instruments[instruments["ticker"] == ticker]
        selected_assets.append(
            {
                "ticker": ticker,
                "name": (
                    match.iloc[0]["name"]
                    if not match.empty
                    else ticker
                ),
            }
        )

    snapshot = {
        "experiment": {
            "id": experiment_id,
            "name": (
                experiment_name.strip()
                if experiment_name.strip()
                else experiment_id
            ),
            "saved_at": now.isoformat(),
            "app": "Quant Portfolio Optimizer",
            "schema_version": 1,
        },

        "asset_selection": {
            "market": market,
            "selection_mode": selection_mode,
            "selected_assets": selected_assets,
            "selected_tickers": list(tickers),
            "screening_results": screening_state,
        },

        "portfolio_parameters": {
            "market": market,
            "selection_mode": selection_mode,
            "risk_free_rate": risk_free_rate,
            "capital": capital,
            "use_fractional_shares": use_fractional_shares,
            "use_max_weight": use_max_weight,
            "max_weight": max_weight,
            "number_of_selected_assets": len(tickers),
        },

        "portfolio_optimization": optimization_state,

        "factor_models": factor_state,
        "fama_french_3": ff3_state,
        "fama_french_5": ff5_state,

        "backtesting": {
            "standard_walk_forward": standard_state,
            "fixed_horizon": fixed_state,
        },
    }

    return _json_safe(snapshot)


def save_experiment_snapshot(snapshot):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    experiment = snapshot["experiment"]
    experiment_id = experiment["id"]
    file_path = EXPERIMENTS_DIR / f"{experiment_id}.json"

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(
            snapshot,
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    return file_path


def list_saved_experiments():
    if not EXPERIMENTS_DIR.exists():
        return []

    experiments = []

    for file_path in sorted(
        EXPERIMENTS_DIR.glob("EXP-*.json"),
        reverse=True,
    ):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            meta = payload.get("experiment", {})
            params = payload.get("portfolio_parameters", {})
            asset_selection = payload.get("asset_selection", {})
            backtesting = payload.get("backtesting", {})

            standard = backtesting.get("standard_walk_forward")
            fixed = backtesting.get("fixed_horizon")

            experiments.append(
                {
                    "ID": meta.get("id", file_path.stem),
                    "Name": meta.get("name", file_path.stem),
                    "Saved At": meta.get("saved_at", ""),
                    "Market": params.get("market", ""),
                    "Selection": params.get("selection_mode", ""),
                    "Assets": params.get(
                        "number_of_selected_assets",
                        len(asset_selection.get("selected_tickers", [])),
                    ),
                    "Optimization": (
                        "Saved"
                        if payload.get("portfolio_optimization") is not None
                        else "—"
                    ),
                    "Factor Models": (
                        "Saved"
                        if payload.get("factor_models") is not None
                        else "—"
                    ),
                    "Standard BT": (
                        "Saved"
                        if standard is not None
                        else "—"
                    ),
                    "Fixed BT": (
                        "Saved"
                        if fixed is not None
                        else "—"
                    ),
                    "_path": str(file_path),
                }
            )

        except Exception:
            continue

    return experiments



def _json_restore(value):
    """Ricostruisce DataFrame, Series e ndarray salvati da _json_safe()."""
    if isinstance(value, list):
        return [_json_restore(item) for item in value]

    if not isinstance(value, dict):
        return value

    value_type = value.get("__type__")

    if value_type == "ndarray":
        return np.array(value.get("data", []))

    if value_type == "series":
        index = [_json_restore(x) for x in value.get("index", [])]
        data = [_json_restore(x) for x in value.get("data", [])]
        return pd.Series(
            data,
            index=index,
            name=_json_restore(value.get("name")),
        )

    if value_type == "dataframe":
        index = [_json_restore(x) for x in value.get("index", [])]
        columns = [_json_restore(x) for x in value.get("columns", [])]
        data = [
            [_json_restore(cell) for cell in row]
            for row in value.get("data", [])
        ]
        return pd.DataFrame(data, index=index, columns=columns)

    return {
        key: _json_restore(item)
        for key, item in value.items()
    }


def load_saved_experiment(file_path):
    with Path(file_path).open("r", encoding="utf-8") as f:
        return _json_restore(json.load(f))


def _saved_pct(value):
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "—"


def _saved_num(value, digits=2):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _saved_money(value):
    try:
        return f"€{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _saved_table(df, percentage_columns=None, money_columns=None):
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No saved table is available for this section.")
        return

    fmt = {}
    for column in percentage_columns or []:
        if column in df.columns:
            fmt[column] = "{:.2%}"
    for column in money_columns or []:
        if column in df.columns:
            fmt[column] = "€{:,.2f}"

    if fmt:
        st.dataframe(df.style.format(fmt, na_rep="—"), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)


def render_ff3_attribution(result, key_suffix=""):
    attribution = result.get("ff3_performance_attribution")
    if not isinstance(attribution, dict):
        return
    summary = attribution.get("summary")
    cumulative = attribution.get("cumulative")
    if not isinstance(summary, pd.Series) or not isinstance(cumulative, pd.DataFrame) or cumulative.empty:
        return
    st.subheader("FF3 Performance Attribution")
    covered = attribution.get("covered_days", 0)
    total = attribution.get("total_days", 0)
    st.caption(
        "Ex-post additive contributions in EUR, linked to initial capital. "
        "Factor loadings are frozen between rebalances; 'Unexplained' is the "
        "realized residual, not regression alpha. 'FX translation' reconciles "
        "USD-denominated factors to EUR portfolio returns. "
        f"Coverage: {covered}/{total} trading days."
    )
    if covered != total:
        st.warning("Incomplete factor/FX coverage: attribution totals cover only matching dates, "
                   "Unattributed days are reported separately at their actual EUR P&L; "
                   "missing factor returns are not interpolated.")
    if "total_return" in attribution:
        total_return = float(attribution["total_return"])
        difference = float(attribution.get("reconciliation_difference", 0.0))
        cols = st.columns(3)
        cols[0].metric("FF3 Total Return", f"{total_return*100:+.2f}%")
        cols[1].metric("Attribution Sum", f"{float(summary.sum())*100:+.2f} pp")
        cols[2].metric("Reconciliation Difference", f"{difference*100:+.6f} pp")
    table = summary.to_frame("Contribution (pp)") * 100
    st.dataframe(table.style.format({"Contribution (pp)": "{:+.2f}"}),
                 use_container_width=True)
    fig = go.Figure()
    for component in cumulative.columns:
        fig.add_trace(go.Scatter(x=cumulative.index,
                                 y=cumulative[component]*100,
                                 mode="lines", name=component))
    fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative contribution (pp)",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True,
                    key=f"ff3_attr_{key_suffix}")


def _render_saved_backtest_summary(result, title):
    if not isinstance(result, dict):
        st.info(f"No saved {title} results.")
        return

    st.markdown(f"#### {title}")

    meta_cols = st.columns(4)
    meta_cols[0].metric(
        "Estimation Window",
        result.get("lookback_label", "—"),
    )
    meta_cols[1].metric(
        "Backtest Period",
        result.get("period_label", "—"),
    )
    meta_cols[2].metric(
        "Rebalancing",
        result.get("rebalance_label", "—"),
    )
    meta_cols[3].metric(
        "Initial Capital",
        _saved_money(result.get("capital")),
    )

    comparison = {}
    metric_map = [
        ("Maximum Sharpe", "max_bt_metrics"),
        ("CAPM Maximum Sharpe", "capm_bt_metrics"),
        ("FF3 Maximum Sharpe", "ff3_bt_metrics"),
        ("Minimum Volatility", "min_bt_metrics"),
        ("Risk Parity", "risk_parity_bt_metrics"),
        ("HRP", "hrp_bt_metrics"),
        ("Equal Weight", "equal_bt_metrics"),
        ("FTSE MIB", "ftse_bt_metrics"),
    ]

    for label, key in metric_map:
        metrics = result.get(key)
        if isinstance(metrics, dict):
            comparison[label] = metrics

    if comparison:
        comparison_df = pd.DataFrame(comparison).T

        formatters = {}
        for column in comparison_df.columns:
            if (
                "Return" in str(column)
                or "Volatility" in str(column)
                or "Drawdown" in str(column)
                or str(column) == "CAGR"
            ):
                formatters[column] = "{:.2%}"
            elif "Sharpe" in str(column) or "Sortino" in str(column):
                formatters[column] = "{:.2f}"
            elif str(column) == "Final Value":
                formatters[column] = "€{:,.2f}"

        st.dataframe(
            comparison_df.style.format(formatters, na_rep="—"),
            use_container_width=True,
        )

    growth_map = [
        ("Maximum Sharpe", "max_sharpe_bt"),
        ("CAPM Maximum Sharpe", "capm_bt"),
        ("FF3 Maximum Sharpe", "ff3_bt"),
        ("Minimum Volatility", "min_vol_bt"),
        ("Risk Parity", "risk_parity_bt"),
        ("HRP", "hrp_bt"),
        ("Equal Weight", "equal_weight_bt"),
        ("FTSE MIB", "ftse_mib_bt"),
    ]

    growth = pd.DataFrame()

    for label, key in growth_map:
        backtest_data = result.get(key)

        # I motori di backtest salvano le curve come DataFrame
        # con la colonna "Portfolio Value". Supportiamo anche
        # Series per compatibilità con eventuali esperimenti legacy.
        if isinstance(backtest_data, pd.DataFrame) and not backtest_data.empty:
            if "Portfolio Value" in backtest_data.columns:
                s = backtest_data["Portfolio Value"].copy()
            elif backtest_data.shape[1] == 1:
                s = backtest_data.iloc[:, 0].copy()
            else:
                continue

        elif isinstance(backtest_data, pd.Series) and not backtest_data.empty:
            s = backtest_data.copy()

        else:
            continue

        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass

        growth[label] = s

    render_ff3_attribution(result, key_suffix=f"saved_{title}")

    ff3_exposure_saved = result.get("ff3_factor_exposure_history")
    if isinstance(ff3_exposure_saved, pd.DataFrame) and not ff3_exposure_saved.empty:
        st.markdown("##### FF3 Factor Exposure History")
        st.dataframe(ff3_exposure_saved.style.format({
            c: "{:.3f}" for c in ("Market Beta", "SMB Exposure", "HML Exposure")
            if c in ff3_exposure_saved.columns
        }), use_container_width=True)

        # Only plot exposure history when there are at least two rebalance dates.
        if len(ff3_exposure_saved) > 1:
            exposure_plot = ff3_exposure_saved.copy()
            exposure_plot.index = pd.to_datetime(exposure_plot.index)
            exposure_fig = go.Figure()
            for factor in ("Market Beta", "SMB Exposure", "HML Exposure"):
                if factor in exposure_plot.columns:
                    exposure_fig.add_trace(go.Scatter(
                        x=exposure_plot.index,
                        y=pd.to_numeric(exposure_plot[factor], errors="coerce"),
                        mode="lines+markers",
                        name=factor,
                    ))
            exposure_fig.update_layout(
                xaxis_title="Rebalance Date",
                yaxis_title="Factor Loading",
                hovermode="x unified",
            )
            st.plotly_chart(exposure_fig, use_container_width=True,
                            key=f"saved_ff3_exposure_{title}")

    if not growth.empty:
        st.markdown("##### Portfolio Growth")

        # Ricostruisce il solo grafico Growth del backtest salvato.
        # I dati sono già persistiti nello snapshot JSON come equity curves,
        # quindi il grafico resta disponibile anche riaprendo l'esperimento.
        growth_fig = go.Figure()

        for column in growth.columns:
            growth_fig.add_trace(
                go.Scatter(
                    x=growth.index,
                    y=growth[column],
                    mode="lines",
                    name=column,
                )
            )

        initial_capital = result.get("capital")
        if initial_capital is not None:
            growth_fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_width=1,
                line_color="rgba(255, 255, 255, 0.45)",
                annotation_text=(
                    f"Initial Capital · €{float(initial_capital):,.0f}"
                ),
                annotation_position="top left",
            )

        growth_fig.update_layout(
            title="Portfolio Growth",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (€)",
            hovermode="x unified",
            legend_title_text="Strategy",
        )

        st.plotly_chart(
            growth_fig,
            use_container_width=True,
            key=f"saved_growth_{title}",
        )


def render_saved_experiment(payload):
    meta = payload.get("experiment", {})
    selection = payload.get("asset_selection", {})
    params = payload.get("portfolio_parameters", {})
    optimization = payload.get("portfolio_optimization")
    factors = payload.get("factor_models")
    ff3_saved = payload.get("fama_french_3")
    ff5_saved = payload.get("fama_french_5")
    backtesting = payload.get("backtesting", {})

    st.subheader(meta.get("name", meta.get("id", "Saved Experiment")))
    st.caption(
        f"{meta.get('id', '')} · Saved "
        f"{str(meta.get('saved_at', ''))[:19].replace('T', ' ')}"
    )

    # =========================================================
    # ASSET SELECTION
    # =========================================================
    with st.expander("Asset Selection", expanded=True):
        cols = st.columns(4)
        cols[0].metric("Market", selection.get("market", "—"))
        cols[1].metric("Mode", selection.get("selection_mode", "—"))
        cols[2].metric(
            "Selected Assets",
            len(selection.get("selected_tickers", [])),
        )

        screening = selection.get("screening_results")
        top_n = screening.get("top_n") if isinstance(screening, dict) else None
        cols[3].metric("Top N", top_n if top_n is not None else "—")

        assets = selection.get("selected_assets", [])
        if assets:
            asset_df = pd.DataFrame(assets)
            asset_df.columns = [
                "Ticker" if c == "ticker" else
                "Company" if c == "name" else c
                for c in asset_df.columns
            ]
            st.markdown("##### Selected Assets")
            st.dataframe(asset_df, use_container_width=True, hide_index=True)

        if isinstance(screening, dict):
            screening_table = screening.get("table")
            if isinstance(screening_table, pd.DataFrame):
                st.markdown("##### Screening Ranking")
                st.dataframe(screening_table, use_container_width=True)

    # =========================================================
    # PORTFOLIO PARAMETERS
    # =========================================================
    with st.expander("Portfolio Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital", _saved_money(params.get("capital")))
        c2.metric(
            "Risk-Free Rate",
            _saved_pct(params.get("risk_free_rate")),
        )
        c3.metric(
            "Max Weight",
            _saved_pct(params.get("max_weight"))
            if params.get("use_max_weight")
            else "No cap",
        )
        c4.metric(
            "Fractional Shares",
            "Yes" if params.get("use_fractional_shares") else "No",
        )

    # =========================================================
    # PORTFOLIO OPTIMIZATION
    # =========================================================
    with st.expander("Portfolio Optimization", expanded=True):
        if not isinstance(optimization, dict):
            st.info("Portfolio Optimization had not been run when this experiment was saved.")
        else:
            st.caption(
                f"Historical period: {optimization.get('period', '—')}"
            )

            summary = pd.DataFrame(
                {
                    "Minimum Volatility": {
                        "Expected Return": optimization.get("min_return"),
                        "Volatility": optimization.get("min_volatility"),
                        "Sharpe": optimization.get("min_sharpe"),
                    },
                    "Maximum Sharpe": {
                        "Expected Return": optimization.get("max_return"),
                        "Volatility": optimization.get("max_volatility"),
                        "Sharpe": optimization.get("max_sharpe"),
                    },
                    "Risk Parity": {
                        "Expected Return": optimization.get("risk_parity_return"),
                        "Volatility": optimization.get("risk_parity_volatility"),
                        "Sharpe": optimization.get("risk_parity_sharpe"),
                    },
                    "HRP": {
                        "Expected Return": optimization.get("hrp_return"),
                        "Volatility": optimization.get("hrp_volatility"),
                        "Sharpe": optimization.get("hrp_sharpe"),
                    },
                }
            ).T

            st.markdown("##### Strategy Summary")
            st.dataframe(
                summary.style.format(
                    {
                        "Expected Return": "{:.2%}",
                        "Volatility": "{:.2%}",
                        "Sharpe": "{:.2f}",
                    },
                    na_rep="—",
                ),
                use_container_width=True,
            )

            allocation_tabs = st.tabs(
                [
                    "Maximum Sharpe",
                    "Minimum Volatility",
                    "Risk Parity",
                    "HRP",
                ]
            )
            allocation_keys = [
                "max_table",
                "min_table",
                "risk_parity_table",
                "hrp_table",
            ]

            for tab, key in zip(allocation_tabs, allocation_keys):
                with tab:
                    table = optimization.get(key)
                    if isinstance(table, pd.DataFrame):
                        st.dataframe(table, use_container_width=True)
                    else:
                        st.info("No saved allocation table.")

    # =========================================================
    # FACTOR MODELS
    # =========================================================
    with st.expander("Factor Models", expanded=True):
        if not isinstance(factors, dict):
            st.info("Factor Models had not been run when this experiment was saved.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", "CAPM")
            c2.metric(
                "Expected Return",
                _saved_pct(factors.get("return")),
            )
            c3.metric(
                "Volatility",
                _saved_pct(factors.get("volatility")),
            )
            c4.metric(
                "Sharpe",
                _saved_num(factors.get("sharpe")),
            )

            capm_table = factors.get("table")
            if isinstance(capm_table, pd.DataFrame):
                st.markdown("##### CAPM Maximum Sharpe Allocation")
                st.dataframe(capm_table, use_container_width=True)

            stats = factors.get("stats")
            if isinstance(stats, pd.DataFrame):
                st.markdown("##### CAPM Asset Statistics")
                st.dataframe(
                    stats.style.format(
                        {
                            "Beta": "{:.2f}",
                            "Alpha": "{:.2%}",
                            "R Squared": "{:.2f}",
                            "Market Correlation": "{:.2f}",
                            "Historical Return": "{:.2%}",
                            "CAPM Expected Return": "{:.2%}",
                        },
                        na_rep="—",
                    ),
                    use_container_width=True,
                )

        # FF3 is saved separately from the CAPM results for backward compatibility.
        if isinstance(ff3_saved, dict):
            st.markdown("#### FF3 Maximum Sharpe Portfolio")
            st.caption(f"Historical estimation period: {ff3_saved.get('period', '—')}")
            ff3_metrics = st.columns(3)
            ff3_metrics[0].metric("Expected Return", _saved_pct(ff3_saved.get("return")))
            ff3_metrics[1].metric("Volatility", _saved_pct(ff3_saved.get("volatility")))
            ff3_metrics[2].metric("Sharpe Ratio", _saved_num(ff3_saved.get("sharpe")))

            ff3_allocation = ff3_saved.get("table")
            if isinstance(ff3_allocation, pd.DataFrame) and not ff3_allocation.empty:
                st.markdown("##### FF3 Portfolio Allocation")
                allocation_format = {
                    "Weight": "{:.2%}", "Target Amount": "€{:,.2f}",
                    "Price": "€{:,.2f}", "Invested": "€{:,.2f}",
                }
                allocation_format = {k: v for k, v in allocation_format.items()
                                     if k in ff3_allocation.columns}
                st.dataframe(ff3_allocation.style.format(allocation_format),
                             use_container_width=True, hide_index=True)

            ff3_stats_saved = ff3_saved.get("stats")
            if isinstance(ff3_stats_saved, pd.DataFrame) and not ff3_stats_saved.empty:
                st.markdown("##### FF3 Asset Statistics")
                st.dataframe(ff3_stats_saved.style.format(precision=3),
                             use_container_width=True)

    # =========================================================
    if isinstance(ff5_saved, dict):
        st.markdown("#### FF5 Maximum Sharpe Portfolio")
        ff5_saved_cols = st.columns(3)
        ff5_saved_cols[0].metric("Expected Return", _saved_pct(ff5_saved.get("return")))
        ff5_saved_cols[1].metric("Volatility", _saved_pct(ff5_saved.get("volatility")))
        ff5_saved_cols[2].metric("Sharpe Ratio", _saved_num(ff5_saved.get("sharpe")))
        ff5_saved_table = ff5_saved.get("table")
        if isinstance(ff5_saved_table, pd.DataFrame) and not ff5_saved_table.empty:
            st.dataframe(ff5_saved_table, use_container_width=True)
        ff5_saved_stats = ff5_saved.get("stats")
        if isinstance(ff5_saved_stats, pd.DataFrame) and not ff5_saved_stats.empty:
            st.dataframe(ff5_saved_stats, use_container_width=True)

    # BACKTESTING
    # =========================================================
    with st.expander("Backtesting", expanded=True):
        standard = backtesting.get("standard_walk_forward")
        fixed = backtesting.get("fixed_horizon")

        if standard is None and fixed is None:
            st.info("Backtesting had not been run when this experiment was saved.")
        else:
            if isinstance(standard, dict):
                # New format: {"fixed": result, "dynamic": result}
                if "max_sharpe_bt" not in standard:
                    for policy, result in standard.items():
                        if isinstance(result, dict):
                            policy_label = result.get(
                                "selection_policy_label",
                                "Fixed at Backtest Start"
                                if policy == "fixed"
                                else "Re-screen at Every Rebalance",
                            )
                            _render_saved_backtest_summary(
                                result,
                                f"Standard Walk-Forward · {policy_label}",
                            )
                else:
                    _render_saved_backtest_summary(
                        standard,
                        "Standard Walk-Forward",
                    )

            if isinstance(fixed, dict):
                st.divider()
                _render_saved_backtest_summary(
                    fixed,
                    "Fixed Horizon",
                )
            if isinstance(standard, dict) and isinstance(fixed, dict):
                render_backtesting_comparison(standard, fixed)


def get_company_name(ticker):
    match = instruments[instruments["ticker"] == ticker]
    if not match.empty:
        return match.iloc[0]["name"]
    return ticker


def colored_metric(label, value, color):

    # =====================================================
    # DO NOT MODIFY THESE COLORS
    # =====================================================

    colors = {
        "green": {
            "bg": "rgba(34, 197, 94, 0.09)",
            "border": "rgba(34, 197, 94, 0.13)",
        },
        "red": {
            "bg": "rgba(239, 68, 68, 0.09)",
            "border": "rgba(239, 68, 68, 0.13)",
        },
        "blue": {
            "bg": "rgba(56, 189, 248, 0.09)",
            "border": "rgba(56, 189, 248, 0.13)",
        },
        "purple": {
            "bg": "rgba(168, 85, 247, 0.09)",
            "border": "rgba(168, 85, 247, 0.13)",
        },
        "yellow": {
            "bg": "rgba(245, 158, 11, 0.09)",
            "border": "rgba(245, 158, 11, 0.13)",
        },
    }

    c = colors[color]

    html = (
        f'<div style="'
        f'background:{c["bg"]};'
        f'border:1px solid {c["border"]};'
        f'border-radius:9px;'
        f'padding:13px 14px;'
        f'min-height:82px;'
        f'box-sizing:border-box;'
        f'">'
        f'<div style="'
        f'font-size:0.72rem;'
        f'opacity:0.72;'
        f'margin-bottom:7px;'
        f'">'
        f'{label}'
        f'</div>'
        f'<div style="'
        f'font-size:1.55rem;'
        f'font-weight:600;'
        f'line-height:1.15;'
        f'">'
        f'{value}'
        f'</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def portfolio_allocation_title():
    st.markdown(
        """
        <div style="
            margin-top:16px;
            margin-bottom:8px;
            font-size:0.95rem;
            font-weight:500;
        ">
            Portfolio allocation
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_portfolio_table(
    weights,
    prices,
    capital,
    use_fractional_shares,
):

    df = weights.rename("Weight").to_frame()

    df.insert(
        0,
        "Company",
        [get_company_name(ticker) for ticker in df.index],
    )

    df["Target Amount"] = df["Weight"] * capital

    df["Price"] = (
        prices.iloc[-1]
        .reindex(df.index)
    )

    if use_fractional_shares:
        df["Quantity"] = (
            df["Target Amount"] / df["Price"]
        )
    else:
        df["Quantity"] = (
            df["Target Amount"] / df["Price"]
        ).fillna(0).astype(int)

    df["Invested"] = (
        df["Quantity"] * df["Price"]
    )

    df = df.sort_values(
        by="Invested",
        ascending=False,
    )

    return df.reset_index(drop=True)


def style_portfolio_table(
    df,
    use_fractional_shares,
):

    styled_df = df.style

    # =====================================================
    # DO NOT MODIFY THESE COLORS
    # =====================================================

    def make_heatmap(
        column,
        rgb,
        min_alpha=0.06,
        max_alpha=0.32,
    ):

        positive_values = df.loc[
            df[column] > 0,
            column,
        ]

        if positive_values.empty:
            return None

        min_value = positive_values.min()
        max_value = positive_values.max()

        def color_cell(value):

            if value <= 0:
                return ""

            if max_value == min_value:
                intensity = 1.0
            else:
                intensity = (
                    (value - min_value)
                    / (max_value - min_value)
                )

            alpha = (
                min_alpha
                + (max_alpha - min_alpha)
                * intensity
            )

            return (
                f"background-color: rgba({rgb}, {alpha:.3f}); "
                "color: #ffffff; "
                "font-weight: 600;"
            )

        return color_cell

    # Weight -> Purple
    weight_color = make_heatmap(
        "Weight",
        "168, 85, 247",
    )

    if weight_color is not None:
        styled_df = styled_df.map(
            weight_color,
            subset=["Weight"],
        )

    # Price -> Green
    price_color = make_heatmap(
        "Price",
        "34, 197, 94",
    )

    if price_color is not None:
        styled_df = styled_df.map(
            price_color,
            subset=["Price"],
        )

    # Quantity -> Blue
    quantity_color = make_heatmap(
        "Quantity",
        "56, 189, 248",
    )

    if quantity_color is not None:
        styled_df = styled_df.map(
            quantity_color,
            subset=["Quantity"],
        )

    # Quantity = 0 -> entire row red
    def highlight_zero_quantity(row):

        if row["Quantity"] == 0:
            return [
                (
                    "background-color: rgba(239, 68, 68, 0.13); "
                    "color: #ffb4b4;"
                )
            ] * len(row)

        return [""] * len(row)

    styled_df = styled_df.apply(
        highlight_zero_quantity,
        axis=1,
    )

    styled_df = styled_df.format(
        {
            "Weight": "{:.2%}",
            "Target Amount": "€{:,.2f}",
            "Price": "€{:,.2f}",
            "Quantity": (
                "{:.4f}"
                if use_fractional_shares
                else "{:.0f}"
            ),
            "Invested": "€{:,.2f}",
        }
    )

    return styled_df


def display_portfolio_table(
    df,
    use_fractional_shares,
):

    styled_table = style_portfolio_table(
        df,
        use_fractional_shares,
    )

    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Company": st.column_config.TextColumn(
                "Instrument",
                width="small",
                help="Nome completo dello strumento",
            ),
            "Weight": st.column_config.Column(
                "Weight",
                width="small",
            ),
            "Target Amount": st.column_config.Column(
                "Target Amount",
                width="small",
            ),
            "Price": st.column_config.Column(
                "Price",
                width="small",
            ),
            "Quantity": st.column_config.Column(
                "Quantity",
                width="small",
            ),
            "Invested": st.column_config.Column(
                "Invested",
                width="small",
            ),
        },
    )


def _comparison_equity(result, key="ff3_bt"):
    data = result.get(key)
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        return None
    if isinstance(data, pd.DataFrame):
        if "Portfolio Value" not in data.columns:
            return None
        data = data["Portfolio Value"]
    out = data.copy().astype(float)
    out.index = pd.to_datetime(out.index).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna()
    return out if len(out) >= 2 and (out > 0).all() else None


def _comparison_diagnostics(result):
    history = result.get("ff3_weights_history")
    if not isinstance(history, pd.DataFrame) or history.empty:
        return None
    weights = history.fillna(0.0).astype(float)
    first = weights.iloc[0]
    first = first[first > 1e-10].sort_values(ascending=False)
    if first.empty:
        return None
    return {
        "Active assets at inception": int(len(first)),
        "Largest initial weight": float(first.max()),
        "Top-3 initial concentration": float(first.head(3).sum()),
        "Initial HHI": float((first ** 2).sum()),
        "Target-weight turnover proxy (ex initial)": float(calculate_turnover(weights).iloc[1:].sum()),
    }


def _build_same_initial_ff3_buy_hold(prices, standard_result):
    """Counterfactual with identical inception date and initial target weights.

    No future data is used for weights. Only subsequent realized asset prices
    value a constant number of fractional shares; no rebalancing or costs.
    """
    equity = _comparison_equity(standard_result)
    history = standard_result.get("ff3_weights_history")
    if equity is None or not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError("Missing FF3 equity or initial weights for the comparison.")
    start, end = equity.index[0], equity.index[-1]
    px = prices.copy()
    px.index = pd.to_datetime(px.index).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    px = px.loc[(px.index >= start) & (px.index <= end)]
    if not px.index.equals(equity.index):
        raise ValueError("Price calendar differs from the Walk-Forward equity calendar.")
    initial = history.iloc[0].astype(float)
    initial = initial[initial > 1e-10]
    if initial.empty or not initial.index.isin(px.columns).all():
        raise ValueError("Initial FF3 holdings are missing from the price matrix.")
    asset_prices = px[initial.index].astype(float)
    if asset_prices.isna().any().any() or (asset_prices <= 0).any().any():
        raise ValueError("Missing or invalid prices for initial FF3 holdings; comparison skipped.")
    initial = initial / initial.sum()
    quantities = initial / asset_prices.iloc[0]
    values = asset_prices.mul(quantities, axis=1).sum(axis=1)
    return (values / values.iloc[0] * float(equity.iloc[0])).rename("Same-initial Buy & Hold")


def render_backtesting_comparison(standard_results, fixed_result):
    if not isinstance(standard_results, dict) or not isinstance(fixed_result, dict):
        return
    if "ff3_bt" in standard_results:
        standard_results = {"fixed": standard_results}
    st.subheader("Backtesting Comparison & Diagnostics")
    st.info("For the isolated impact of rebalancing, use the Same Initial FF3 comparison below. "
            "The original Fixed Horizon comparison also changes the initial portfolio and start date.")
    st.caption("Common-date returns are re-based to the same starting capital. "
               "They are not new optimizations and do not change saved strategy results.")
    fixed_equity = _comparison_equity(fixed_result)
    if fixed_equity is None:
        st.info("No Fixed Horizon FF3 equity curve available for comparison.")
        return
    curves = {"Fixed Horizon FF3": fixed_equity}
    for policy, result in standard_results.items():
        if isinstance(result, dict):
            eq = _comparison_equity(result)
            if eq is not None:
                curves[f"Walk-Forward {policy.title()} FF3"] = eq
    if len(curves) < 2:
        st.info("Run Standard and Fixed Horizon to compare their FF3 results.")
        return
    common_dates = None
    for eq in curves.values():
        common_dates = eq.index if common_dates is None else common_dates.intersection(eq.index)
    common_dates = common_dates.sort_values()
    if len(common_dates) < 2:
        st.warning("There are fewer than two shared trading dates.")
        return
    common_growth = pd.DataFrame({
        name: eq.reindex(common_dates) / eq.reindex(common_dates).iloc[0]
        for name, eq in curves.items()
    })
    common_returns = (common_growth.iloc[-1] - 1).rename("Common-date return")
    st.markdown("##### Matched out-of-sample dates")
    st.caption(f"Shared period: {common_dates[0]:%Y-%m-%d} to "
               f"{common_dates[-1]:%Y-%m-%d} ({len(common_dates)} observations). "
               "This controls for dates, not differences in initial holdings.")
    st.dataframe(common_returns.to_frame().style.format("{:+.2%}"), use_container_width=True)
    fig = go.Figure()
    for name in common_growth.columns:
        fig.add_trace(go.Scatter(x=common_growth.index, y=common_growth[name],
                                 name=name, mode="lines"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Growth of 1 EUR",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key="matched_ff3_growth")
    if "fixed" in standard_results:
        wf = standard_results["fixed"]
        cf = wf.get("ff3_same_initial_buy_hold")
        wf_equity = _comparison_equity(wf)
        if isinstance(cf, pd.Series) and wf_equity is not None:
            cf = cf.copy()
            cf.index = pd.to_datetime(cf.index).normalize()
            if cf.index.equals(wf_equity.index):
                st.markdown("##### Same initial FF3 portfolio: rebalance vs hold")
                wf_ret = wf_equity.iloc[-1] / wf_equity.iloc[0] - 1
                bh_ret = cf.iloc[-1] / cf.iloc[0] - 1
                st.caption("Identical initial date, assets and target weights. "
                           "Buy & Hold keeps fractional share quantities fixed; "
                           "Walk-Forward follows its original rebalancing logic. "
                           "Both exclude transaction costs.")
                st.dataframe(pd.DataFrame({
                    "Return": [f"{wf_ret:+.2%}", f"{bh_ret:+.2%}",
                               f"{100 * (wf_ret - bh_ret):+.2f} pp"],
                    "Start date": [str(wf_equity.index[0].date())] * 3,
                    "End date": [str(wf_equity.index[-1].date())] * 3,
                }, index=["FF3 Walk-Forward Fixed", "Same-initial Buy & Hold",
                          "Difference: rebalancing vs hold"]), use_container_width=True)
                fig2 = go.Figure()
                for name, eq in [("FF3 Walk-Forward Fixed", wf_equity),
                                 ("Same-initial Buy & Hold", cf)]:
                    fig2.add_trace(go.Scatter(x=eq.index, y=eq / eq.iloc[0],
                                              name=name, mode="lines"))
                fig2.update_layout(xaxis_title="Date", yaxis_title="Growth of 1 EUR",
                                   hovermode="x unified")
                st.plotly_chart(fig2, use_container_width=True, key="same_initial_ff3_growth")
        elif wf_equity is not None:
            st.info("Same-initial Buy & Hold is not available in this run. "
                    "Run the backtests again to populate the counterfactual.")
    st.markdown("##### Concentration, turnover and factor coverage")
    diag_rows = {}
    for label, result in [(f"Walk-Forward {k.title()}", v)
                          for k, v in standard_results.items() if isinstance(v, dict)] + [
                              ("Fixed Horizon", fixed_result)]:
        diag = _comparison_diagnostics(result)
        if diag is None:
            continue
        attr = result.get("ff3_performance_attribution")
        if isinstance(attr, dict):
            diag["Factor coverage"] = (f"{attr.get('covered_days', 0)}/"
                                       f"{attr.get('total_days', 0)}")
            diag["Unattributed contribution"] = float(
                attr.get("summary", pd.Series(dtype=float)).get("Unattributed days", 0.0))
        diag_rows[label] = diag
    if diag_rows:
        diag_df = pd.DataFrame(diag_rows).T
        st.dataframe(diag_df.style.format({
            "Largest initial weight": "{:.1%}",
            "Top-3 initial concentration": "{:.1%}",
            "Initial HHI": "{:.3f}",
            "Target-weight turnover proxy (ex initial)": "{:.1%}",
            "Unattributed contribution": "{:+.2%}",
        }, na_rep="—"), use_container_width=True)
        st.caption("Turnover is half the absolute change in successive TARGET weights, "
                   "not drift-adjusted executed trade turnover. "
                   "Sector concentration and actual trading costs are not estimated here.")
    st.warning("Factor attribution may be incomplete when recent Fama–French data "
               "are unavailable. 'Unattributed days' reconciles realized P&L "
               "but does not assign missing factor contributions.")


def calculate_drawdown(backtest_data):

    values = backtest_data["Portfolio Value"]
    running_max = values.cummax()

    return (values / running_max) - 1


def build_benchmark_backtest(
    benchmark_prices,
    start_date,
    end_date,
    initial_capital,
):

    series = (
        benchmark_prices.iloc[:, 0]
        .copy()
    )

    series = series.loc[
        (series.index >= start_date)
        & (series.index <= end_date)
    ].dropna()

    if len(series) < 2:
        raise ValueError(
            "Not enough FTSE MIB data "
            "for the selected backtest period."
        )

    normalized_value = (
        series / series.iloc[0]
    ) * initial_capital

    benchmark_backtest = pd.DataFrame(
        {
            "Portfolio Value":
                normalized_value
        }
    )

    benchmark_backtest["Daily Return"] = (
        benchmark_backtest[
            "Portfolio Value"
        ]
        .pct_change()
        .fillna(0.0)
    )

    return benchmark_backtest


def calculate_turnover(weights_history):

    weights_history = (
        weights_history
        .copy()
        .fillna(0.0)
    )

    weights_history[
        weights_history.abs() < 1e-8
    ] = 0.0

    turnover = pd.Series(
        index=weights_history.index,
        dtype=float,
    )

    if len(weights_history) == 0:
        return turnover

    turnover.iloc[0] = 1.0

    for i in range(
        1,
        len(weights_history),
    ):

        previous_weights = (
            weights_history.iloc[i - 1]
        )

        new_weights = (
            weights_history.iloc[i]
        )

        turnover.iloc[i] = (
            0.5
            * (
                new_weights
                - previous_weights
            )
            .abs()
            .sum()
        )

    return turnover


def prepare_rebalancing_history(
    weights_history,
):

    df = (
        weights_history
        .copy()
        .fillna(0.0)
    )

    df[
        df.abs() < 1e-8
    ] = 0.0

    turnover = calculate_turnover(df)

    rename_map = {
        ticker: get_company_name(ticker)
        for ticker in df.columns
    }

    df = df.rename(
        columns=rename_map
    )

    df.insert(
        0,
        "Turnover",
        turnover.values,
    )

    df.index = pd.to_datetime(df.index)
    df.index.name = "Rebalance Date"

    return df


def get_fixed_horizon_prices(
    tickers,
    estimation_days,
    backtest_days,
):

    full_prices = get_prices(
        tickers,
        period="10y",
    )

    required_days = (
        estimation_days
        + backtest_days
        + 2
    )

    if len(full_prices) < required_days:

        raise ValueError(
            "Not enough historical data for the "
            "selected Fixed-Horizon configuration."
        )

    return full_prices.tail(
        required_days
    )
def run_fixed_horizon_engine(
    tickers,
    prices,
    estimation_days,
    backtest_days,
    risk_free_rate,
    max_weight,
    capital,
    benchmark_period="10y",
    selection_top_n=None,
    selection_policy="fixed",
    ff3_factors=None,
    eurusd_prices=None,
):
    """
    Fixed-Horizon Backtest.

    L'ottimizzazione viene eseguita UNA SOLA VOLTA alla fine
    del periodo di estimation.

    I pesi ottenuti vengono poi mantenuti invariati durante
    tutto il periodo di backtest.
    """

    prices = prices.sort_index().copy()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(axis=1, how="all")

    required_rows = (
        estimation_days
        + backtest_days
        + 2
    )

    if len(prices) < required_rows:
        raise ValueError(
            "Not enough historical data for the "
            "selected Fixed-Horizon configuration."
        )

    prices = prices.tail(required_rows)

    # =====================================================
    # ESTIMATION / BACKTEST SPLIT
    # =====================================================

    # Il periodo di estimation termina prima
    # dell'inizio del backtest.
    estimation_prices = prices.iloc[
        :estimation_days + 1
    ]

    # Il primo giorno di questa finestra è il giorno
    # in cui viene effettuato l'investimento simulato.
    test_prices = prices.iloc[
        estimation_days + 1:
        estimation_days + backtest_days + 2
    ]

    if len(test_prices) < 2:
        raise ValueError(
            "Not enough observations for the "
            "Fixed-Horizon backtest."
        )

    # =====================================================
    # POINT-IN-TIME ASSET SELECTION AT T0
    # =====================================================
    if selection_top_n is not None:
        _, selected_tickers = screen_prices_point_in_time(
            estimation_prices,
            top_n=int(selection_top_n),
            min_observations=min(estimation_days + 1, 253),
        )

        first_test_prices = test_prices.iloc[0]
        selected_tickers = [
            ticker for ticker in selected_tickers
            if ticker in first_test_prices.index
            and pd.notna(first_test_prices[ticker])
            and first_test_prices[ticker] > 0
        ]

        if len(selected_tickers) < 2:
            raise ValueError(
                "Point-in-time screening produced fewer than two "
                "tradable assets at the Fixed-Horizon start."
            )

        estimation_prices = estimation_prices[selected_tickers].dropna()
        test_prices = test_prices[selected_tickers].ffill()
        tickers = selected_tickers
    else:
        estimation_prices = estimation_prices.dropna()
        test_prices = test_prices.dropna()

    # =====================================================
    # PORTFOLIO OPTIMIZATION AT T0
    # =====================================================

    estimation_returns = calculate_returns(
        estimation_prices
    )

    expected_returns = annualized_returns(
        estimation_returns
    )

    covariance = covariance_matrix(
        estimation_returns
    )

    # Maximum Sharpe calcolato una sola volta
    max_sharpe_weights = (
        optimize_maximum_sharpe(
            expected_returns,
            covariance,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
        )
    )

    # CAPM Maximum Sharpe calcolato una sola volta usando solo
    # il periodo di estimation precedente al backtest.
    market_prices = get_prices(
        [FTSE_MIB_TICKER],
        period=benchmark_period,
    )
    market_series = market_prices.iloc[:, 0]
    estimation_market_prices = (
        market_series.reindex(estimation_prices.index).dropna()
    )
    estimation_market_returns = (
        estimation_market_prices.pct_change().dropna()
    )
    capm_mu = capm_expected_returns(
        estimation_returns,
        estimation_market_returns,
        risk_free_rate=risk_free_rate,
    )
    capm_weights = optimize_maximum_sharpe(
        capm_mu,
        covariance,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
    )

    # FF3: stima esclusivamente sulla finestra di estimation precedente al test.
    ff3_mu, ff3_stats = estimate_ff3_eur_returns(
        estimation_prices, ff3_factors, eurusd_prices, return_stats=True
    )
    ff3_mu = ff3_mu.reindex(covariance.index)
    ff3_weights = optimize_maximum_sharpe(
        ff3_mu, covariance, risk_free_rate=risk_free_rate, max_weight=max_weight
    )

    # Minimum Volatility calcolato una sola volta
    min_vol_weights = (
        optimize_minimum_volatility(
            expected_returns,
            covariance,
            max_weight=max_weight,
        )
    )

    # Risk Parity calcolato una sola volta
    risk_parity_weights = (
        optimize_risk_parity(
            covariance,
            max_weight=max_weight,
        )
    )

    # HRP calcolato una sola volta
    hrp_weights = (
        optimize_hrp(
            covariance,
            max_weight=max_weight,
        )
    )

    # Equal Weight come benchmark interno
    equal_weight_weights = pd.Series(
        1.0 / len(tickers),
        index=expected_returns.index,
    )

    # =====================================================
    # BUY & HOLD SIMULATION
    # =====================================================

    def build_buy_and_hold(weights):

        weights = (
            weights
            .reindex(test_prices.columns)
            .fillna(0.0)
        )

        weights = weights / weights.sum()

        start_prices = test_prices.iloc[0]

        # Nel motore di backtest usiamo quantità frazionarie
        # per replicare esattamente i pesi teorici.
        quantities = (
            capital
            * weights
            / start_prices
        )

        portfolio_value = (
            test_prices
            * quantities
        ).sum(axis=1)

        backtest = pd.DataFrame(
            {
                "Portfolio Value":
                    portfolio_value
            }
        )

        backtest["Daily Return"] = (
            backtest[
                "Portfolio Value"
            ]
            .pct_change()
            .fillna(0.0)
        )

        return backtest

    max_sharpe_bt = build_buy_and_hold(
        max_sharpe_weights
    )

    capm_bt = build_buy_and_hold(
        capm_weights
    )

    ff3_bt = build_buy_and_hold(ff3_weights)

    min_vol_bt = build_buy_and_hold(
        min_vol_weights
    )

    risk_parity_bt = build_buy_and_hold(
        risk_parity_weights
    )

    hrp_bt = build_buy_and_hold(
        hrp_weights
    )

    equal_weight_bt = build_buy_and_hold(
        equal_weight_weights
    )

    # =====================================================
    # EFFECTIVE DATES
    # =====================================================

    effective_start = (
        test_prices.index[0]
    )

    effective_end = (
        test_prices.index[-1]
    )

    estimation_start = (
        estimation_prices.index[0]
    )

    estimation_end = (
        estimation_prices.index[-1]
    )

    # =====================================================
    # INITIAL ALLOCATION HISTORY
    # =====================================================

    # Nel Fixed Horizon esiste una sola allocazione:
    # quella determinata all'inizio del backtest.

    max_sharpe_weights_history = (
        pd.DataFrame(
            [
                max_sharpe_weights
                .reindex(test_prices.columns)
                .fillna(0.0)
            ],
            index=[effective_start],
        )
    )

    capm_weights_history = (
        pd.DataFrame(
            [
                capm_weights
                .reindex(test_prices.columns)
                .fillna(0.0)
            ],
            index=[effective_start],
        )
    )

    ff3_weights_history = pd.DataFrame(
        [ff3_weights.reindex(test_prices.columns).fillna(0.0)],
        index=[effective_start],
    )
    ff3_aligned_stats = ff3_stats.reindex(ff3_weights.index)
    ff3_factor_exposure_history = pd.DataFrame([{
        "Market Beta": float((ff3_weights * ff3_aligned_stats["Beta MKT"]).sum()),
        "SMB Exposure": float((ff3_weights * ff3_aligned_stats["Beta SMB"]).sum()),
        "HML Exposure": float((ff3_weights * ff3_aligned_stats["Beta HML"]).sum()),
        "Active Assets": int((ff3_weights > 1e-10).sum()),
    }], index=pd.DatetimeIndex([effective_start], name="Rebalance Date"))


    min_vol_weights_history = (
        pd.DataFrame(
            [
                min_vol_weights
                .reindex(test_prices.columns)
                .fillna(0.0)
            ],
            index=[effective_start],
        )
    )

    risk_parity_weights_history = (
        pd.DataFrame(
            [
                risk_parity_weights
                .reindex(test_prices.columns)
                .fillna(0.0)
            ],
            index=[effective_start],
        )
    )

    hrp_weights_history = (
        pd.DataFrame(
            [
                hrp_weights
                .reindex(test_prices.columns)
                .fillna(0.0)
            ],
            index=[effective_start],
        )
    )

    equal_weight_weights_history = (
        pd.DataFrame(
            [
                equal_weight_weights
                .reindex(test_prices.columns)
                .fillna(0.0)
            ],
            index=[effective_start],
        )
    )

    # =====================================================
    # FTSE MIB BENCHMARK
    # =====================================================

    ftse_mib_bt = (
        build_benchmark_backtest(
            market_prices,
            effective_start,
            effective_end,
            capital,
        )
    )

    # =====================================================
    # METRICS
    # =====================================================

    max_bt_metrics = (
        calculate_backtest_metrics(
            max_sharpe_bt,
            risk_free_rate,
        )
    )

    capm_bt_metrics = (
        calculate_backtest_metrics(
            capm_bt,
            risk_free_rate,
        )
    )

    ff3_bt_metrics = calculate_backtest_metrics(ff3_bt, risk_free_rate)

    min_bt_metrics = (
        calculate_backtest_metrics(
            min_vol_bt,
            risk_free_rate,
        )
    )

    risk_parity_bt_metrics = (
        calculate_backtest_metrics(
            risk_parity_bt,
            risk_free_rate,
        )
    )

    hrp_bt_metrics = (
        calculate_backtest_metrics(
            hrp_bt,
            risk_free_rate,
        )
    )

    equal_bt_metrics = (
        calculate_backtest_metrics(
            equal_weight_bt,
            risk_free_rate,
        )
    )

    ftse_bt_metrics = (
        calculate_backtest_metrics(
            ftse_mib_bt,
            risk_free_rate,
        )
    )

    # =====================================================
    # RESULT
    # =====================================================

    return {
        "tickers":
            tickers.copy(),

        "capital":
            capital,

        "risk_free_rate":
            risk_free_rate,

        "max_weight":
            max_weight,

        "effective_start":
            effective_start,

        "effective_end":
            effective_end,

        "estimation_start":
            estimation_start,

        "estimation_end":
            estimation_end,

        "max_sharpe_bt":
            max_sharpe_bt,

        "ff3_bt": ff3_bt,
        "ff3_weights_history": ff3_weights_history,
        "ff3_bt_metrics": ff3_bt_metrics,
        "ff3_factor_exposure_history": ff3_factor_exposure_history,

        "capm_bt":
            capm_bt,

        "min_vol_bt":
            min_vol_bt,

        "risk_parity_bt":
            risk_parity_bt,

        "hrp_bt":
            hrp_bt,

        "equal_weight_bt":
            equal_weight_bt,

        "ftse_mib_bt":
            ftse_mib_bt,

        "max_sharpe_weights_history":
            max_sharpe_weights_history,

        "capm_weights_history":
            capm_weights_history,

        "min_vol_weights_history":
            min_vol_weights_history,

        "risk_parity_weights_history":
            risk_parity_weights_history,

        "hrp_weights_history":
            hrp_weights_history,

        "equal_weight_weights_history":
            equal_weight_weights_history,

        "max_bt_metrics":
            max_bt_metrics,

        "capm_bt_metrics":
            capm_bt_metrics,

        "min_bt_metrics":
            min_bt_metrics,

        "risk_parity_bt_metrics":
            risk_parity_bt_metrics,

        "hrp_bt_metrics":
            hrp_bt_metrics,

        "equal_bt_metrics":
            equal_bt_metrics,

        "ftse_bt_metrics":
            ftse_bt_metrics,

        "is_fixed_horizon":
            True,
    }

def validate_portfolio(
    tickers,
    max_weight,
):

    if len(tickers) < 2:

        st.error(
            "Seleziona almeno due strumenti."
        )

        return False

    if max_weight < 1 / len(tickers):

        minimum_feasible_weight = (
            1 / len(tickers)
        )

        st.error(
            f"Il peso massimo selezionato "
            f"non è compatibile con "
            f"{len(tickers)} asset. "
            f"Deve essere almeno "
            f"{minimum_feasible_weight:.1%}."
        )

        return False

    return True


def run_backtest_engine(
    tickers,
    prices,
    lookback_days,
    rebalance_days,
    risk_free_rate,
    max_weight,
    capital,
    fixed_backtest_days=None,
    benchmark_period="10y",
    selection_top_n=None,
    selection_policy="fixed",
    market_prices=None,
    shared_window_cache=None,
    ff3_factors=None,
    eurusd_prices=None,
):

    # Il benchmark viene condiviso tra le politiche e le strategie.
    if market_prices is None:
        market_prices = get_prices([FTSE_MIB_TICKER], period=benchmark_period)
    if shared_window_cache is None:
        shared_window_cache = {}

    (
        max_sharpe_bt,
        max_sharpe_weights_history,
    ) = backtest_portfolio(
        prices,
        strategy="max_sharpe",
        initial_capital=capital,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        selection_top_n=selection_top_n,
        selection_policy=selection_policy,
        shared_window_cache=shared_window_cache,
    )

    (
        capm_bt,
        capm_weights_history,
    ) = backtest_portfolio(
        prices,
        strategy="capm_max_sharpe",
        initial_capital=capital,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        market_prices=market_prices,
        selection_top_n=selection_top_n,
        selection_policy=selection_policy,
        shared_window_cache=shared_window_cache,
    )

    ff3_exposure_records = []
    (
        ff3_bt,
        ff3_weights_history,
    ) = backtest_portfolio(
        prices, strategy="ff3_max_sharpe", initial_capital=capital,
        lookback_days=lookback_days, rebalance_days=rebalance_days,
        risk_free_rate=risk_free_rate, max_weight=max_weight,
        selection_top_n=selection_top_n, selection_policy=selection_policy,
        shared_window_cache=shared_window_cache,
        ff3_factors=ff3_factors, eurusd_prices=eurusd_prices,
        ff3_exposure_records=ff3_exposure_records,
    )
    ff3_factor_exposure_history = pd.DataFrame(ff3_exposure_records)
    if not ff3_factor_exposure_history.empty:
        ff3_factor_exposure_history = ff3_factor_exposure_history.set_index("Rebalance Date")

    (
        min_vol_bt,
        min_vol_weights_history,
    ) = backtest_portfolio(
        prices,
        strategy="min_volatility",
        initial_capital=capital,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        selection_top_n=selection_top_n,
        selection_policy=selection_policy,
        shared_window_cache=shared_window_cache,
    )

    (
        risk_parity_bt,
        risk_parity_weights_history,
    ) = backtest_portfolio(
        prices,
        strategy="risk_parity",
        initial_capital=capital,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        selection_top_n=selection_top_n,
        selection_policy=selection_policy,
        shared_window_cache=shared_window_cache,
    )

    (
        hrp_bt,
        hrp_weights_history,
    ) = backtest_portfolio(
        prices,
        strategy="hrp",
        initial_capital=capital,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        selection_top_n=selection_top_n,
        selection_policy=selection_policy,
        shared_window_cache=shared_window_cache,
    )

    (
        equal_weight_bt,
        equal_weight_weights_history,
    ) = backtest_portfolio(
        prices,
        strategy="equal_weight",
        initial_capital=capital,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        selection_top_n=selection_top_n,
        selection_policy=selection_policy,
        shared_window_cache=shared_window_cache,
    )

    # -----------------------------------------------------
    # FIXED-HORIZON TRIM
    # -----------------------------------------------------

    if fixed_backtest_days is not None:

        max_sharpe_bt = (
            max_sharpe_bt
            .iloc[:fixed_backtest_days + 1]
        )

        ff3_bt = ff3_bt.reindex(max_sharpe_bt.index).dropna()

        capm_bt = (
            capm_bt
            .reindex(max_sharpe_bt.index)
            .dropna()
        )

        min_vol_bt = (
            min_vol_bt
            .reindex(max_sharpe_bt.index)
            .dropna()
        )

        risk_parity_bt = (
            risk_parity_bt
            .reindex(max_sharpe_bt.index)
            .dropna()
        )

        hrp_bt = (
            hrp_bt
            .reindex(max_sharpe_bt.index)
            .dropna()
        )

        equal_weight_bt = (
            equal_weight_bt
            .reindex(max_sharpe_bt.index)
            .dropna()
        )

        effective_end_for_weights = (
            max_sharpe_bt.index[-1]
        )

        max_sharpe_weights_history = (
            max_sharpe_weights_history.loc[
                max_sharpe_weights_history.index
                <= effective_end_for_weights
            ]
        )

        ff3_weights_history = ff3_weights_history.loc[
            ff3_weights_history.index <= effective_end_for_weights
        ]

        capm_weights_history = (
            capm_weights_history.loc[
                capm_weights_history.index
                <= effective_end_for_weights
            ]
        )

        min_vol_weights_history = (
            min_vol_weights_history.loc[
                min_vol_weights_history.index
                <= effective_end_for_weights
            ]
        )

        risk_parity_weights_history = (
            risk_parity_weights_history.loc[
                risk_parity_weights_history.index
                <= effective_end_for_weights
            ]
        )

        hrp_weights_history = (
            hrp_weights_history.loc[
                hrp_weights_history.index
                <= effective_end_for_weights
            ]
        )

        equal_weight_weights_history = (
            equal_weight_weights_history.loc[
                equal_weight_weights_history.index
                <= effective_end_for_weights
            ]
        )

    effective_start = (
        max_sharpe_bt.index[0]
    )

    effective_end = (
        max_sharpe_bt.index[-1]
    )

    # -----------------------------------------------------
    # FTSE MIB
    # -----------------------------------------------------

    ftse_mib_bt = (
        build_benchmark_backtest(
            market_prices,
            effective_start,
            effective_end,
            capital,
        )
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    max_bt_metrics = (
        calculate_backtest_metrics(
            max_sharpe_bt,
            risk_free_rate,
        )
    )

    ff3_bt_metrics = calculate_backtest_metrics(ff3_bt, risk_free_rate)

    capm_bt_metrics = (
        calculate_backtest_metrics(
            capm_bt,
            risk_free_rate,
        )
    )

    min_bt_metrics = (
        calculate_backtest_metrics(
            min_vol_bt,
            risk_free_rate,
        )
    )

    risk_parity_bt_metrics = (
        calculate_backtest_metrics(
            risk_parity_bt,
            risk_free_rate,
        )
    )

    hrp_bt_metrics = (
        calculate_backtest_metrics(
            hrp_bt,
            risk_free_rate,
        )
    )

    equal_bt_metrics = (
        calculate_backtest_metrics(
            equal_weight_bt,
            risk_free_rate,
        )
    )

    ftse_bt_metrics = (
        calculate_backtest_metrics(
            ftse_mib_bt,
            risk_free_rate,
        )
    )

    return {
        "tickers":
            tickers.copy(),

        "capital":
            capital,

        "risk_free_rate":
            risk_free_rate,

        "max_weight":
            max_weight,

        "effective_start":
            effective_start,

        "effective_end":
            effective_end,

        "max_sharpe_bt":
            max_sharpe_bt,

        "ff3_bt": ff3_bt,
        "ff3_weights_history": ff3_weights_history,
        "ff3_bt_metrics": ff3_bt_metrics,
        "ff3_factor_exposure_history": ff3_factor_exposure_history,

        "capm_bt":
            capm_bt,

        "min_vol_bt":
            min_vol_bt,

        "risk_parity_bt":
            risk_parity_bt,

        "hrp_bt":
            hrp_bt,

        "equal_weight_bt":
            equal_weight_bt,

        "ftse_mib_bt":
            ftse_mib_bt,

        "max_sharpe_weights_history":
            max_sharpe_weights_history,

        "capm_weights_history":
            capm_weights_history,

        "min_vol_weights_history":
            min_vol_weights_history,

        "risk_parity_weights_history":
            risk_parity_weights_history,

        "hrp_weights_history":
            hrp_weights_history,

        "equal_weight_weights_history":
            equal_weight_weights_history,

        "max_bt_metrics":
            max_bt_metrics,

        "capm_bt_metrics":
            capm_bt_metrics,

        "min_bt_metrics":
            min_bt_metrics,

        "risk_parity_bt_metrics":
            risk_parity_bt_metrics,

        "hrp_bt_metrics":
            hrp_bt_metrics,

        "equal_bt_metrics":
            equal_bt_metrics,

        "ftse_bt_metrics":
            ftse_bt_metrics,
    }


def display_backtest_results(
    result,
    title,
    description,
    lookback_label,
    backtest_period_label,
    rebalance_label,
    result_key,
):

    if "ff3_bt" not in result:
        st.info("These results predate FF3 backtesting. Run Backtests again to include FF3.")
        return

    st.markdown("---")

    st.header(title)

    st.caption(description)

    # =====================================================
    # CONFIGURATION SUMMARY
    # =====================================================

    info_col1, info_col2, info_col3, info_col4, info_col5 = (
        st.columns(5)
    )

    info_col1.metric(
        "Initial Capital",
        f"€{result['capital']:,.2f}",
    )

    info_col2.metric(
        "Estimation Window",
        lookback_label,
    )

    info_col3.metric(
        "Backtest Period",
        backtest_period_label,
    )

    info_col4.metric(
        "Rebalancing",
        rebalance_label,
    )

    info_col5.metric(
        "Backtest Start",
        result[
            "effective_start"
        ].strftime("%d/%m/%Y"),
    )

    st.caption(
        "Effective backtest: "
        f"{result['effective_start'].strftime('%d/%m/%Y')} "
        "→ "
        f"{result['effective_end'].strftime('%d/%m/%Y')}"
    )

    # =====================================================
    # STRATEGY COMPARISON
    # =====================================================

    st.subheader(
        "Strategy Comparison"
    )

    comparison = pd.DataFrame(
        {
            "Maximum Sharpe":
                result["max_bt_metrics"],

            "CAPM Maximum Sharpe":
                result["capm_bt_metrics"],

            "FF3 Maximum Sharpe":
                result["ff3_bt_metrics"],

            "Minimum Volatility":
                result["min_bt_metrics"],

            "Risk Parity":
                result["risk_parity_bt_metrics"],

            "HRP":
                result["hrp_bt_metrics"],

            "Equal Weight":
                result["equal_bt_metrics"],

            "FTSE MIB":
                result["ftse_bt_metrics"],
        }
    ).T

    percentage_columns = [
        column
        for column
        in comparison.columns
        if (
            "Return" in column
            or "Volatility" in column
            or "Drawdown" in column
            or column == "CAGR"
        )
    ]

    comparison_format = {
        column: "{:.2%}"
        for column
        in percentage_columns
    }

    for column in comparison.columns:

        if (
            "Sharpe" in column
            or "Sortino" in column
        ):
            comparison_format[column] = (
                "{:.2f}"
            )

        if column == "Final Value":
            comparison_format[column] = (
                "€{:,.2f}"
            )

    st.dataframe(
        comparison.style.format(
            comparison_format
        ),
        use_container_width=True,
    )

    st.caption(
        "FTSE MIB is shown as a passive buy-and-hold "
        "benchmark starting from the same date and "
        "capital as the portfolio strategies."
    )

    # =====================================================
    # PORTFOLIO GROWTH
    # =====================================================

    st.subheader(
        "Portfolio Growth"
    )

    portfolio_fig = go.Figure()

    portfolio_fig.add_trace(
        go.Scatter(
            x=result[
                "max_sharpe_bt"
            ].index,
            y=result[
                "max_sharpe_bt"
            ]["Portfolio Value"],
            mode="lines",
            name="Maximum Sharpe",
        )
    )

    portfolio_fig.add_trace(
        go.Scatter(
            x=result["capm_bt"].index,
            y=result["capm_bt"]["Portfolio Value"],
            mode="lines",
            name="CAPM Maximum Sharpe",
        )
    )

    portfolio_fig.add_trace(go.Scatter(
        x=result["ff3_bt"].index, y=result["ff3_bt"]["Portfolio Value"],
        mode="lines", name="FF3 Maximum Sharpe",
    ))

    portfolio_fig.add_trace(
        go.Scatter(
            x=result[
                "min_vol_bt"
            ].index,
            y=result[
                "min_vol_bt"
            ]["Portfolio Value"],
            mode="lines",
            name="Minimum Volatility",
        )
    )

    portfolio_fig.add_trace(
        go.Scatter(
            x=result[
                "risk_parity_bt"
            ].index,
            y=result[
                "risk_parity_bt"
            ]["Portfolio Value"],
            mode="lines",
            name="Risk Parity",
        )
    )

    portfolio_fig.add_trace(
        go.Scatter(
            x=result[
                "hrp_bt"
            ].index,
            y=result[
                "hrp_bt"
            ]["Portfolio Value"],
            mode="lines",
            name="HRP",
        )
    )

    portfolio_fig.add_trace(
        go.Scatter(
            x=result[
                "equal_weight_bt"
            ].index,
            y=result[
                "equal_weight_bt"
            ]["Portfolio Value"],
            mode="lines",
            name="Equal Weight",
        )
    )

    portfolio_fig.add_trace(
        go.Scatter(
            x=result[
                "ftse_mib_bt"
            ].index,
            y=result[
                "ftse_mib_bt"
            ]["Portfolio Value"],
            mode="lines",
            name="FTSE MIB",
        )
    )

    portfolio_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value (€)",
        hovermode="x unified",
        legend_title="Strategy",
    )

    portfolio_fig.update_xaxes(
        title_text="Date",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.45)",
        linewidth=1,
        zeroline=False,
    )

    portfolio_fig.update_yaxes(
        title_text="Portfolio Value (€)",
        tickprefix="€",
        tickformat=",.0f",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.45)",
        linewidth=1,
        zeroline=False,
    )

    portfolio_fig.add_hline(
        y=result["capital"],
        line_dash="dash",
        line_width=1,
        line_color="rgba(255, 255, 255, 0.45)",
        annotation_text=f"Initial Capital · €{result['capital']:,.0f}",
        annotation_position="top left",
    )

    st.plotly_chart(
        portfolio_fig,
        use_container_width=True,
        key=f"growth_{result_key}",
    )

    # =====================================================
    # DRAWDOWN
    # =====================================================

    st.subheader(
        "Drawdown"
    )

    max_drawdown = (
        calculate_drawdown(
            result["max_sharpe_bt"]
        )
    )

    capm_drawdown = calculate_drawdown(
        result["capm_bt"]
    )

    ff3_drawdown = calculate_drawdown(result["ff3_bt"])

    min_drawdown = (
        calculate_drawdown(
            result["min_vol_bt"]
        )
    )

    risk_parity_drawdown = (
        calculate_drawdown(
            result["risk_parity_bt"]
        )
    )

    hrp_drawdown = (
        calculate_drawdown(
            result["hrp_bt"]
        )
    )

    equal_drawdown = (
        calculate_drawdown(
            result["equal_weight_bt"]
        )
    )

    ftse_drawdown = (
        calculate_drawdown(
            result["ftse_mib_bt"]
        )
    )

    drawdown_fig = go.Figure()

    drawdown_fig.add_trace(
        go.Scatter(
            x=max_drawdown.index,
            y=max_drawdown,
            mode="lines",
            name="Maximum Sharpe",
        )
    )

    drawdown_fig.add_trace(
        go.Scatter(
            x=capm_drawdown.index,
            y=capm_drawdown,
            mode="lines",
            name="CAPM Maximum Sharpe",
        )
    )

    drawdown_fig.add_trace(go.Scatter(
        x=ff3_drawdown.index, y=ff3_drawdown, mode="lines",
        name="FF3 Maximum Sharpe",
    ))

    drawdown_fig.add_trace(
        go.Scatter(
            x=min_drawdown.index,
            y=min_drawdown,
            mode="lines",
            name="Minimum Volatility",
        )
    )

    drawdown_fig.add_trace(
        go.Scatter(
            x=risk_parity_drawdown.index,
            y=risk_parity_drawdown,
            mode="lines",
            name="Risk Parity",
        )
    )

    drawdown_fig.add_trace(
        go.Scatter(
            x=hrp_drawdown.index,
            y=hrp_drawdown,
            mode="lines",
            name="HRP",
        )
    )

    drawdown_fig.add_trace(
        go.Scatter(
            x=equal_drawdown.index,
            y=equal_drawdown,
            mode="lines",
            name="Equal Weight",
        )
    )

    drawdown_fig.add_trace(
        go.Scatter(
            x=ftse_drawdown.index,
            y=ftse_drawdown,
            mode="lines",
            name="FTSE MIB",
        )
    )

    drawdown_fig.update_layout(
        hovermode="x unified",
        legend_title="Strategy",
    )

    drawdown_fig.update_xaxes(
        title_text="Date",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.45)",
        linewidth=1,
        zeroline=False,
    )

    drawdown_fig.update_yaxes(
        title_text="Drawdown",
        tickformat=".0%",
        showline=True,
        linecolor="rgba(255, 255, 255, 0.45)",
        linewidth=1,
        zeroline=False,
    )

    st.plotly_chart(
        drawdown_fig,
        use_container_width=True,
        key=f"drawdown_{result_key}",
    )

    # =====================================================
    # REBALANCING HISTORY
    # =====================================================

    if result.get(
        "is_fixed_horizon",
        False,
    ):

        st.subheader(
            "Initial Optimal Allocation"
        )

        st.caption(
            "These are the portfolio weights calculated "
            "at the beginning of the backtest using only "
            "the preceding estimation period. "
            "The allocation is then held unchanged for "
            "the entire Fixed-Horizon test."
        )

    else:

        st.subheader(
            "Rebalancing History"
        )

        st.caption(
            "The table shows the target portfolio weights "
            "selected at every rebalance. Turnover measures "
            "how much of the portfolio allocation changed "
            "compared with the previous rebalance."
        )

    rebalance_strategy = st.selectbox(
        "Strategy",
        [
            "Maximum Sharpe",
            "CAPM Maximum Sharpe",
            "FF3 Maximum Sharpe",
            "Minimum Volatility",
            "Risk Parity",
            "HRP",
            "Equal Weight",
        ],
        key=f"rebalance_strategy_{result_key}",
    )

    if (
        rebalance_strategy
        == "Maximum Sharpe"
    ):

        selected_weights_history = (
            result[
                "max_sharpe_weights_history"
            ]
        )

    elif (
        rebalance_strategy
        == "CAPM Maximum Sharpe"
    ):

        selected_weights_history = (
            result["capm_weights_history"]
        )

    elif rebalance_strategy == "FF3 Maximum Sharpe":
        selected_weights_history = result["ff3_weights_history"]

    elif (
        rebalance_strategy
        == "Minimum Volatility"
    ):

        selected_weights_history = (
            result[
                "min_vol_weights_history"
            ]
        )

    elif (
        rebalance_strategy
        == "Risk Parity"
    ):

        selected_weights_history = (
            result[
                "risk_parity_weights_history"
            ]
        )

    elif (
        rebalance_strategy
        == "HRP"
    ):

        selected_weights_history = (
            result[
                "hrp_weights_history"
            ]
        )

    else:

        selected_weights_history = (
            result[
                "equal_weight_weights_history"
            ]
        )

    rebalance_table = (
        prepare_rebalancing_history(
            selected_weights_history
        )
    )

    rebalance_format = {
        column: "{:.2%}"
        for column
        in rebalance_table.columns
    }

    st.dataframe(
        rebalance_table.style.format(
            rebalance_format
        ),
        use_container_width=True,
    )

    render_ff3_attribution(result, key_suffix=result_key)

    # FF3 factor exposures: target-weighted regression sensitivities at each rebalance.
    ff3_exposures = result.get("ff3_factor_exposure_history")
    if isinstance(ff3_exposures, pd.DataFrame) and not ff3_exposures.empty:
        st.subheader("FF3 Factor Exposure History")
        st.caption(
            "Target-weighted FF3 factor loadings estimated using historical data "
            "at each rebalance. These are sensitivities, not factor returns "
            "or a performance attribution."
        )
        exposure_display = ff3_exposures.copy()
        exposure_display.index = pd.to_datetime(exposure_display.index)
        exposure_display.index.name = "Rebalance Date"
        exposure_format = {
            name: "{:.3f}" for name in
            ("Market Beta", "SMB Exposure", "HML Exposure")
            if name in exposure_display.columns
        }
        st.dataframe(exposure_display.style.format(exposure_format),
                     use_container_width=True)
        if len(exposure_display) > 1:
            exposure_fig = go.Figure()
            for name in ("Market Beta", "SMB Exposure", "HML Exposure"):
                if name in exposure_display.columns:
                    exposure_fig.add_trace(go.Scatter(
                        x=exposure_display.index, y=exposure_display[name],
                        name=name, mode="lines+markers"))
            exposure_fig.update_layout(xaxis_title="Rebalance Date",
                                       yaxis_title="Factor Loading", hovermode="x unified")
            st.plotly_chart(exposure_fig, use_container_width=True)

    # =====================================================
    # TURNOVER
    # =====================================================

    turnover_series = (
        calculate_turnover(
            selected_weights_history
        )
    )

    if len(turnover_series) > 1:

        average_turnover = (
            turnover_series
            .iloc[1:]
            .mean()
        )

        max_turnover = (
            turnover_series
            .iloc[1:]
            .max()
        )

        total_turnover = (
            turnover_series
            .iloc[1:]
            .sum()
        )

    else:

        average_turnover = 0.0
        max_turnover = 0.0
        total_turnover = 0.0

    turnover_col1, turnover_col2, turnover_col3 = (
        st.columns(3)
    )

    turnover_col1.metric(
        "Average Turnover",
        f"{average_turnover:.2%}",
    )

    turnover_col2.metric(
        "Maximum Turnover",
        f"{max_turnover:.2%}",
    )

    turnover_col3.metric(
        "Cumulative Turnover",
        f"{total_turnover:.2%}",
    )

    # =====================================================
    # FINAL VALUES
    # =====================================================

    st.subheader(
        "Final Portfolio Value"
    )

    (
        final_col1,
        final_col2,
        final_col3,
        final_col4,
        final_col5,
        final_col6,
        final_col7,
    ) = st.columns(7)

    final_max = (
        result["max_sharpe_bt"]
        ["Portfolio Value"]
        .iloc[-1]
    )

    final_capm = (
        result["capm_bt"]
        ["Portfolio Value"]
        .iloc[-1]
    )

    final_min = (
        result["min_vol_bt"]
        ["Portfolio Value"]
        .iloc[-1]
    )

    final_risk_parity = (
        result["risk_parity_bt"]
        ["Portfolio Value"]
        .iloc[-1]
    )

    final_hrp = (
        result["hrp_bt"]
        ["Portfolio Value"]
        .iloc[-1]
    )

    final_equal = (
        result["equal_weight_bt"]
        ["Portfolio Value"]
        .iloc[-1]
    )

    final_ftse = (
        result["ftse_mib_bt"]
        ["Portfolio Value"]
        .iloc[-1]
    )

    capital = result["capital"]

    final_col1.metric(
        "Maximum Sharpe",
        f"€{final_max:,.2f}",
        delta=(
            f"{final_max / capital - 1:.2%}"
        ),
    )

    final_col2.metric(
        "CAPM Max Sharpe",
        f"€{final_capm:,.2f}",
        delta=(
            f"{final_capm / capital - 1:.2%}"
        ),
    )

    final_col3.metric(
        "Minimum Volatility",
        f"€{final_min:,.2f}",
        delta=(
            f"{final_min / capital - 1:.2%}"
        ),
    )

    final_col4.metric(
        "Risk Parity",
        f"€{final_risk_parity:,.2f}",
        delta=(
            f"{final_risk_parity / capital - 1:.2%}"
        ),
    )

    final_col5.metric(
        "HRP",
        f"€{final_hrp:,.2f}",
        delta=(
            f"{final_hrp / capital - 1:.2%}"
        ),
    )

    final_col6.metric(
        "Equal Weight",
        f"€{final_equal:,.2f}",
        delta=(
            f"{final_equal / capital - 1:.2%}"
        ),
    )

    final_col7.metric(
        "FTSE MIB",
        f"€{final_ftse:,.2f}",
        delta=(
            f"{final_ftse / capital - 1:.2%}"
        ),
    )



    final_ff3 = result["ff3_bt"]["Portfolio Value"].iloc[-1]
    st.metric("FF3 Maximum Sharpe", f"€{final_ff3:,.2f}",
              delta=f"{final_ff3 / capital - 1:.2%}")

# =========================================================
# SESSION STATE
# =========================================================

if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = None

if "factor_model_results" not in st.session_state:
    st.session_state.factor_model_results = None

if (
    st.session_state.factor_model_results is not None
    and (
        "asset_returns" not in st.session_state.factor_model_results
        or "market_returns" not in st.session_state.factor_model_results
    )
):
    st.session_state.factor_model_results = None

if "ff3_results" not in st.session_state:
    st.session_state.ff3_results = None
if "ff5_results" not in st.session_state:
    st.session_state.ff5_results = None

if "standard_backtest_results" not in st.session_state:
    st.session_state.standard_backtest_results = None

if "fixed_backtest_results" not in st.session_state:
    st.session_state.fixed_backtest_results = None

# Invalida risultati creati con versioni precedenti dell'app,
# prima dell'introduzione di HRP.
if (
    st.session_state.optimization_results is not None
    and "hrp_weights" not in st.session_state.optimization_results
):
    st.session_state.optimization_results = None

# Standard Walk-Forward può avere due formati:
# 1) legacy/singola policy: risultato flat con "hrp_bt" e "capm_bt"
# 2) Fixed/Dynamic: dict {"fixed": result, "dynamic": result}
#
# Non bisogna invalidare il secondo formato: altrimenti al rerun causato
# da "Save Experiment" i risultati Standard vengono cancellati prima
# di essere inseriti nello snapshot.
if st.session_state.standard_backtest_results is not None:
    _standard_state = st.session_state.standard_backtest_results

    if isinstance(_standard_state, dict):
        _is_flat_standard_result = (
            "hrp_bt" in _standard_state
            and "capm_bt" in _standard_state
        )

        _policy_results = [
            result
            for policy, result in _standard_state.items()
            if policy in ("fixed", "dynamic")
            and isinstance(result, dict)
        ]

        _is_policy_standard_result = (
            len(_policy_results) > 0
            and all(
                "hrp_bt" in result
                and "capm_bt" in result
                for result in _policy_results
            )
        )

        if not (
            _is_flat_standard_result
            or _is_policy_standard_result
        ):
            st.session_state.standard_backtest_results = None
    else:
        st.session_state.standard_backtest_results = None

if (
    st.session_state.fixed_backtest_results is not None
    and (
        "hrp_bt" not in st.session_state.fixed_backtest_results
        or "capm_bt" not in st.session_state.fixed_backtest_results
    )
):
    st.session_state.fixed_backtest_results = None


# =========================================================
# HEADER
# =========================================================

st.title(
    "Quant Portfolio Optimizer"
)

st.write(
    """
    Quantitative portfolio analysis and optimization
    based on Modern Portfolio Theory.
    """
)


# =========================================================
# ASSET SELECTION STATE
# =========================================================

if "screening_results" not in st.session_state:
    st.session_state.screening_results = None

if "quant_selected_tickers" not in st.session_state:
    st.session_state.quant_selected_tickers = []


# =========================================================
# SIDEBAR — ONLY COMMON PORTFOLIO PARAMETERS
# =========================================================

st.sidebar.header(
    "Portfolio Parameters"
)


# ---------------------------------------------------------
# MARKET
# ---------------------------------------------------------

market = st.sidebar.selectbox(
    "Market",
    sorted(
        instruments[
            "market"
        ].unique()
    ),
)

market_instruments = (
    instruments[
        instruments["market"]
        == market
    ]
    .sort_values("name")
)


# ---------------------------------------------------------
# ASSET SELECTION MODE
# ---------------------------------------------------------

selection_mode = st.sidebar.radio(
    "Asset selection",
    ["Manual Selection", "Quant Selection"],
    horizontal=True,
)

if selection_mode == "Manual Selection":

    selected_names = st.sidebar.multiselect(
        "Select instruments",
        options=market_instruments["name"].tolist(),
        default=[],
        placeholder="Type to search a company...",
    )

    selected_instruments = market_instruments[
        market_instruments["name"].isin(selected_names)
    ]

    tickers = selected_instruments["ticker"].tolist()

    if selected_instruments.empty:
        st.sidebar.caption("No instruments selected.")
    else:
        st.sidebar.caption(
            f"{len(selected_instruments)} instruments selected"
        )

else:

    quant_tickers = [
        ticker
        for ticker in st.session_state.quant_selected_tickers
        if ticker in set(market_instruments["ticker"])
    ]

    selected_instruments = market_instruments[
        market_instruments["ticker"].isin(quant_tickers)
    ]

    tickers = selected_instruments["ticker"].tolist()

    if not tickers:
        st.sidebar.caption(
            "Run the Asset Selection screener to generate the portfolio universe."
        )
    else:
        st.sidebar.caption(
            f"{len(tickers)} instruments selected by the quant screener"
        )


# ---------------------------------------------------------
# RISK-FREE RATE
# ---------------------------------------------------------

risk_free_rate = (
    st.sidebar.number_input(
        "Risk-free rate",
        min_value=0.0,
        max_value=0.20,
        value=0.0,
        step=0.005,
        format="%.3f",
    )
)


# ---------------------------------------------------------
# CAPITAL
# ---------------------------------------------------------

capital = (
    st.sidebar.number_input(
        "Investment capital",
        min_value=100.0,
        value=10000.0,
        step=1000.0,
        format="%.2f",
    )
)


# ---------------------------------------------------------
# FRACTIONAL SHARES
# ---------------------------------------------------------

st.sidebar.subheader(
    "Quote frazionarie"
)

use_fractional_shares = (
    st.sidebar.toggle(
        "Utilizza quote frazionarie",
        value=False,
    )
)

st.sidebar.info(
    "Attivando questa opzione, il portafoglio può "
    "utilizzare quote frazionarie (es. 3,42 azioni) "
    "per avvicinarsi maggiormente all'allocazione "
    "ottimale. Se disattivata, vengono utilizzate "
    "solo quote intere e il capitale non investito "
    "rimane come liquidità residua."
)


# ---------------------------------------------------------
# DIVERSIFICATION
# ---------------------------------------------------------

st.sidebar.subheader(
    "Diversificazione"
)

use_max_weight = (
    st.sidebar.toggle(
        "Limita peso massimo per asset",
        value=False,
    )
)

if use_max_weight:

    max_weight_percent = (
        st.sidebar.slider(
            "Peso massimo per singolo asset",
            min_value=10,
            max_value=100,
            value=40,
            step=5,
            format="%d%%",
        )
    )

    max_weight = (
        max_weight_percent / 100
    )

    st.sidebar.info(
        f"Ogni singolo asset potrà "
        f"rappresentare al massimo "
        f"il {max_weight:.0%} "
        f"del portafoglio."
    )

else:

    max_weight = 1.0



# =========================================================
# EXPERIMENT LAB — SAVE COMPLETE RUN
# =========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("Experiment Lab")

experiment_name = st.sidebar.text_input(
    "Experiment name",
    placeholder="e.g. 3Y · Dynamic + Fixed · Quarterly",
    key="experiment_name",
)

save_experiment = st.sidebar.button(
    "Save Experiment",
    use_container_width=True,
    type="primary",
)

if save_experiment:
    has_any_result = any(
        st.session_state.get(key) is not None
        for key in (
            "screening_results",
            "optimization_results",
            "factor_model_results",
            "standard_backtest_results",
            "fixed_backtest_results",
        )
    )

    if not has_any_result:
        st.sidebar.error(
            "Run at least one analysis before saving the experiment."
        )
    else:
        try:
            snapshot = build_experiment_snapshot(
                experiment_name=experiment_name,
                market=market,
                selection_mode=selection_mode,
                tickers=tickers,
                risk_free_rate=risk_free_rate,
                capital=capital,
                use_fractional_shares=use_fractional_shares,
                use_max_weight=use_max_weight,
                max_weight=max_weight,
            )

            saved_path = save_experiment_snapshot(snapshot)

            st.session_state["last_saved_experiment"] = {
                "id": snapshot["experiment"]["id"],
                "name": snapshot["experiment"]["name"],
                "path": str(saved_path),
            }

            st.sidebar.success(
                f"Saved: {snapshot['experiment']['name']}"
            )

        except Exception as e:
            st.sidebar.error(
                f"Experiment save failed: {e}"
            )

saved_experiments = list_saved_experiments()

with st.sidebar.expander(
    f"Saved Experiments ({len(saved_experiments)})",
    expanded=False,
):
    if not saved_experiments:
        st.caption("No saved experiments yet.")
    else:
        for saved in saved_experiments[:10]:
            st.markdown(
                f"**{saved['Name']}**  \n"
                f"{saved['Saved At'][:19].replace('T', ' ')} · "
                f"{saved['Assets']} assets"
            )

            saved_file = Path(saved["_path"])
            if saved_file.exists():
                st.download_button(
                    "Download JSON",
                    data=saved_file.read_bytes(),
                    file_name=saved_file.name,
                    mime="application/json",
                    key=f"download_{saved['ID']}",
                    use_container_width=True,
                )

            st.caption(
                " · ".join(
                    [
                        f"Optimization: {saved['Optimization']}",
                        f"Factors: {saved['Factor Models']}",
                        f"Standard BT: {saved['Standard BT']}",
                        f"Fixed BT: {saved['Fixed BT']}",
                    ]
                )
            )


# =========================================================
# MAIN NAVIGATION
# =========================================================

asset_selection_tab, optimization_tab, factor_models_tab, backtest_tab, saved_experiments_tab = st.tabs(
    [
        "Asset Selection",
        "Portfolio Optimization",
        "Factor Models",
        "Backtesting",
        "Saved Experiments",
    ]
)


# =========================================================
# ASSET SELECTION TAB
# =========================================================

with asset_selection_tab:

    st.header("Asset Selection")

    st.caption(
        "Screen the full selected market universe using transparent quantitative "
        "rules before portfolio construction. Invalid or unavailable tickers are "
        "excluded automatically and never stop the analysis."
    )

    st.markdown("### Quantitative Screener")

    st.caption(
        "The current model combines four equally weighted signal families: "
        "Momentum, Risk, Value and Quality. Each available component is converted "
        "to a cross-sectional percentile score from 0 to 100."
    )

    screener_col1, screener_col2, screener_col3 = st.columns(
        [2, 2, 1],
        vertical_alignment="bottom",
    )

    with screener_col1:
        screening_period = st.selectbox(
            "Screening history",
            ["1y", "2y", "5y"],
            index=1,
            key="screening_period",
        )

    with screener_col2:
        max_top_n = max(2, min(30, len(market_instruments)))
        default_top_n = min(10, max_top_n)

        screening_top_n = st.number_input(
            "Number of assets to select",
            min_value=2,
            max_value=max_top_n,
            value=default_top_n,
            step=1,
            key="screening_top_n",
        )

    with screener_col3:
        run_screening = st.button(
            "Run Screening",
            type="primary",
            use_container_width=True,
        )

    st.info(
        "Ticker validation is fault-tolerant: instruments with missing, invalid "
        "or unavailable Yahoo Finance price data are marked as Excluded. "
        "The remaining universe continues to be analysed normally."
    )

    if run_screening:

        with st.spinner(
            f"Screening {len(market_instruments)} instruments..."
        ):
            try:
                screening_results, quant_selected = screen_universe(
                    market_instruments,
                    period=screening_period,
                    top_n=int(screening_top_n),
                    min_components=2,
                )

                st.session_state.screening_results = {
                    "market": market,
                    "period": screening_period,
                    "top_n": int(screening_top_n),
                    "table": screening_results,
                }

                st.session_state.quant_selected_tickers = quant_selected

            except Exception as e:
                st.error(f"Asset screening failed: {e}")

    screening_state = st.session_state.screening_results

    if screening_state is None:
        st.info(
            "Run the screener to rank the current market universe and "
            "automatically select the Top N assets."
        )

    elif screening_state["market"] != market:
        st.warning(
            "The current screening results belong to another market. "
            "Run the screener again for the selected market."
        )

    else:

        screening_table = screening_state["table"].copy()

        selected_count = int(screening_table["Selected"].sum())
        excluded_count = int(
            screening_table["Status"].eq("Excluded").sum()
        )
        eligible_count = int(
            screening_table["Rank"].notna().sum()
        )

        q1, q2, q3 = st.columns(3)

        with q1:
            colored_metric(
                "Selected Assets",
                f"{selected_count}",
                "green",
            )

        with q2:
            colored_metric(
                "Eligible Assets",
                f"{eligible_count}",
                "blue",
            )

        with q3:
            colored_metric(
                "Excluded Assets",
                f"{excluded_count}",
                "red",
            )

        st.markdown("#### Quant Ranking")

        display_columns = [
            "Rank",
            "Instrument",
            "Ticker",
            "Momentum Score",
            "Risk Score",
            "Value Score",
            "Quality Score",
            "Quant Score",
            "Status",
            "Reason",
        ]

        st.dataframe(
            screening_table[display_columns].style.format({
                "Rank": "{:.0f}",
                "Momentum Score": "{:.1f}",
                "Risk Score": "{:.1f}",
                "Value Score": "{:.1f}",
                "Quality Score": "{:.1f}",
                "Quant Score": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        selected_rows = screening_table[
            screening_table["Selected"]
        ].copy()

        if not selected_rows.empty:
            st.markdown("#### Selected Universe")
            st.caption(
                "These instruments become the active universe when "
                "'Quant Selection' is enabled in the sidebar."
            )

            st.dataframe(
                selected_rows[
                    [
                        "Rank",
                        "Instrument",
                        "Ticker",
                        "Quant Score",
                    ]
                ].style.format({
                    "Rank": "{:.0f}",
                    "Quant Score": "{:.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        excluded_rows = screening_table[
            screening_table["Status"].eq("Excluded")
        ]

        if not excluded_rows.empty:
            with st.expander(
                f"Excluded instruments ({len(excluded_rows)})"
            ):
                st.dataframe(
                    excluded_rows[
                        [
                            "Instrument",
                            "Ticker",
                            "Reason",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        st.caption(
            "Scoring methodology: Momentum uses 12-1 momentum when enough "
            "history is available (with shorter-history fallbacks); Risk rewards "
            "lower volatility and less severe drawdowns; Value uses Earnings "
            "Yield and Book-to-Market; Quality uses ROE and lower leverage. "
            "Missing fundamental fields are not imputed: the composite score "
            "uses only available signal families, with at least two required."
        )


# =========================================================
# PORTFOLIO OPTIMIZATION TAB
# =========================================================

with optimization_tab:

    st.header(
        "Portfolio Optimization"
    )

    st.caption(
        "Estimate the optimal portfolio allocation "
        "today using historical returns and risk."
    )

    optimization_control_col1, optimization_control_col2 = (
        st.columns(
            [3, 1],
            vertical_alignment="bottom",
        )
    )

    with optimization_control_col1:

        optimization_period = (
            st.selectbox(
                "Historical estimation period",
                [f"{year}y" for year in range(1, 11)],
                index=4,
                key="optimization_period",
                help=(
                    "Historical data used to estimate "
                    "expected returns and covariance "
                    "for today's optimal portfolio."
                ),
            )
        )

    with optimization_control_col2:

        run_optimization = (
            st.button(
                "Run Portfolio Optimization",
                type="primary",
                use_container_width=True,
            )
        )


    # =====================================================
    # RUN OPTIMIZATION
    # =====================================================

    if run_optimization:

        if validate_portfolio(
            tickers,
            max_weight,
        ):

            try:

                with st.spinner(
                    "Running portfolio optimization..."
                ):

                    prices = get_prices(
                        tickers,
                        period=optimization_period,
                    )

                    returns = calculate_returns(
                        prices
                    )

                    expected_returns = (
                        annualized_returns(
                            returns
                        )
                    )

                    volatility = (
                        annualized_volatility(
                            returns
                        )
                    )

                    covariance = (
                        covariance_matrix(
                            returns
                        )
                    )

                    min_weights = (
                        optimize_minimum_volatility(
                            expected_returns,
                            covariance,
                            max_weight=max_weight,
                        )
                    )

                    max_weights = (
                        optimize_maximum_sharpe(
                            expected_returns,
                            covariance,
                            risk_free_rate,
                            max_weight=max_weight,
                        )
                    )

                    risk_parity_weights = (
                        optimize_risk_parity(
                            covariance,
                            max_weight=max_weight,
                        )
                    )

                    risk_parity_contributions = (
                        risk_contributions(
                            risk_parity_weights,
                            covariance,
                        )
                    )

                    hrp_weights = (
                        optimize_hrp(
                            covariance,
                            max_weight=max_weight,
                        )
                    )

                    hrp_order = hrp_cluster_order(
                        covariance
                    )

                    min_return = portfolio_return(
                        min_weights.values,
                        expected_returns,
                    )

                    min_volatility = (
                        portfolio_volatility(
                            min_weights.values,
                            covariance,
                        )
                    )

                    min_sharpe = (
                        (
                            min_return
                            - risk_free_rate
                        )
                        / min_volatility
                        if min_volatility > 0
                        else 0
                    )

                    max_return = portfolio_return(
                        max_weights.values,
                        expected_returns,
                    )

                    max_volatility = (
                        portfolio_volatility(
                            max_weights.values,
                            covariance,
                        )
                    )

                    max_sharpe = (
                        (
                            max_return
                            - risk_free_rate
                        )
                        / max_volatility
                        if max_volatility > 0
                        else 0
                    )

                    risk_parity_return = portfolio_return(
                        risk_parity_weights.values,
                        expected_returns,
                    )

                    risk_parity_volatility = (
                        portfolio_volatility(
                            risk_parity_weights.values,
                            covariance,
                        )
                    )

                    risk_parity_sharpe = (
                        (
                            risk_parity_return
                            - risk_free_rate
                        )
                        / risk_parity_volatility
                        if risk_parity_volatility > 0
                        else 0
                    )

                    hrp_return = portfolio_return(
                        hrp_weights.values,
                        expected_returns,
                    )

                    hrp_volatility = (
                        portfolio_volatility(
                            hrp_weights.values,
                            covariance,
                        )
                    )

                    hrp_sharpe = (
                        (hrp_return - risk_free_rate)
                        / hrp_volatility
                        if hrp_volatility > 0
                        else 0
                    )

                    frontier = efficient_frontier(
                        expected_returns,
                        covariance,
                        max_weight=max_weight,
                        points=100,
                    )

                    min_table = (
                        prepare_portfolio_table(
                            min_weights,
                            prices,
                            capital,
                            use_fractional_shares,
                        )
                    )

                    max_table = (
                        prepare_portfolio_table(
                            max_weights,
                            prices,
                            capital,
                            use_fractional_shares,
                        )
                    )

                    risk_parity_table = (
                        prepare_portfolio_table(
                            risk_parity_weights,
                            prices,
                            capital,
                            use_fractional_shares,
                        )
                    )

                    hrp_table = (
                        prepare_portfolio_table(
                            hrp_weights,
                            prices,
                            capital,
                            use_fractional_shares,
                        )
                    )

                    min_total_invested = (
                        min_table[
                            "Invested"
                        ].sum()
                    )

                    min_cash = (
                        capital
                        - min_total_invested
                    )

                    max_total_invested = (
                        max_table[
                            "Invested"
                        ].sum()
                    )

                    max_cash = (
                        capital
                        - max_total_invested
                    )

                    risk_parity_total_invested = (
                        risk_parity_table[
                            "Invested"
                        ].sum()
                    )

                    risk_parity_cash = (
                        capital
                        - risk_parity_total_invested
                    )

                    hrp_total_invested = (
                        hrp_table["Invested"].sum()
                    )

                    hrp_cash = (
                        capital - hrp_total_invested
                    )

                    st.session_state.optimization_results = {
                        "period":
                            optimization_period,

                        "capital":
                            capital,

                        "risk_free_rate":
                            risk_free_rate,

                        "use_fractional_shares":
                            use_fractional_shares,

                        "prices":
                            prices,

                        "expected_returns":
                            expected_returns,

                        "volatility":
                            volatility,

                        "covariance":
                            covariance,

                        "min_weights":
                            min_weights,

                        "max_weights":
                            max_weights,

                        "risk_parity_weights":
                            risk_parity_weights,

                        "risk_parity_contributions":
                            risk_parity_contributions,

                        "hrp_weights":
                            hrp_weights,

                        "hrp_order":
                            hrp_order,

                        "min_return":
                            min_return,

                        "min_volatility":
                            min_volatility,

                        "min_sharpe":
                            min_sharpe,

                        "max_return":
                            max_return,

                        "max_volatility":
                            max_volatility,

                        "max_sharpe":
                            max_sharpe,

                        "risk_parity_return":
                            risk_parity_return,

                        "risk_parity_volatility":
                            risk_parity_volatility,

                        "risk_parity_sharpe":
                            risk_parity_sharpe,

                        "hrp_return":
                            hrp_return,

                        "hrp_volatility":
                            hrp_volatility,

                        "hrp_sharpe":
                            hrp_sharpe,

                        "frontier":
                            frontier,

                        "min_table":
                            min_table,

                        "max_table":
                            max_table,

                        "risk_parity_table":
                            risk_parity_table,

                        "hrp_table":
                            hrp_table,

                        "min_total_invested":
                            min_total_invested,

                        "min_cash":
                            min_cash,

                        "max_total_invested":
                            max_total_invested,

                        "max_cash":
                            max_cash,

                        "risk_parity_total_invested":
                            risk_parity_total_invested,

                        "risk_parity_cash":
                            risk_parity_cash,

                        "hrp_total_invested":
                            hrp_total_invested,

                        "hrp_cash":
                            hrp_cash,
                    }

            except Exception as e:

                st.error(
                    f"Portfolio optimization failed: {e}"
                )


    # =====================================================
    # OPTIMIZATION RESULTS
    # =====================================================

    optimization_results = (
        st.session_state.optimization_results
    )

    if optimization_results is None:

        st.info(
            "Select the instruments and parameters, "
            "then click **Run Portfolio Optimization**."
        )

    else:

        r = optimization_results

        st.divider()

        st.caption(
            f"Optimal allocation estimated using "
            f"{r['period']} of historical data."
        )

        min_column, max_column = (
            st.columns(2)
        )


        # =================================================
        # MINIMUM VOLATILITY
        # =================================================

        with min_column:

            st.subheader(
                "Minimum Volatility"
            )

            metric_col1, metric_col2, metric_col3 = (
                st.columns(3)
            )

            with metric_col1:

                colored_metric(
                    "Expected Return",
                    f"{r['min_return']:.2%}",
                    "green",
                )

            with metric_col2:

                colored_metric(
                    "Volatility",
                    f"{r['min_volatility']:.2%}",
                    "red",
                )

            with metric_col3:

                colored_metric(
                    "Sharpe Ratio",
                    f"{r['min_sharpe']:.2f}",
                    "blue",
                )

            portfolio_allocation_title()

            display_portfolio_table(
                r["min_table"],
                r["use_fractional_shares"],
            )

            summary_col1, summary_col2 = (
                st.columns(2)
            )

            with summary_col1:

                colored_metric(
                    "Total Invested",
                    f"€{r['min_total_invested']:,.2f}",
                    "purple",
                )

            with summary_col2:

                colored_metric(
                    "Residual Cash",
                    f"€{r['min_cash']:,.2f}",
                    "yellow",
                )


        # =================================================
        # MAXIMUM SHARPE
        # =================================================

        with max_column:

            st.subheader(
                "Maximum Sharpe"
            )

            metric_col1, metric_col2, metric_col3 = (
                st.columns(3)
            )

            with metric_col1:

                colored_metric(
                    "Expected Return",
                    f"{r['max_return']:.2%}",
                    "green",
                )

            with metric_col2:

                colored_metric(
                    "Volatility",
                    f"{r['max_volatility']:.2%}",
                    "red",
                )

            with metric_col3:

                colored_metric(
                    "Sharpe Ratio",
                    f"{r['max_sharpe']:.2f}",
                    "blue",
                )

            portfolio_allocation_title()

            display_portfolio_table(
                r["max_table"],
                r["use_fractional_shares"],
            )

            summary_col1, summary_col2 = (
                st.columns(2)
            )

            with summary_col1:

                colored_metric(
                    "Total Invested",
                    f"€{r['max_total_invested']:,.2f}",
                    "purple",
                )

            with summary_col2:

                colored_metric(
                    "Residual Cash",
                    f"€{r['max_cash']:,.2f}",
                    "yellow",
                )


        # =================================================
        # RISK PARITY
        # =================================================

        st.divider()

        st.subheader(
            "Risk Parity"
        )

        st.caption(
            "Equal Risk Contribution portfolio. Capital weights "
            "are selected so that each asset contributes as evenly "
            "as possible to total portfolio risk."
        )

        (
            risk_metric_col1,
            risk_metric_col2,
            risk_metric_col3,
        ) = st.columns(3)

        with risk_metric_col1:

            colored_metric(
                "Expected Return",
                f"{r['risk_parity_return']:.2%}",
                "green",
            )

        with risk_metric_col2:

            colored_metric(
                "Volatility",
                f"{r['risk_parity_volatility']:.2%}",
                "red",
            )

        with risk_metric_col3:

            colored_metric(
                "Sharpe Ratio",
                f"{r['risk_parity_sharpe']:.2f}",
                "blue",
            )

        portfolio_allocation_title()

        display_portfolio_table(
            r["risk_parity_table"],
            r["use_fractional_shares"],
        )

        risk_summary_col1, risk_summary_col2 = (
            st.columns(2)
        )

        with risk_summary_col1:

            colored_metric(
                "Total Invested",
                f"€{r['risk_parity_total_invested']:,.2f}",
                "purple",
            )

        with risk_summary_col2:

            colored_metric(
                "Residual Cash",
                f"€{r['risk_parity_cash']:,.2f}",
                "yellow",
            )

        st.markdown(
            "#### Risk Contribution"
        )

        risk_contribution_table = pd.DataFrame(
            {
                "Instrument": [
                    get_company_name(ticker)
                    for ticker
                    in r["risk_parity_contributions"].index
                ],
                "Portfolio Weight":
                    r["risk_parity_weights"].values,
                "Risk Contribution":
                    r["risk_parity_contributions"].values,
            }
        )

        risk_contribution_table = (
            risk_contribution_table
            .sort_values(
                "Risk Contribution",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        st.dataframe(
            risk_contribution_table.style.format(
                {
                    "Portfolio Weight": "{:.2%}",
                    "Risk Contribution": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


        # =================================================
        # HIERARCHICAL RISK PARITY
        # =================================================

        st.divider()

        st.subheader(
            "Hierarchical Risk Parity (HRP)"
        )

        st.caption(
            "HRP groups assets according to their correlation structure "
            "and allocates capital recursively across the resulting "
            "clusters. It does not use expected returns to determine weights."
        )

        hrp_metric_col1, hrp_metric_col2, hrp_metric_col3 = (
            st.columns(3)
        )

        with hrp_metric_col1:
            colored_metric(
                "Expected Return",
                f"{r['hrp_return']:.2%}",
                "green",
            )

        with hrp_metric_col2:
            colored_metric(
                "Volatility",
                f"{r['hrp_volatility']:.2%}",
                "red",
            )

        with hrp_metric_col3:
            colored_metric(
                "Sharpe Ratio",
                f"{r['hrp_sharpe']:.2f}",
                "blue",
            )

        portfolio_allocation_title()

        display_portfolio_table(
            r["hrp_table"],
            r["use_fractional_shares"],
        )

        hrp_summary_col1, hrp_summary_col2 = st.columns(2)

        with hrp_summary_col1:
            colored_metric(
                "Total Invested",
                f"€{r['hrp_total_invested']:,.2f}",
                "purple",
            )

        with hrp_summary_col2:
            colored_metric(
                "Residual Cash",
                f"€{r['hrp_cash']:,.2f}",
                "yellow",
            )

        st.markdown("#### Hierarchical Cluster Order")

        hrp_cluster_table = pd.DataFrame(
            {
                "Cluster Order": range(1, len(r["hrp_order"]) + 1),
                "Instrument": [
                    get_company_name(ticker)
                    for ticker in r["hrp_order"]
                ],
                "Portfolio Weight": [
                    r["hrp_weights"].loc[ticker]
                    for ticker in r["hrp_order"]
                ],
            }
        )

        st.dataframe(
            hrp_cluster_table.style.format(
                {"Portfolio Weight": "{:.2%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )


        # =================================================
        # EFFICIENT FRONTIER
        # =================================================

        st.divider()

        st.header(
            "Efficient Frontier"
        )

        fig = plot_efficient_frontier(
            r["frontier"],
            r["expected_returns"],
            r["covariance"],
            r["min_weights"],
            r["max_weights"],
            r["risk_free_rate"],
        )

        # Assi visibili, coerenti con i grafici CAPM.
        fig.update_xaxes(
            title_text="Volatility",
            tickformat=".0%",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.45)",
            linewidth=1,
            zeroline=False,
        )
        fig.update_yaxes(
            title_text="Expected Return",
            tickformat=".1%",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.45)",
            linewidth=1,
            zeroline=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


        # =================================================
        # ASSET STATISTICS
        # =================================================

        st.header(
            "Asset Statistics"
        )

        asset_statistics = pd.DataFrame(
            {
                "Expected Return":
                    r["expected_returns"],

                "Volatility":
                    r["volatility"],
            }
        )

        asset_statistics.insert(
            0,
            "Instrument",
            [
                get_company_name(ticker)
                for ticker
                in asset_statistics.index
            ],
        )

        asset_statistics = (
            asset_statistics
            .sort_values(
                "Expected Return",
                ascending=False,
            )
        )

        st.dataframe(
            asset_statistics.style.format(
                {
                    "Expected Return":
                        "{:.2%}",

                    "Volatility":
                        "{:.2%}",
                }
            ),
            use_container_width=True,
        )


# =========================================================
# FACTOR MODELS TAB
# =========================================================

with factor_models_tab:

    st.header("Factor Models")
    st.caption(
        "Estimate expected returns with factor models and feed them "
        "into the same portfolio optimization engine."
    )

    st.subheader("Capital Asset Pricing Model (CAPM)")
    st.caption(
        "Market proxy: FTSE MIB. CAPM estimates each asset's beta, "
        "alpha and expected return relative to the market, then uses "
        "those expected returns in a Maximum Sharpe portfolio."
    )

    capm_col1, capm_col2 = st.columns([3, 1], vertical_alignment="bottom")

    with capm_col1:
        capm_period = st.selectbox(
            "Historical estimation period",
            ["1y", "2y", "5y", "10y"],
            index=2,
            key="capm_period",
        )

    with capm_col2:
        run_capm = st.button(
            "Run CAPM Analysis",
            type="primary",
            use_container_width=True,
        )

    if run_capm and validate_portfolio(tickers, max_weight):
        try:
            with st.spinner("Running CAPM analysis..."):
                capm_prices = get_prices(tickers, period=capm_period)
                market_prices = get_prices([FTSE_MIB_TICKER], period=capm_period)

                asset_returns = calculate_returns(capm_prices)
                market_returns = calculate_returns(market_prices).iloc[:, 0]

                common_index = asset_returns.index.intersection(market_returns.index)
                asset_returns = asset_returns.loc[common_index]
                market_returns = market_returns.loc[common_index]

                capm_stats = calculate_capm(
                    asset_returns,
                    market_returns,
                    risk_free_rate=risk_free_rate,
                )

                capm_mu = capm_stats["CAPM Expected Return"]
                capm_covariance = covariance_matrix(asset_returns)

                capm_weights = optimize_maximum_sharpe(
                    capm_mu,
                    capm_covariance,
                    risk_free_rate=risk_free_rate,
                    max_weight=max_weight,
                )

                # Frontiera efficiente costruita usando i rendimenti
                # attesi impliciti del CAPM e la covarianza storica.
                capm_min_weights = optimize_minimum_volatility(
                    capm_mu,
                    capm_covariance,
                    max_weight=max_weight,
                )

                capm_frontier = efficient_frontier(
                    capm_mu,
                    capm_covariance,
                    max_weight=max_weight,
                    points=100,
                )

                capm_portfolio_return = portfolio_return(
                    capm_weights.values, capm_mu
                )
                capm_portfolio_volatility = portfolio_volatility(
                    capm_weights.values, capm_covariance
                )
                capm_portfolio_sharpe = (
                    (capm_portfolio_return - risk_free_rate)
                    / capm_portfolio_volatility
                    if capm_portfolio_volatility > 0 else 0.0
                )

                capm_table = prepare_portfolio_table(
                    capm_weights, capm_prices, capital, use_fractional_shares
                )
                capm_total_invested = capm_table["Invested"].sum()

                st.session_state.factor_model_results = {
                    "period": capm_period,
                    "stats": capm_stats,
                    "expected_returns": capm_mu,
                    "covariance": capm_covariance,
                    "frontier": capm_frontier,
                    "min_weights": capm_min_weights,
                    "weights": capm_weights,
                    "prices": capm_prices,
                    "table": capm_table,
                    "return": capm_portfolio_return,
                    "volatility": capm_portfolio_volatility,
                    "sharpe": capm_portfolio_sharpe,
                    "total_invested": capm_total_invested,
                    "cash": capital - capm_total_invested,
                    "use_fractional_shares": use_fractional_shares,
                    "market_expected_return": float(market_returns.mean() * 252),
                    "asset_returns": asset_returns,
                    "market_returns": market_returns,
                }
        except Exception as e:
            st.error(f"CAPM analysis failed: {e}")

    capm_result = st.session_state.factor_model_results

    if capm_result is None:
        st.info(
            "Select the instruments and parameters, then click "
            "**Run CAPM Analysis**."
        )
    else:
        cr = capm_result
        st.divider()
        st.caption(
            f"CAPM estimated using {cr['period']} of historical data "
            "and FTSE MIB as the market proxy."
        )


        st.subheader("CAPM Maximum Sharpe Portfolio")
        st.caption(
            "Maximum Sharpe allocation using CAPM expected returns "
            "instead of historical mean returns."
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            colored_metric("Expected Return", f"{cr['return']:.2%}", "green")
        with m2:
            colored_metric("Volatility", f"{cr['volatility']:.2%}", "red")
        with m3:
            colored_metric("Sharpe Ratio", f"{cr['sharpe']:.2f}", "blue")

        portfolio_allocation_title()
        display_portfolio_table(cr["table"], cr["use_fractional_shares"])

        s1, s2 = st.columns(2)
        with s1:
            colored_metric(
                "Total Invested", f"€{cr['total_invested']:,.2f}", "purple"
            )
        with s2:
            colored_metric("Residual Cash", f"€{cr['cash']:,.2f}", "yellow")


        st.divider()
        stats_display = cr["stats"].copy()
        stats_display.insert(
            0,
            "Instrument",
            [get_company_name(t) for t in stats_display.index],
        )
        st.markdown("#### CAPM Asset Statistics")
        st.dataframe(
            stats_display.style.format({
                "Beta": "{:.2f}",
                "Alpha": "{:.2%}",
                "R Squared": "{:.2%}",
                "Market Correlation": "{:.2f}",
                "Historical Return": "{:.2%}",
                "CAPM Expected Return": "{:.2%}",
            }),
            use_container_width=True,
        )

        # =================================================
        # CAPM REGRESSION
        # =================================================

        st.markdown("#### CAPM Regression")

        st.caption(
            "Each point represents one trading day. The horizontal axis shows "
            "the FTSE MIB excess return and the vertical axis shows the selected "
            "asset's excess return. The fitted line is the CAPM regression: "
            "its slope is Beta and its intercept is Alpha."
        )

        regression_ticker = st.selectbox(
            "Select an asset",
            options=list(cr["asset_returns"].columns),
            format_func=get_company_name,
            key="capm_regression_asset",
        )

        # Il CAPM è stimato sui rendimenti in eccesso.
        # Convertiamo quindi il risk-free annuale in giornaliero.
        daily_risk_free_rate = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0

        regression_x = (
            cr["market_returns"] - daily_risk_free_rate
        ).rename("Market Excess Return")

        regression_y = (
            cr["asset_returns"][regression_ticker] - daily_risk_free_rate
        ).rename("Asset Excess Return")

        regression_data = pd.concat(
            [regression_x, regression_y],
            axis=1,
        ).dropna()

        regression_alpha_annual = float(
            cr["stats"].loc[regression_ticker, "Alpha"]
        )
        regression_beta = float(
            cr["stats"].loc[regression_ticker, "Beta"]
        )
        regression_r_squared = float(
            cr["stats"].loc[regression_ticker, "R Squared"]
        )

        # calculate_capm espone Alpha annualizzato.
        # Per disegnare la retta sui rendimenti giornalieri
        # riconvertiamo l'intercetta alla scala giornaliera.
        regression_alpha_daily = regression_alpha_annual / 252.0

        x_min = float(regression_data["Market Excess Return"].min())
        x_max = float(regression_data["Market Excess Return"].max())

        regression_line_x = [x_min, x_max]
        regression_line_y = [
            regression_alpha_daily + regression_beta * x_min,
            regression_alpha_daily + regression_beta * x_max,
        ]

        regression_fig = go.Figure()

        regression_fig.add_trace(
            go.Scatter(
                x=regression_data["Market Excess Return"],
                y=regression_data["Asset Excess Return"],
                mode="markers",
                name="Daily Returns",
                marker=dict(
                    size=6,
                    opacity=0.55,
                ),
                customdata=regression_data.index.strftime("%d/%m/%Y"),
                hovertemplate=(
                    "Date: %{customdata}"
                    "<br>FTSE MIB Excess Return: %{x:.2%}"
                    "<br>Asset Excess Return: %{y:.2%}"
                    "<extra></extra>"
                ),
            )
        )

        regression_fig.add_trace(
            go.Scatter(
                x=regression_line_x,
                y=regression_line_y,
                mode="lines",
                name="CAPM Regression",
                hovertemplate=(
                    "FTSE MIB Excess Return: %{x:.2%}"
                    "<br>Fitted Asset Excess Return: %{y:.2%}"
                    "<extra>CAPM Regression</extra>"
                ),
            )
        )

        regression_fig.update_layout(
            hovermode="closest",
            legend_title="Series",
        )

        regression_fig.update_xaxes(
            title_text="FTSE MIB Excess Return",
            tickformat=".1%",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.45)",
            linewidth=1,
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.18)",
        )

        regression_fig.update_yaxes(
            title_text=f"{get_company_name(regression_ticker)} Excess Return",
            tickformat=".1%",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.45)",
            linewidth=1,
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.18)",
        )



        st.plotly_chart(
            regression_fig,
            use_container_width=True,
            key="capm_regression_chart",
        )

        regression_col1, regression_col2, regression_col3 = st.columns(3)

        regression_col1.metric(
            "Beta",
            f"{regression_beta:.2f}",
        )

        regression_col2.metric(
            "Alpha (annualized)",
            f"{regression_alpha_annual:.2%}",
        )

        regression_col3.metric(
            "R²",
            f"{regression_r_squared:.2%}",
        )

        st.divider()

        st.markdown("#### Security Market Line")
        st.caption(
            "The Security Market Line (SML) shows the relationship between "
            "systematic risk, measured by Beta, and the expected return implied "
            "by CAPM. Assets with higher Beta require a higher expected return "
            "as compensation for greater exposure to market risk."
        )

        sml_stats = cr["stats"].sort_values("Beta")
        beta_min = min(0.0, float(sml_stats["Beta"].min()))
        beta_max = max(1.0, float(sml_stats["Beta"].max()))
        beta_line = pd.Series(
            [beta_min + (beta_max - beta_min) * i / 100 for i in range(101)]
        )
        sml_return = (
            risk_free_rate
            + beta_line * (cr["market_expected_return"] - risk_free_rate)
        )
        sml_fig = go.Figure()
        sml_fig.add_trace(go.Scatter(
            x=beta_line, y=sml_return, mode="lines", name="Security Market Line"
        ))
        sml_fig.add_trace(go.Scatter(
            x=sml_stats["Beta"],
            y=sml_stats["CAPM Expected Return"],
            mode="markers+text",
            text=[get_company_name(t) for t in sml_stats.index],
            textposition="top center",
            name="Assets",
        ))
        sml_fig.update_layout(
            hovermode="closest",
        )
        sml_fig.update_xaxes(
            title_text="Beta",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.45)",
            linewidth=1,
            zeroline=False,
        )
        sml_fig.update_yaxes(
            title_text="Expected Return",
            tickformat=".1%",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.45)",
            linewidth=1,
            zeroline=False,
        )
        st.plotly_chart(sml_fig, use_container_width=True)

        # =================================================
        # CAPM EFFICIENT FRONTIER + CAPITAL MARKET LINE
        # =================================================

        st.divider()
        st.markdown("#### CAPM Efficient Frontier & Capital Market Line")
        st.caption(
            "The Efficient Frontier represents the portfolios offering the "
            "highest expected return for each level of volatility. The Capital "
            "Market Line (CML) starts from the risk-free asset and is tangent "
            "to the frontier at the CAPM Maximum Sharpe portfolio, representing "
            "combinations of the risk-free asset and the tangency portfolio."
        )

        capm_frontier_fig = plot_efficient_frontier(
            cr["frontier"],
            cr["expected_returns"],
            cr["covariance"],
            cr["min_weights"],
            cr["weights"],
            risk_free_rate,
        )

        # Il Maximum Sharpe della frontiera CAPM è il portafoglio
        # di tangenza della Capital Market Line.
        tangency_volatility = cr["volatility"]
        tangency_return = cr["return"]

        # Estendiamo la CML oltre il portafoglio di tangenza, come nella
        # rappresentazione teorica classica. L'estensione è puramente
        # grafica e non modifica i pesi del portafoglio.
        frontier_max_volatility = max(
            float(cr["frontier"]["volatility"].max()),
            float(cr["expected_returns"].index.to_series().map(
                lambda ticker: cr["covariance"].loc[ticker, ticker] ** 0.5
            ).max()),
            float(tangency_volatility),
        )
        cml_max_volatility = frontier_max_volatility * 1.08

        if tangency_volatility > 0:
            cml_slope = (
                (tangency_return - risk_free_rate)
                / tangency_volatility
            )
            cml_end_return = (
                risk_free_rate
                + cml_slope * cml_max_volatility
            )

            capm_frontier_fig.add_trace(
                go.Scatter(
                    x=[0.0, cml_max_volatility],
                    y=[risk_free_rate, cml_end_return],
                    mode="lines",
                    name="Capital Market Line",
                    line=dict(
                        color="rgba(56, 189, 248, 0.95)",
                        width=2,
                    ),
                    hovertemplate=(
                        "Capital Market Line<br>"
                        "Volatility: %{x:.2%}<br>"
                        "Expected Return: %{y:.2%}"
                        "<extra></extra>"
                    ),
                )
            )

        # Risk-free asset: volatilità zero e rendimento pari al tasso
        # privo di rischio impostato nella sidebar.
        capm_frontier_fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[risk_free_rate],
                mode="markers+text",
                text=["Risk-Free Asset"],
                textposition="top right",
                name="Risk-Free Asset",
                marker=dict(
                    size=10,
                    symbol="diamond",
                    color="rgba(245, 158, 11, 1.0)",
                ),
                hovertemplate=(
                    "Risk-Free Asset<br>"
                    "Volatility: 0.00%<br>"
                    "Expected Return: %{y:.2%}"
                    "<extra></extra>"
                ),
            )
        )

        # Rinominiamo il punto Maximum Sharpe già prodotto dalla funzione
        # standard per chiarire che qui è il portafoglio CAPM di tangenza.
        for trace in capm_frontier_fig.data:
            if trace.name == "Maximum Sharpe":
                trace.name = "CAPM Maximum Sharpe"

        # Il titolo è già mostrato da Streamlit sopra il grafico:
        # evitiamo quindi di duplicarlo dentro Plotly.
        capm_frontier_fig.update_layout(
            title=None,
            hovermode="closest",
        )
        capm_frontier_fig.update_xaxes(
            title_text="Volatility",
            tickformat=".0%",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.45)",
            linewidth=1,
            zeroline=False,
            rangemode="tozero",
        )
        capm_frontier_fig.update_yaxes(
            title_text="Expected Return",
            tickformat=".1%",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.45)",
            linewidth=1,
            zeroline=False,
        )

        st.plotly_chart(
            capm_frontier_fig,
            use_container_width=True,
        )




    # =====================================================
    # FAMA-FRENCH THREE-FACTOR MODEL (EUROPE)
    # =====================================================
    st.divider()
    st.subheader("Fama–French Three-Factor Model (FF3)")
    st.caption(
        "European daily MKT−RF, SMB and HML factors from the Kenneth French Data Library. "
        "Assets are converted from EUR to USD before estimating factor exposures."
    )
    ff3_col1, ff3_col2 = st.columns([3, 1], vertical_alignment="bottom")
    with ff3_col1:
        ff3_period = st.selectbox(
            "FF3 historical estimation period", ["1y", "2y", "5y", "10y"],
            index=2, key="ff3_period"
        )
    with ff3_col2:
        run_ff3 = st.button("Run FF3 Analysis", use_container_width=True)

    if run_ff3 and validate_portfolio(tickers, max_weight):
        try:
            with st.spinner("Estimating European Fama–French factors..."):
                ff3_eur_prices = get_prices(tickers, period=ff3_period)
                ff3_fx_prices = get_prices(["EURUSD=X"], period=ff3_period)
                ff3_usd_prices = convert_eur_prices_to_usd(
                    ff3_eur_prices, ff3_fx_prices
                )
                ff3_usd_returns = calculate_returns(ff3_usd_prices)
                ff3_factors = load_europe_ff3()
                ff3_factors = ff3_factors.loc[
                    (ff3_factors.index >= ff3_usd_returns.index.min())
                    & (ff3_factors.index <= ff3_usd_returns.index.max())
                ]
                ff3_stats = fit_ff3(ff3_usd_returns, ff3_factors)
                # Approximate EUR expected return using the historical
                # EURUSD arithmetic drift, while keeping EUR covariance.
                fx_daily = ff3_fx_prices.iloc[:, 0].pct_change().dropna()
                aligned_fx = fx_daily.reindex(ff3_usd_returns.index).dropna()
                if len(aligned_fx) < 80:
                    raise ValueError("Not enough aligned FX observations.")
                fx_drift = float(aligned_fx.mean())
                mu_usd_daily = ff3_stats["FF3 Expected Return USD"] / 252.0
                ff3_mu_eur = ((1.0 + mu_usd_daily) / (1.0 + fx_drift) - 1.0) * 252.0
                ff3_mu_eur.name = "FF3 Expected Return EUR"
                ff3_eur_returns = calculate_returns(ff3_eur_prices)
                ff3_cov = covariance_matrix(ff3_eur_returns)
                ff3_mu_eur = ff3_mu_eur.reindex(ff3_cov.index)
                ff3_weights = optimize_maximum_sharpe(
                    ff3_mu_eur, ff3_cov,
                    risk_free_rate=risk_free_rate, max_weight=max_weight
                )
                ff3_return = portfolio_return(ff3_weights.values, ff3_mu_eur)
                ff3_vol = portfolio_volatility(ff3_weights.values, ff3_cov)
                ff3_table = prepare_portfolio_table(
                    ff3_weights, ff3_eur_prices, capital, use_fractional_shares
                )
                st.session_state.ff3_results = {
                    "period": ff3_period, "stats": ff3_stats,
                    "expected_returns_eur": ff3_mu_eur,
                    "weights": ff3_weights, "table": ff3_table,
                    "return": ff3_return, "volatility": ff3_vol,
                    "sharpe": (ff3_return-risk_free_rate)/ff3_vol if ff3_vol>0 else 0.0,
                    "total_invested": float(ff3_table["Invested"].sum()),
                    "use_fractional_shares": use_fractional_shares,
                    "fx_drift_annualized": fx_drift*252,
                    "factor_observations": len(ff3_factors),
                }
        except Exception as exc:
            st.error(f"FF3 analysis failed: {exc}")

    ff3_result = st.session_state.ff3_results
    if ff3_result is not None:
        st.caption(
            f"FF3 estimation: {ff3_result['period']} · "
            f"{ff3_result['factor_observations']} factor dates · "
            "EUR portfolio covariance and EUR risk-free rate."
        )
        st.markdown("#### FF3 Maximum Sharpe Portfolio")
        ff3_m1, ff3_m2, ff3_m3 = st.columns(3)
        with ff3_m1:
            colored_metric("Expected Return", f"{ff3_result['return']:.2%}", "green")
        with ff3_m2:
            colored_metric("Volatility", f"{ff3_result['volatility']:.2%}", "red")
        with ff3_m3:
            colored_metric("Sharpe Ratio", f"{ff3_result['sharpe']:.2f}", "blue")
        portfolio_allocation_title()
        display_portfolio_table(ff3_result["table"], ff3_result["use_fractional_shares"])
        ff3_cash_1, ff3_cash_2 = st.columns(2)
        with ff3_cash_1:
            colored_metric("Total Invested", f"€{ff3_result['total_invested']:,.2f}", "purple")
        with ff3_cash_2:
            colored_metric("Residual Cash", f"€{capital-ff3_result['total_invested']:,.2f}", "yellow")
        st.markdown("#### FF3 Asset Statistics")
        ff3_display = ff3_result["stats"].copy()
        ff3_display.insert(0, "Instrument", [get_company_name(t) for t in ff3_display.index])
        ff3_display["FF3 Expected Return EUR"] = ff3_result["expected_returns_eur"]
        st.dataframe(ff3_display.style.format({
            "Alpha (annualized)": "{:.2%}", "Beta MKT": "{:.3f}",
            "Beta SMB": "{:.3f}", "Beta HML": "{:.3f}",
            "R Squared": "{:.2%}",
            "FF3 Expected Return USD": "{:.2%}",
            "FF3 Expected Return EUR": "{:.2%}",
            "Historical Return USD": "{:.2%}",
        }), use_container_width=True)
        st.caption(
            "Expected returns use historical factor averages, excluding estimated alpha. "
            "The USD-to-EUR conversion uses historical average EUR/USD drift as an "
            "approximation, not an FX forecast. Factor exposures are estimated in USD. "
            "This is an in-sample allocation, not a walk-forward FF3 backtest."
        )
    else:
        st.info("Click **Run FF3 Analysis** to estimate the three-factor model.")


    # =====================================================
    # FAMA-FRENCH FIVE-FACTOR MODEL (EUROPE)
    # =====================================================
    st.divider()
    st.subheader("Fama–French Five-Factor Model (FF5)")
    st.caption(
        "European daily MKT−RF, SMB, HML, RMW and CMA factors. "
        "The same EUR-to-USD conversion, EUR covariance and "
        "Maximum Sharpe optimization used by FF3 are retained."
    )
    ff5_c1, ff5_c2 = st.columns([3, 1], vertical_alignment="bottom")
    with ff5_c1:
        ff5_period = st.selectbox(
            "FF5 historical estimation period", ["1y", "2y", "5y", "10y"],
            index=2, key="ff5_period"
        )
    with ff5_c2:
        run_ff5 = st.button("Run FF5 Analysis", use_container_width=True)

    if run_ff5 and validate_portfolio(tickers, max_weight):
        try:
            with st.spinner("Estimating European FF5 factors..."):
                ff5_eur_prices = get_prices(tickers, period=ff5_period)
                ff5_fx_prices = get_prices(["EURUSD=X"], period=ff5_period)
                ff5_usd_prices = convert_eur_prices_to_usd(
                    ff5_eur_prices, ff5_fx_prices
                )
                ff5_usd_returns = calculate_returns(ff5_usd_prices)
                ff5_factors = load_europe_ff5()
                ff5_factors = ff5_factors.loc[
                    (ff5_factors.index >= ff5_usd_returns.index.min())
                    & (ff5_factors.index <= ff5_usd_returns.index.max())
                ]
                ff5_stats = fit_ff5(ff5_usd_returns, ff5_factors)
                ff5_fx_daily = ff5_fx_prices.iloc[:, 0].pct_change().dropna()
                ff5_aligned_fx = ff5_fx_daily.reindex(ff5_usd_returns.index).dropna()
                if len(ff5_aligned_fx) < 80:
                    raise ValueError("Not enough aligned FX observations.")
                ff5_fx_drift = float(ff5_aligned_fx.mean())
                ff5_mu_usd_daily = ff5_stats["FF5 Expected Return USD"] / 252.0
                ff5_mu_eur = (
                    (1.0 + ff5_mu_usd_daily) / (1.0 + ff5_fx_drift) - 1.0
                ) * 252.0
                ff5_mu_eur.name = "FF5 Expected Return EUR"
                ff5_cov = covariance_matrix(calculate_returns(ff5_eur_prices))
                ff5_mu_eur = ff5_mu_eur.reindex(ff5_cov.index)
                ff5_weights = optimize_maximum_sharpe(
                    ff5_mu_eur, ff5_cov,
                    risk_free_rate=risk_free_rate, max_weight=max_weight
                )
                ff5_return = portfolio_return(ff5_weights.values, ff5_mu_eur)
                ff5_vol = portfolio_volatility(ff5_weights.values, ff5_cov)
                ff5_table = prepare_portfolio_table(
                    ff5_weights, ff5_eur_prices, capital, use_fractional_shares
                )
                st.session_state.ff5_results = {
                    "period": ff5_period, "stats": ff5_stats,
                    "expected_returns_eur": ff5_mu_eur,
                    "weights": ff5_weights, "table": ff5_table,
                    "return": ff5_return, "volatility": ff5_vol,
                    "sharpe": (ff5_return-risk_free_rate)/ff5_vol if ff5_vol>0 else 0.0,
                    "total_invested": float(ff5_table["Invested"].sum()),
                    "use_fractional_shares": use_fractional_shares,
                    "fx_drift_annualized": ff5_fx_drift*252,
                    "factor_observations": len(ff5_factors),
                }
        except Exception as exc:
            st.error(f"FF5 analysis failed: {exc}")

    ff5_result = st.session_state.ff5_results
    if ff5_result is not None:
        st.caption(
            f"FF5 estimation: {ff5_result['period']} · "
            f"{ff5_result['factor_observations']} factor dates · "
            "EUR portfolio covariance and EUR risk-free rate."
        )
        st.markdown("#### FF5 Maximum Sharpe Portfolio")
        ff5_m1, ff5_m2, ff5_m3 = st.columns(3)
        with ff5_m1:
            colored_metric("Expected Return", f"{ff5_result['return']:.2%}", "green")
        with ff5_m2:
            colored_metric("Volatility", f"{ff5_result['volatility']:.2%}", "red")
        with ff5_m3:
            colored_metric("Sharpe Ratio", f"{ff5_result['sharpe']:.2f}", "blue")
        portfolio_allocation_title()
        display_portfolio_table(ff5_result["table"], ff5_result["use_fractional_shares"])
        ff5_cash_1, ff5_cash_2 = st.columns(2)
        with ff5_cash_1:
            colored_metric("Total Invested", f"€{ff5_result['total_invested']:,.2f}", "purple")
        with ff5_cash_2:
            colored_metric("Residual Cash", f"€{capital-ff5_result['total_invested']:,.2f}", "yellow")
        st.markdown("#### FF5 Asset Statistics")
        ff5_display = ff5_result["stats"].copy()
        ff5_display.insert(
            0, "Instrument", [get_company_name(t) for t in ff5_display.index]
        )
        ff5_display["FF5 Expected Return EUR"] = ff5_result["expected_returns_eur"]
        st.dataframe(ff5_display.style.format({
            "Alpha (annualized)": "{:.2%}",
            "Beta MKT": "{:.3f}", "Beta SMB": "{:.3f}",
            "Beta HML": "{:.3f}", "Beta RMW": "{:.3f}",
            "Beta CMA": "{:.3f}", "R Squared": "{:.2%}",
            "FF5 Expected Return USD": "{:.2%}",
            "FF5 Expected Return EUR": "{:.2%}",
            "Historical Return USD": "{:.2%}",
        }), use_container_width=True)
        st.caption(
            "Expected returns use historical factor means, excluding estimated alpha. "
            "FX drift is an approximation, not a forecast. This is an in-sample "
            "allocation, not a walk-forward FF5 backtest."
        )
        ff3_compare = st.session_state.get("ff3_results")
        if ff3_compare is not None:
            st.markdown("#### FF3 vs FF5 — In-sample Comparison")
            st.dataframe(pd.DataFrame({
                "FF3": {
                    "Expected Return": ff3_compare["return"],
                    "Volatility": ff3_compare["volatility"],
                    "Sharpe Ratio": ff3_compare["sharpe"],
                },
                "FF5": {
                    "Expected Return": ff5_result["return"],
                    "Volatility": ff5_result["volatility"],
                    "Sharpe Ratio": ff5_result["sharpe"],
                },
            }).style.format("{:.3f}"), use_container_width=True)
            if ff3_compare["period"] != ff5_result["period"]:
                st.warning(
                    "FF3 and FF5 use different estimation periods. "
                    "Choose matching periods before comparing."
                )
    else:
        st.info("Click **Run FF5 Analysis** to estimate the five-factor model.")


# =========================================================
# BACKTESTING TAB
# =========================================================

with backtest_tab:

    st.header(
        "Backtesting"
    )

    st.caption(
        "Test how the portfolio methodology would have "
        "behaved historically. You can run either "
        "methodology separately or run both together."
    )

    if selection_mode == "Quant Selection":
        st.info(
            "Point-in-time Quant Selection is enabled for backtesting. "
            "Historical screening uses only price information available at each "
            "decision date (Momentum + Risk). Current Value and Quality fundamentals "
            "are intentionally excluded to avoid look-ahead bias."
        )


    # =====================================================
    # BACKTEST CONFIGURATION COLUMNS
    # =====================================================

    standard_col, fixed_col = (
        st.columns(2)
    )


    # =====================================================
    # STANDARD WALK-FORWARD SETTINGS
    # =====================================================

    with standard_col:

        st.subheader(
            "Standard Walk-Forward"
        )

        st.caption(
            "Uses one historical dataset. The first "
            "part is used for estimation and the "
            "remaining observations form the backtest. "
            "The estimation window then rolls forward "
            "at every rebalance."
        )

        run_standard = (
            st.checkbox(
                "Run Standard Walk-Forward",
                value=True,
                key="run_standard",
            )
        )

        standard_period = (
            st.selectbox(
                "Historical dataset",
                [f"{year}y" for year in range(1, 11)],
                index=4,
                key="standard_period",
            )
        )

        standard_dataset_years = int(standard_period[:-1])
        standard_lookback_options = {
            f"{year} year" + ("s" if year != 1 else ""): 252 * year
            for year in range(1, standard_dataset_years)
        }
        if not standard_lookback_options:
            standard_lookback_options = {"6 months": 126}

        standard_lookback_label = (
            st.selectbox(
                "Estimation window",
                list(
                    standard_lookback_options.keys()
                ),
                index=min(1, len(standard_lookback_options) - 1),
                key="standard_lookback",
            )
        )

        standard_lookback_days = (
            standard_lookback_options[
                standard_lookback_label
            ]
        )

        standard_rebalance_options = {
            "Monthly": 21,
            "Quarterly": 63,
            "Semiannual": 126,
            "Annual": 252,
        }

        standard_rebalance_label = (
            st.selectbox(
                "Rebalancing frequency",
                list(
                    standard_rebalance_options.keys()
                ),
                index=1,
                key="standard_rebalance",
            )
        )

        standard_rebalance_days = (
            standard_rebalance_options[
                standard_rebalance_label
            ]
        )

        if selection_mode == "Quant Selection":
            standard_selection_policy_labels = st.multiselect(
                "Asset selection policy",
                [
                    "Fixed at Backtest Start",
                    "Re-screen at Every Rebalance",
                ],
                default=["Fixed at Backtest Start"],
                key="standard_selection_policies",
                help=(
                    "You can run either policy or both. Fixed selects assets "
                    "once at the beginning of the out-of-sample period and "
                    "subsequent rebalances update weights only. Dynamic "
                    "rebuilds the Momentum + Risk ranking at every rebalance, "
                    "so both constituents and weights may change."
                ),
            )
        else:
            standard_selection_policy_labels = ["Manual universe"]

        standard_oos_years = standard_dataset_years - standard_lookback_days / 252
        st.info(
            f"Historical dataset: {standard_dataset_years} years\n\n"
            f"Estimation window: {standard_lookback_label}\n\n"
            f"Out-of-sample: approximately {standard_oos_years:g} years "
            "(rolling estimation at every rebalance)."
        )


    # =====================================================
    # FIXED-HORIZON SETTINGS
    # =====================================================

    with fixed_col:

        st.subheader(
            "Fixed Horizon"
        )

        st.caption(
            "Separates estimation from testing. "
            "For example, 5 years estimation + "
            "1 year backtest asks how the strategy "
            "would have performed if it had started "
            "one year ago using the previous five "
            "years of information."
        )

        run_fixed = (
            st.checkbox(
                "Run Fixed Horizon",
                value=True,
                key="run_fixed",
            )
        )

        fixed_lookback_options = {
            f"{year} year" + ("s" if year != 1 else ""): 252 * year
            for year in range(1, 11)
        }

        fixed_lookback_label = (
            st.selectbox(
                "Estimation window",
                list(
                    fixed_lookback_options.keys()
                ),
                index=3,
                key="fixed_lookback",
            )
        )

        fixed_lookback_days = (
            fixed_lookback_options[
                fixed_lookback_label
            ]
        )

        fixed_backtest_options = {
            f"{year} year" + ("s" if year != 1 else ""): 252 * year
            for year in range(1, 11)
        }

        fixed_backtest_label = (
            st.selectbox(
                "Backtest period",
                list(
                    fixed_backtest_options.keys()
                ),
                index=0,
                key="fixed_backtest",
            )
        )

        fixed_backtest_days = (
            fixed_backtest_options[
                fixed_backtest_label
            ]
        )



        required_years = (
            fixed_lookback_days
            + fixed_backtest_days
        ) / 252

        st.info(
            "Example\n\n"
            f"Estimation: {fixed_lookback_label}\n\n"
            f"Backtest: {fixed_backtest_label}\n\n"
            f"→ approximately "
            f"{required_years:.0f} years "
            f"of historical data required."
        )


    # =====================================================
    # RUN BACKTEST BUTTON
    # =====================================================

    st.write("")

    button_space_left, button_center, button_space_right = (
        st.columns([1, 2, 1])
    )

    with button_center:

        run_backtests = (
            st.button(
                "Run Backtests",
                type="primary",
                use_container_width=True,
            )
        )


    # =====================================================
    # RUN SELECTED BACKTESTS
    # =====================================================

    if run_backtests:

        if not run_standard and not run_fixed:

            st.warning(
                "Select at least one backtest methodology."
            )

        elif validate_portfolio(
            tickers,
            max_weight,
        ):

            shared_ff3_factors = None
            shared_eurusd = None

            # =================================================
            # STANDARD WALK-FORWARD
            # =================================================

            if run_standard:

                try:

                    with st.spinner(
                        "Running Standard Walk-Forward backtest..."
                    ):

                        backtest_selection_top_n = (
                            len(tickers)
                            if selection_mode == "Quant Selection"
                            else None
                        )

                        if selection_mode == "Quant Selection":
                            standard_prices = build_price_matrix(
                                market_instruments,
                                period=standard_period,
                            )
                        else:
                            standard_prices = get_prices(
                                tickers,
                                period=standard_period,
                            )

                        if (
                            len(standard_prices)
                            <= standard_lookback_days + 1
                        ):

                            raise ValueError(
                                "The available historical "
                                "data is too short for the "
                                "selected Standard estimation "
                                "window."
                            )

                        if not standard_selection_policy_labels:
                            raise ValueError(
                                "Select at least one Asset Selection Policy."
                            )

                        standard_results = {}
                        standard_market_prices = get_prices(
                            [FTSE_MIB_TICKER], period=standard_period
                        )
                        # FF3: scarica fattori e FX una sola volta per entrambe le policy.
                        if shared_ff3_factors is None:
                            shared_ff3_factors = load_europe_ff3()
                        if shared_eurusd is None:
                            shared_eurusd = get_prices(["EURUSD=X"], period="10y")
                        standard_ff3_factors = shared_ff3_factors
                        standard_eurusd = shared_eurusd
                        # Un'unica cache per entrambe le politiche e le sette strategie.
                        standard_shared_cache = {}

                        for policy_label in standard_selection_policy_labels:
                            policy = (
                                "dynamic"
                                if policy_label == "Re-screen at Every Rebalance"
                                else "fixed"
                            )

                            standard_result = run_backtest_engine(
                                tickers=tickers,
                                prices=standard_prices,
                                lookback_days=standard_lookback_days,
                                rebalance_days=standard_rebalance_days,
                                risk_free_rate=risk_free_rate,
                                max_weight=max_weight,
                                capital=capital,
                                fixed_backtest_days=None,
                                benchmark_period=standard_period,
                                selection_top_n=backtest_selection_top_n,
                                selection_policy=policy,
                                market_prices=standard_market_prices,
                                shared_window_cache=standard_shared_cache,
                                ff3_factors=standard_ff3_factors,
                                eurusd_prices=standard_eurusd,
                            )

                            try:
                                standard_result["ff3_performance_attribution"] = compute_ff3_attribution(
                                    standard_result["ff3_bt"],
                                    standard_result["ff3_factor_exposure_history"],
                                    standard_ff3_factors, standard_eurusd,
                                )
                            except (ValueError, KeyError, TypeError) as attribution_error:
                                standard_result["ff3_attribution_error"] = str(attribution_error)

                            if policy == "fixed":
                                try:
                                    standard_result["ff3_same_initial_buy_hold"] = (
                                        _build_same_initial_ff3_buy_hold(
                                            standard_prices, standard_result
                                        )
                                    )
                                except ValueError as comparison_error:
                                    standard_result["ff3_comparison_error"] = str(comparison_error)

                            standard_result["lookback_label"] = standard_lookback_label
                            standard_result["period_label"] = "Remaining historical period"
                            standard_result["rebalance_label"] = standard_rebalance_label
                            standard_result["historical_dataset"] = standard_period
                            standard_result["selection_policy"] = policy
                            standard_result["selection_policy_label"] = policy_label

                            standard_results[policy] = standard_result

                        st.session_state["standard_backtest_results"] = standard_results

                except Exception as e:

                    st.error(
                        "Standard Walk-Forward "
                        f"backtest failed: {e}"
                    )


            # =================================================
            # FIXED HORIZON
            # =================================================

            if run_fixed:

                try:

                    with st.spinner(
                        "Running Fixed-Horizon backtest..."
                    ):

                        backtest_selection_top_n = (
                            len(tickers)
                            if selection_mode == "Quant Selection"
                            else None
                        )

                        if selection_mode == "Quant Selection":
                            fixed_prices = build_price_matrix(
                                market_instruments,
                                period="10y",
                            )

                            required_days = (
                                fixed_lookback_days
                                + fixed_backtest_days
                                + 2
                            )
                            if len(fixed_prices) < required_days:
                                raise ValueError(
                                    "Not enough historical universe data for "
                                    "the selected Fixed-Horizon configuration."
                                )
                            fixed_prices = fixed_prices.tail(required_days)
                        else:
                            fixed_prices = get_fixed_horizon_prices(
                                tickers,
                                fixed_lookback_days,
                                fixed_backtest_days,
                            )

                        if shared_ff3_factors is None:
                            shared_ff3_factors = load_europe_ff3()
                        if shared_eurusd is None:
                            shared_eurusd = get_prices(["EURUSD=X"], period="10y")
                        fixed_ff3_factors = shared_ff3_factors
                        fixed_eurusd = shared_eurusd
                        fixed_result = (
                            run_fixed_horizon_engine(
                                tickers=tickers,
                                prices=fixed_prices,
                                estimation_days=
                                    fixed_lookback_days,
                                backtest_days=
                                    fixed_backtest_days,
                                ff3_factors=fixed_ff3_factors,
                                eurusd_prices=fixed_eurusd,
                                risk_free_rate=
                                    risk_free_rate,
                                max_weight=
                                    max_weight,
                                capital=
                                    capital,
                                benchmark_period=
                                    "10y",
                                selection_top_n=
                                    backtest_selection_top_n,
                            )
                        )

                        try:
                            fixed_result["ff3_performance_attribution"] = compute_ff3_attribution(
                                fixed_result["ff3_bt"],
                                fixed_result["ff3_factor_exposure_history"],
                                fixed_ff3_factors, fixed_eurusd,
                            )
                        except (ValueError, KeyError, TypeError) as attribution_error:
                            fixed_result["ff3_attribution_error"] = str(attribution_error)

                        fixed_result[
                            "lookback_label"
                        ] = (
                            fixed_lookback_label
                        )

                        fixed_result[
                            "period_label"
                        ] = (
                            fixed_backtest_label
                        )
                        fixed_result[
                            "rebalance_label"
                        ] = "None (Buy & Hold)"

                        st.session_state[
                            "fixed_backtest_results"
                        ] = fixed_result

                except Exception as e:

                    st.error(
                        "Fixed-Horizon "
                        f"backtest failed: {e}"
                    )


    # =====================================================
    # RESULTS
    # =====================================================

    standard_result = (
        st.session_state[
            "standard_backtest_results"
        ]
    )

    fixed_result = (
        st.session_state[
            "fixed_backtest_results"
        ]
    )


    # =====================================================
    # STANDARD RESULTS
    # =====================================================

    if standard_result is not None:

        # Backward compatibility with results saved by older app versions.
        if "max_sharpe_bt" in standard_result:
            standard_results_to_display = {"fixed": standard_result}
        else:
            standard_results_to_display = standard_result

        for policy_key, policy_result in standard_results_to_display.items():
            policy_label = policy_result.get(
                "selection_policy_label",
                "Fixed at Backtest Start" if policy_key == "fixed"
                else "Re-screen at Every Rebalance",
            )

            display_backtest_results(
                result=policy_result,
                title=f"Standard Walk-Forward Backtest · {policy_label}",
                description=(
                    "The historical dataset contains both the initial "
                    "estimation window and the subsequent out-of-sample "
                    "simulation. At every rebalance the estimation window "
                    "rolls forward using only information available at that "
                    "date. With Fixed selection, constituents are frozen at "
                    "t0 and only weights are rebalanced; with Dynamic "
                    "selection, the Momentum + Risk ranking is recomputed "
                    "at every rebalance."
                ),
                lookback_label=policy_result["lookback_label"],
                backtest_period_label=policy_result["period_label"],
                rebalance_label=policy_result["rebalance_label"],
                result_key=f"standard_{policy_key}",
            )


    # =====================================================
    # FIXED-HORIZON RESULTS
    # =====================================================

    if fixed_result is not None:

        display_backtest_results(
            result=fixed_result,

            title=(
                "Fixed-Horizon Backtest"
            ),

            description=(
                "The estimation period is separated "
                "from the actual out-of-sample test. "
                "The optimal allocation is calculated "
                "once at the beginning of the backtest "
                f"using the previous "
                f"{fixed_result['lookback_label']} of data. "
                "The resulting portfolio is then held "
                "unchanged for the entire "
                f"{fixed_result['period_label']} test period."
            ),

            lookback_label=
                fixed_result[
                    "lookback_label"
                ],

            backtest_period_label=
                fixed_result[
                    "period_label"
                ],

            rebalance_label=
                fixed_result[
                    "rebalance_label"
                ],

            result_key="fixed",
        )


    if standard_result is not None and fixed_result is not None:
        render_backtesting_comparison(standard_result, fixed_result)


    # =====================================================
    # EMPTY STATE
    # =====================================================

    if (
        standard_result is None
        and fixed_result is None
    ):

        st.info(
            "Configure Standard Walk-Forward, "
            "Fixed Horizon, or both, then click "
            "**Run Backtests**."
        )

# =========================================================
# SAVED EXPERIMENTS TAB
# =========================================================

with saved_experiments_tab:

    st.header("Saved Experiments")

    st.caption(
        "Open previously saved quantitative experiments and review the complete "
        "snapshot: Asset Selection, Portfolio Parameters, Portfolio Optimization, "
        "Factor Models and Backtesting."
    )

    saved_experiments_page = list_saved_experiments()

    if not saved_experiments_page:
        st.info(
            "No saved experiments yet. Run the analyses and use "
            "**Save Experiment** in the sidebar."
        )

    else:
        experiment_options = {
            (
                f"{item['Name']} · "
                f"{item['Saved At'][:19].replace('T', ' ')} · "
                f"{item['Assets']} assets"
            ): item
            for item in saved_experiments_page
        }

        selected_experiment_label = st.selectbox(
            "Open experiment",
            options=list(experiment_options.keys()),
            key="saved_experiment_selector",
        )

        selected_experiment = experiment_options[
            selected_experiment_label
        ]

        action_col1, action_col2 = st.columns([1, 4])

        with action_col1:
            if st.button(
                "Delete Experiment",
                key=f"delete_{selected_experiment['ID']}",
                use_container_width=True,
            ):
                try:
                    Path(selected_experiment["_path"]).unlink(missing_ok=True)
                    st.success("Experiment deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

        with action_col2:
            st.caption(
                "The saved snapshot is immutable: opening it does not overwrite "
                "the current live analysis."
            )

        try:
            saved_payload = load_saved_experiment(
                selected_experiment["_path"]
            )
            render_saved_experiment(saved_payload)

        except Exception as e:
            st.error(
                f"Could not open the saved experiment: {e}"
            )

        st.divider()
        st.subheader("Experiment Archive") 

        archive_df = pd.DataFrame(
            [
                {
                    "Name": item["Name"],
                    "Saved At": item["Saved At"][:19].replace("T", " "),
                    "Market": item["Market"],
                    "Selection": item["Selection"],
                    "Assets": item["Assets"],
                    "Optimization": item["Optimization"],
                    "Factor Models": item["Factor Models"],
                    "Standard BT": item["Standard BT"],
                    "Fixed BT": item["Fixed BT"],
                }
                for item in saved_experiments_page
            ]
        )

        st.dataframe(
            archive_df,
            use_container_width=True,
            hide_index=True,
        )
