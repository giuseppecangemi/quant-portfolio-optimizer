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
from src.fama_french_5 import load_europe_ff5, fit_ff5, estimate_ff5_eur_returns
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
from src.monte_carlo import extract_equity as mc_extract_equity, aligned_returns as mc_aligned_returns, simulate as mc_simulate
from src.transaction_costs import PRESETS as TRANSACTION_COST_PRESETS

from src.backtest import (
    backtest_portfolio,
    calculate_backtest_metrics,
)
from src.robustness import run_rolling_robustness
from src.transaction_costs import PRESETS, CostAssumptions
from src.volatility_forecasting import (analyze_volatility, backtest_volatility, load_volatility_prices, ARCH_AVAILABLE, portfolio_log_returns, filtered_garch_volatility, forecast_volatility_path, compare_volatility_models, portfolio_risk_contributions)


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


def _saved_cost_details(result):
    """Capture execution assumptions and trade logs without altering engine results."""
    if not isinstance(result, dict):
        return None
    # Walk-forward may contain one result per selection policy.
    if "max_sharpe_bt" not in result and "transaction_cost_summary" not in result:
        nested = {key: _saved_cost_details(value) for key, value in result.items()
                  if isinstance(value, dict)}
        return {key: value for key, value in nested.items() if value is not None} or None

    assumptions = result.get("transaction_cost_preset")
    if assumptions is not None:
        from dataclasses import asdict, is_dataclass
        assumption_values = asdict(assumptions) if is_dataclass(assumptions) else assumptions
    else:
        assumption_values = None

    curves = {
        "Maximum Sharpe": "max_sharpe_bt", "CAPM": "capm_bt",
        "FF3": "ff3_bt", "FF5": "ff5_bt",
        "Minimum Volatility": "min_vol_bt", "Risk Parity": "risk_parity_bt",
        "HRP": "hrp_bt", "Equal Weight": "equal_weight_bt",
    }
    logs = {}
    for label, key in curves.items():
        curve = result.get(key)
        if isinstance(curve, pd.DataFrame):
            log = curve.attrs.get("transaction_log")
            if isinstance(log, pd.DataFrame) and not log.empty:
                logs[label] = log.copy()

    return {
        "enabled": assumptions is not None,
        "assumptions": assumption_values,
        "summary": result.get("transaction_cost_summary", {}),
        "trade_logs": logs,
    }


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
            "transaction_costs": {
                "standard_walk_forward": _saved_cost_details(standard_state),
                "fixed_horizon": _saved_cost_details(fixed_state),
            },
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


def _render_saved_transaction_costs(result, saved_costs=None):
    """Render costs saved with an experiment; legacy experiments remain readable."""
    details = saved_costs if isinstance(saved_costs, dict) else {}
    summary = details.get("summary") or result.get("transaction_cost_summary", {})
    assumptions = details.get("assumptions")
    if assumptions is None:
        raw = result.get("transaction_cost_preset")
        from dataclasses import asdict, is_dataclass
        if is_dataclass(raw):
            assumptions = asdict(raw)

    st.markdown("##### Transaction Costs — Gross vs Net")
    if assumptions:
        st.caption(
            f"Commission: {float(assumptions.get('commission_rate', 0))*100:.3f}% "
            f"(minimum €{float(assumptions.get('min_commission', 0)):.2f}/order); "
            f"half-spread: {float(assumptions.get('half_spread_bps', 0)):g} bps; "
            f"slippage: {float(assumptions.get('slippage_bps', 0)):g} bps; "
            f"buy tax: {float(assumptions.get('buy_tax_rate', 0))*100:.3f}%."
        )
    elif not summary:
        st.caption("No transaction costs recorded for this experiment (legacy or zero-cost run).")
        return
    else:
        st.caption("Cost parameters were not saved explicitly in this legacy experiment.")

    if isinstance(summary, dict) and summary:
        table = pd.DataFrame.from_dict(summary, orient="index")
        st.dataframe(table.round(2), use_container_width=True)
    logs = details.get("trade_logs", {})
    if isinstance(logs, dict) and logs:
        with st.expander("Transaction orders and rebalance costs"):
            for label, log in logs.items():
                if isinstance(log, pd.DataFrame) and not log.empty:
                    st.markdown(f"**{label}**")
                    st.dataframe(log, use_container_width=True)


def _render_saved_backtest_summary(result, title, saved_costs=None):
    if not isinstance(result, dict):
        st.info(f"No saved {title} results.")
        return

    st.markdown(f"#### {title}")
    _render_saved_transaction_costs(result, saved_costs)

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
        ("FF5 Maximum Sharpe", "ff5_bt_metrics"),
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
        ("FF5 Maximum Sharpe", "ff5_bt"),
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
        saved_costs = backtesting.get("transaction_costs") or {}
        standard_costs = saved_costs.get("standard_walk_forward") or {}
        fixed_costs = saved_costs.get("fixed_horizon")

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
                                standard_costs.get(policy),
                            )
                else:
                    _render_saved_backtest_summary(
                        standard,
                        "Standard Walk-Forward",
                        standard_costs,
                    )

            if isinstance(fixed, dict):
                st.divider()
                _render_saved_backtest_summary(
                    fixed,
                    "Fixed Horizon",
                    fixed_costs,
                )
            if isinstance(standard, dict) and isinstance(fixed, dict):
                render_backtesting_comparison(standard, fixed, key_prefix=f"saved_{meta.get('id', 'experiment')}")


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


def _build_same_initial_model_buy_hold(prices, standard_result, model):
    """Same-inception, same-model weights; no future data in allocation."""
    key = {"capm": "capm_bt", "ff3": "ff3_bt", "ff5": "ff5_bt"}[model]
    equity = _comparison_equity(standard_result, key)
    history = standard_result.get(f"{model}_weights_history")
    if equity is None or not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError(f"Missing {model} equity or initial weights.")
    px = prices.copy()
    px.index = pd.to_datetime(px.index).normalize()
    px = px[~px.index.duplicated(keep="last")].sort_index()
    px = px.loc[(px.index >= equity.index[0]) & (px.index <= equity.index[-1])]
    if not px.index.equals(equity.index):
        raise ValueError("Price calendar differs from the Walk-Forward equity calendar.")
    initial = history.iloc[0].astype(float)
    initial = initial[initial > 1e-10]
    if initial.empty or not initial.index.isin(px.columns).all():
        raise ValueError(f"Initial {model} holdings are missing from the price matrix.")
    asset_prices = px[initial.index].astype(float)
    if asset_prices.isna().any().any() or (asset_prices <= 0).any().any():
        raise ValueError("Missing or invalid initial holding prices.")
    initial = initial / initial.sum()
    quantities = initial / asset_prices.iloc[0]
    values = asset_prices.mul(quantities, axis=1).sum(axis=1)
    return (values / values.iloc[0] * float(equity.iloc[0])).rename("Same-initial Buy & Hold")


def render_capm_ff3_ff5_backtest_chart(standard_results, fixed_result, key_prefix="live"):
    """Compare realized CAPM, FF3 and FF5 returns from existing equity curves."""
    if not isinstance(standard_results, dict) or not isinstance(fixed_result, dict):
        return

    standard = (
        {"fixed": standard_results}
        if "ff3_bt" in standard_results else standard_results
    )
    scenarios = [
        ("WF fisso", standard.get("fixed")),
        ("WF dinamico", standard.get("dynamic")),
        ("Fixed Horizon", fixed_result),
    ]
    labels, capm_values, ff3_values, ff5_values = [], [], [], []
    for label, result in scenarios:
        if not isinstance(result, dict):
            continue
        capm_curve = _comparison_equity(result, "capm_bt")
        ff3_curve = _comparison_equity(result, "ff3_bt")
        ff5_curve = _comparison_equity(result, "ff5_bt")
        if capm_curve is None or ff3_curve is None or ff5_curve is None:
            continue
        labels.append(label)
        capm_values.append(100 * (capm_curve.iloc[-1] / capm_curve.iloc[0] - 1))
        ff3_values.append(100 * (ff3_curve.iloc[-1] / ff3_curve.iloc[0] - 1))
        ff5_values.append(100 * (ff5_curve.iloc[-1] / ff5_curve.iloc[0] - 1))

    if not labels:
        return

    st.markdown("##### CAPM vs FF3 vs FF5 — Backtest Performance")
    st.caption("Realized cumulative returns for each backtest; periods may differ.")
    fig = go.Figure()
    for name, values, color in (
        ("CAPM", capm_values, "#9B59B6"),
        ("FF3", ff3_values, "#28599B"),
        ("FF5", ff5_values, "#388545"),
    ):
        fig.add_trace(go.Bar(
            name=name,
            x=labels,
            y=values,
            marker=dict(color=color, line=dict(width=0)),
            opacity=0.55,
            text=[f"{value:+.2f}%" for value in values],
            textposition="outside",
            hovertemplate="%{x}<br>" + name + ": %{y:+.2f}%<extra></extra>",
        ))
    all_values = capm_values + ff3_values + ff5_values
    fig.update_layout(
        template="plotly_dark",
        barmode="group",
        bargap=0.22,
        bargroupgap=0.04,
        height=460,
        margin=dict(l=20, r=20, t=55, b=35),
        legend=dict(orientation="h", y=1.12, x=0.0),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="Total Return (%)", ticksuffix="%", zeroline=True,
                   gridcolor="rgba(255,255,255,0.08)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB"),
        uniformtext_minsize=11,
        uniformtext_mode="hide",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_capm_ff3_ff5_comparison")


def render_backtesting_comparison(standard_results, fixed_result, key_prefix="live"):
    if not isinstance(standard_results, dict) or not isinstance(fixed_result, dict):
        return
    if "ff3_bt" in standard_results:
        standard_results = {"fixed": standard_results}
    render_capm_ff3_ff5_backtest_chart(standard_results, fixed_result, key_prefix=key_prefix)
    st.subheader("Backtesting Comparison & Diagnostics")
    st.caption("Common-date growth is normalized to 1 EUR. Strategies retain their original weights and optimization rules.")

    model_specs = [("CAPM", "capm_bt", "#9B59B6"),
                   ("FF3", "ff3_bt", "#28599B"),
                   ("FF5", "ff5_bt", "#388545")]
    scenarios = [("Fixed Horizon", fixed_result)] + [
        (f"Walk-Forward {policy.title()}", result)
        for policy, result in standard_results.items() if isinstance(result, dict)
    ]
    curves = {}
    for scenario, result in scenarios:
        for model, key, color in model_specs:
            equity = _comparison_equity(result, key)
            if equity is not None:
                curves[f"{scenario} · {model}"] = (equity, color, scenario)
    if len(curves) >= 2:
        common_dates = None
        for equity, _, _ in curves.values():
            common_dates = equity.index if common_dates is None else common_dates.intersection(equity.index)
        common_dates = common_dates.sort_values()
        if len(common_dates) >= 2:
            growth = pd.DataFrame({
                name: eq.reindex(common_dates) / eq.reindex(common_dates).iloc[0]
                for name, (eq, _, _) in curves.items()
            })
            st.markdown("##### Matched out-of-sample dates — CAPM / FF3 / FF5")
            st.caption(f"Shared period: {common_dates[0]:%Y-%m-%d} to {common_dates[-1]:%Y-%m-%d} "
                       f"({len(common_dates)} observations). Dates are matched; holdings can differ.")
            st.dataframe((growth.iloc[-1] - 1).rename("Common-date return").to_frame()
                         .style.format("{:+.2%}"), use_container_width=True)
            fig = go.Figure()
            for name, (eq, color, scenario) in curves.items():
                fig.add_trace(go.Scatter(
                    x=common_dates, y=growth[name], mode="lines", name=name,
                    line=dict(color=color, width=2,
                              dash="solid" if scenario == "Fixed Horizon" else
                              "dash" if "Dynamic" in scenario else "dot"),
                    opacity=0.8,
                ))
            fig.update_layout(xaxis_title="Date", yaxis_title="Growth of 1 EUR",
                              hovermode="x unified", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_matched_capm_ff3_ff5_growth")
        else:
            st.info("Fewer than two shared trading dates for the matched comparison.")

    if "fixed" in standard_results:
        wf = standard_results["fixed"]
        st.markdown("##### Same initial portfolio: rebalance vs hold — CAPM / FF3 / FF5")
        st.caption("Each Buy & Hold uses its own model's initial weights and start date. "
                   "Fractional shares remain fixed; transaction costs are excluded.")
        rows = []
        fig2 = go.Figure()
        for model, key, color in model_specs:
            wf_equity = _comparison_equity(wf, key)
            cf = wf.get(f"{model.lower()}_same_initial_buy_hold")
            if wf_equity is None or not isinstance(cf, pd.Series):
                continue
            cf = cf.copy()
            cf.index = pd.to_datetime(cf.index).normalize()
            if not cf.index.equals(wf_equity.index) or cf.isna().any() or (cf <= 0).any():
                continue
            wf_ret = wf_equity.iloc[-1] / wf_equity.iloc[0] - 1
            bh_ret = cf.iloc[-1] / cf.iloc[0] - 1
            rows.extend([
                {"Model": model, "Strategy": "Walk-Forward Fixed", "Return": wf_ret,
                 "Start date": wf_equity.index[0].date(), "End date": wf_equity.index[-1].date()},
                {"Model": model, "Strategy": "Same-initial Buy & Hold", "Return": bh_ret,
                 "Start date": wf_equity.index[0].date(), "End date": wf_equity.index[-1].date()},
                {"Model": model, "Strategy": "Difference (pp)", "Return": 100*(wf_ret-bh_ret),
                 "Start date": wf_equity.index[0].date(), "End date": wf_equity.index[-1].date()},
            ])
            for name, equity, dash in (("Walk-Forward", wf_equity, "solid"),
                                       ("Buy & Hold", cf, "dash")):
                fig2.add_trace(go.Scatter(x=equity.index, y=equity/equity.iloc[0],
                                          name=f"{model} {name}", mode="lines",
                                          line=dict(color=color, dash=dash, width=2), opacity=0.85))
        if rows:
            display_rows = pd.DataFrame(rows)
            display_rows["Return"] = [f"{v:+.2f} pp" if strategy == "Difference (pp)"
                                      else f"{v:+.2%}" for v, strategy in
                                      zip(display_rows["Return"], display_rows["Strategy"])]
            st.dataframe(display_rows, use_container_width=True, hide_index=True)
            fig2.update_layout(xaxis_title="Date", yaxis_title="Growth of 1 EUR",
                               hovermode="x unified", template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_same_initial_capm_ff3_ff5_growth")
        else:
            st.info("Same-initial comparisons are not stored for this run. "
                    "Run Walk-Forward Fixed again to populate them.")
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
def _cost_adjusted_metrics(curve, capital, risk_free_rate):
    """Include the initial order cost in annual return, Sharpe and drawdown."""
    m = calculate_backtest_metrics(curve, risk_free_rate)
    if "Transaction Cost" not in curve.columns:
        return m
    values = curve["Portfolio Value"].astype(float)
    m["Total Return"] = float(values.iloc[-1] / capital - 1)
    wealth = np.r_[float(capital), values.to_numpy()]
    m["Max Drawdown"] = float(np.min(wealth / np.maximum.accumulate(wealth) - 1))
    return m


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
    ff5_factors=None,
    transaction_costs=None,
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

    ff5_mu, ff5_stats = estimate_ff5_eur_returns(
        estimation_prices, ff5_factors, eurusd_prices, return_stats=True
    )
    ff5_mu = ff5_mu.reindex(covariance.index)
    ff5_weights = optimize_maximum_sharpe(
        ff5_mu, covariance, risk_free_rate=risk_free_rate, max_weight=max_weight
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

        gross_value = (test_prices * quantities).sum(axis=1)
        portfolio_value = gross_value
        cost_log = pd.DataFrame()
        if transaction_costs is not None:
            from src.transaction_costs import estimate_order_costs
            budget = float(capital)
            for _ in range(30):
                desired_qty = budget * weights / start_prices
                total_cost, detail = estimate_order_costs(
                    desired_qty, start_prices, transaction_costs
                )
                new_budget = capital - total_cost
                if new_budget <= 0:
                    raise ValueError("Initial transaction costs exhaust capital")
                if abs(new_budget - budget) < 1e-8:
                    budget = new_budget
                    break
                budget = new_budget
            quantities = budget * weights / start_prices
            portfolio_value = (test_prices * quantities).sum(axis=1)
            cost_log = pd.DataFrame([{
                "Rebalance Date": test_prices.index[0], **detail,
                "Total Cost": total_cost,
            }])

        backtest = pd.DataFrame(
            {
                "Portfolio Value":
                    portfolio_value
            }
        )

        backtest["Daily Return"] = (
            backtest["Portfolio Value"].pct_change().fillna(0.0)
        )
        if transaction_costs is not None:
            backtest["Gross Portfolio Value"] = gross_value
            backtest["Transaction Cost"] = 0.0
            backtest.loc[backtest.index[0], "Transaction Cost"] = total_cost
            backtest.attrs["transaction_log"] = cost_log
            backtest["Daily Return"] = backtest["Portfolio Value"].pct_change().fillna(0.0)
            backtest.loc[backtest.index[0], "Daily Return"] = budget / capital - 1
        return backtest

    max_sharpe_bt = build_buy_and_hold(
        max_sharpe_weights
    )

    capm_bt = build_buy_and_hold(
        capm_weights
    )

    ff3_bt = build_buy_and_hold(ff3_weights)
    ff5_bt = build_buy_and_hold(ff5_weights)

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


    ff5_weights_history = pd.DataFrame(
        [ff5_weights.reindex(test_prices.columns).fillna(0.0)],
        index=[effective_start],
    )
    ff5_aligned_stats = ff5_stats.reindex(ff5_weights.index)
    ff5_factor_exposure_history = pd.DataFrame([{
        **{label: float((ff5_weights * ff5_aligned_stats[beta]).sum())
           for label, beta in (("Market Beta", "Beta MKT"),
                               ("SMB Exposure", "Beta SMB"),
                               ("HML Exposure", "Beta HML"),
                               ("RMW Exposure", "Beta RMW"),
                               ("CMA Exposure", "Beta CMA"))},
        "Active Assets": int((ff5_weights > 1e-10).sum()),
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
        _cost_adjusted_metrics(max_sharpe_bt, capital, risk_free_rate)
    )

    capm_bt_metrics = (
        _cost_adjusted_metrics(capm_bt, capital, risk_free_rate)
    )

    ff3_bt_metrics = _cost_adjusted_metrics(ff3_bt, capital, risk_free_rate)
    ff5_bt_metrics = _cost_adjusted_metrics(ff5_bt, capital, risk_free_rate)

    min_bt_metrics = (
        _cost_adjusted_metrics(min_vol_bt, capital, risk_free_rate)
    )

    risk_parity_bt_metrics = (
        _cost_adjusted_metrics(risk_parity_bt, capital, risk_free_rate)
    )

    hrp_bt_metrics = (
        _cost_adjusted_metrics(hrp_bt, capital, risk_free_rate)
    )

    equal_bt_metrics = (
        _cost_adjusted_metrics(equal_weight_bt, capital, risk_free_rate)
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

    transaction_cost_summary = {}
    if transaction_costs is not None:
        for name, curve in {
            "Max Sharpe": max_sharpe_bt, "CAPM": capm_bt,
            "FF3": ff3_bt, "FF5": ff5_bt,
            "Min Volatility": min_vol_bt, "Risk Parity": risk_parity_bt,
            "HRP": hrp_bt, "Equal Weight": equal_weight_bt,
        }.items():
            log = curve.attrs.get("transaction_log", pd.DataFrame())
            gross = curve.get("Gross Portfolio Value")
            transaction_cost_summary[name] = {
                "Total Costs (€)": float(log["Total Cost"].sum()) if not log.empty else 0.0,
                "Orders": int(log["Orders"].sum()) if not log.empty else 0,
                "Gross Return (%)": (float(gross.iloc[-1]) / capital - 1) * 100,
                "Net Return (%)": (float(curve["Portfolio Value"].iloc[-1]) / capital - 1) * 100,
            }

    return {
        "transaction_cost_summary": transaction_cost_summary,
        "transaction_cost_preset": transaction_costs,
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
        "ff5_bt": ff5_bt,
        "ff5_weights_history": ff5_weights_history,
        "ff5_bt_metrics": ff5_bt_metrics,
        "ff5_factor_exposure_history": ff5_factor_exposure_history,


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
    ff5_factors=None,
    transaction_costs=None,
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
        transaction_costs=transaction_costs,
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
        transaction_costs=transaction_costs,
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
        transaction_costs=transaction_costs,
        ff3_factors=ff3_factors, eurusd_prices=eurusd_prices,
        ff3_exposure_records=ff3_exposure_records,
    )
    ff3_factor_exposure_history = pd.DataFrame(ff3_exposure_records)
    if not ff3_factor_exposure_history.empty:
        ff3_factor_exposure_history = ff3_factor_exposure_history.set_index("Rebalance Date")

    ff5_exposure_records = []
    ff5_bt, ff5_weights_history = backtest_portfolio(
        prices, strategy="ff5_max_sharpe", initial_capital=capital,
        lookback_days=lookback_days, rebalance_days=rebalance_days,
        risk_free_rate=risk_free_rate, max_weight=max_weight,
        selection_top_n=selection_top_n, selection_policy=selection_policy,
        shared_window_cache=shared_window_cache,
        transaction_costs=transaction_costs,
        ff5_factors=ff5_factors, eurusd_prices=eurusd_prices,
        ff5_exposure_records=ff5_exposure_records,
    )
    ff5_factor_exposure_history = pd.DataFrame(ff5_exposure_records)
    if not ff5_factor_exposure_history.empty:
        ff5_factor_exposure_history = ff5_factor_exposure_history.set_index("Rebalance Date")

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
        transaction_costs=transaction_costs,
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
        transaction_costs=transaction_costs,
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
        transaction_costs=transaction_costs,
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
        transaction_costs=transaction_costs,
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
        ff5_bt = ff5_bt.reindex(max_sharpe_bt.index).dropna()

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
        ff5_weights_history = ff5_weights_history.loc[
            ff5_weights_history.index <= effective_end_for_weights
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
        _cost_adjusted_metrics(max_sharpe_bt, capital, risk_free_rate)
    )

    ff3_bt_metrics = _cost_adjusted_metrics(ff3_bt, capital, risk_free_rate)
    ff5_bt_metrics = _cost_adjusted_metrics(ff5_bt, capital, risk_free_rate)

    capm_bt_metrics = (
        _cost_adjusted_metrics(capm_bt, capital, risk_free_rate)
    )

    min_bt_metrics = (
        _cost_adjusted_metrics(min_vol_bt, capital, risk_free_rate)
    )

    risk_parity_bt_metrics = (
        _cost_adjusted_metrics(risk_parity_bt, capital, risk_free_rate)
    )

    hrp_bt_metrics = (
        _cost_adjusted_metrics(hrp_bt, capital, risk_free_rate)
    )

    equal_bt_metrics = (
        _cost_adjusted_metrics(equal_weight_bt, capital, risk_free_rate)
    )

    ftse_bt_metrics = (
        calculate_backtest_metrics(
            ftse_mib_bt,
            risk_free_rate,
        )
    )

    transaction_cost_summary = {}
    if transaction_costs is not None:
        for name, curve in {
            "Max Sharpe": max_sharpe_bt, "CAPM": capm_bt,
            "FF3": ff3_bt, "FF5": ff5_bt,
            "Min Volatility": min_vol_bt, "Risk Parity": risk_parity_bt,
            "HRP": hrp_bt, "Equal Weight": equal_weight_bt,
        }.items():
            log = curve.attrs.get("transaction_log", pd.DataFrame())
            gross = curve.get("Gross Portfolio Value")
            transaction_cost_summary[name] = {
                "Total Costs (€)": float(log["Total Cost"].sum()) if not log.empty else 0.0,
                "Orders": int(log["Orders"].sum()) if not log.empty else 0,
                "Gross Return (%)": (float(gross.iloc[-1]) / capital - 1) * 100 if gross is not None and len(gross) else float("nan"),
                "Net Return (%)": (float(curve["Portfolio Value"].iloc[-1]) / capital - 1) * 100,
            }

    return {
        "transaction_cost_summary": transaction_cost_summary,
        "transaction_cost_preset": transaction_costs,
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
        "ff5_bt": ff5_bt,
        "ff5_weights_history": ff5_weights_history,
        "ff5_bt_metrics": ff5_bt_metrics,
        "ff5_factor_exposure_history": ff5_factor_exposure_history,

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

    cost_summary = result.get("transaction_cost_summary", {})
    if cost_summary:
        st.subheader("Transaction Costs — Gross vs Net")
        st.caption("Costs are deducted from the portfolio cash at each actual rebalance; "
                   "gross is a cost-free counterfactual using the same target weights. "
                   "The index benchmark has no simulated trading costs.")
        st.dataframe(pd.DataFrame.from_dict(cost_summary, orient="index").round(2),
                     use_container_width=True)

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

            **({"FF5 Maximum Sharpe": result["ff5_bt_metrics"]}
               if "ff5_bt_metrics" in result else {}),

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
    if "ff5_bt" in result:
        portfolio_fig.add_trace(go.Scatter(
            x=result["ff5_bt"].index, y=result["ff5_bt"]["Portfolio Value"],
            mode="lines", name="FF5 Maximum Sharpe",
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
    ff5_drawdown = calculate_drawdown(result["ff5_bt"]) if "ff5_bt" in result else None

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
    if ff5_drawdown is not None:
        drawdown_fig.add_trace(go.Scatter(
            x=ff5_drawdown.index, y=ff5_drawdown, mode="lines",
            name="FF5 Maximum Sharpe",
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
            "FF5 Maximum Sharpe",
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

    elif rebalance_strategy == "FF5 Maximum Sharpe":
        selected_weights_history = result.get("ff5_weights_history", pd.DataFrame())

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

    ff5_exposures = result.get("ff5_factor_exposure_history")
    if isinstance(ff5_exposures, pd.DataFrame) and not ff5_exposures.empty:
        st.subheader("FF5 Factor Exposure History")
        st.dataframe(ff5_exposures.style.format(precision=3), use_container_width=True)

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



    if "ff5_bt" in result:
        final_ff5 = result["ff5_bt"]["Portfolio Value"].iloc[-1]
        st.metric("FF5 Maximum Sharpe", f"€{final_ff5:,.2f}",
                  delta=f"{final_ff5 / capital - 1:.2%}")

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

asset_selection_tab, optimization_tab, factor_models_tab, backtest_tab, monte_carlo_tab, robustness_tab, volatility_tab, saved_experiments_tab = st.tabs(
    [
        "Asset Selection",
        "Portfolio Optimization",
        "Factor Models",
        "Backtesting",
        "Monte Carlo Simulation",
        "Robustness Analysis",
        "Volatility Analytics",
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
# ROLLING ROBUSTNESS ANALYSIS (INDEPENDENT TAB)
# =========================================================

with robustness_tab:
    st.header("Robustness Analysis")
    st.caption(
        "Historical rolling out-of-sample validation. Each window independently "
        "re-estimates CAPM, FF3 and FF5 using the existing backtesting engine."
    )
    st.info(
        "Historical Quant Selection uses point-in-time Momentum + Risk only. "
        "Today's instrument list may still introduce survivorship bias. "
        "Overlapping windows are not independent observations."
    )
    rb1, rb2, rb3 = st.columns(3)
    with rb1:
        rb_start = st.number_input("First OOS year", min_value=2010, max_value=2026, value=2010, step=1, key="rb_start")
        rb_end = st.number_input("Last complete OOS year", min_value=2010, max_value=2025, value=2025, step=1, key="rb_end")
    with rb2:
        rb_est = st.selectbox("Estimation Window", [1, 2, 3, 5, 7, 10], index=3, format_func=lambda x: f"{x} years", key="rb_est")
        rb_horizon = st.selectbox("Investment Horizon", [3, 6, 12, 24, 36, 60, 120], index=2, format_func=lambda x: f"{x} months", key="rb_horizon")
    with rb3:
        rb_step = st.selectbox("Rolling Step", [1, 3, 6, 12], index=3, format_func=lambda x: f"{x} months", key="rb_step")
        rb_policy = st.selectbox("Selection Policy", ["Fixed", "Dynamic"], key="rb_policy")

    rb_top = st.number_input("Historical Top N", min_value=2, max_value=100, value=20, step=1, key="rb_top")
    rb_rebalance = st.selectbox("Rebalance Frequency", [21, 63, 126, 252], index=1,
                                format_func=lambda x: {21: "Monthly", 63: "Quarterly", 126: "Semiannual", 252: "Annual"}[x], key="rb_rebalance")
    st.caption("The historical universe comes from the selected market, not the current Top N selection.")
    with st.expander("Transaction costs — empirical-informed scenarios", expanded=True):
        rb_cost_preset = st.selectbox("Cost scenario", list(PRESETS), index=1, key="rb_cost_preset")
        rb_assumption = PRESETS[rb_cost_preset]
        st.caption(
            f"Per order: max(€{rb_assumption.min_commission:.2f}, "
            f"{rb_assumption.commission_rate:.2%} × notional); "
            f"half-spread {rb_assumption.half_spread_bps:.0f} bps + "
            f"slippage {rb_assumption.slippage_bps:.0f} bps on traded notional. "
            "Costs include initial purchase and actual rebalance trades. "
            "No blanket transaction tax: issuer/date-specific tax treatment requires separate data."
        )
        st.caption(
            "Broker commission reference: Interactive Brokers Europe fixed/tiered schedules. "
            "Spread/slippage values are sensitivity assumptions informed by market microstructure "
            "research (Frazzini, Israel & Moskowitz, Trading Costs, 2018), "
            "NOT instrument-specific measured quotes. "
            "Same-close execution, fractional shares and no financing/FX costs remain limitations."
        )
        st.markdown(
            "Sources: [IBKR EU commissions](https://www.interactivebrokers.ie/en/pricing/commissions-stocks.php) · "
            "[Frazzini et al. (2018)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3229719)"
        )


    if st.button("Run Rolling Robustness", type="primary", key="rb_run"):
        if rb_start > rb_end:
            st.error("First OOS year must not exceed the last OOS year.")
        else:
            try:
                rb_tickers = market_instruments["ticker"].dropna().astype(str).tolist()
                rb_progress = st.progress(0.0, text="Running historical windows...")
                with st.spinner("Downloading historical prices and evaluating rolling windows..."):
                    rb_results, rb_skipped, rb_info = run_rolling_robustness(
                        tickers=rb_tickers,
                        benchmark_ticker=FTSE_MIB_TICKER,
                        start_year=int(rb_start), end_year=int(rb_end),
                        estimation_years=int(rb_est), horizon_months=int(rb_horizon),
                        step_months=int(rb_step), top_n=int(rb_top),
                        selection_policy=rb_policy.lower(),
                        rebalance_days=int(rb_rebalance),
                        capital=10000.0, risk_free_rate=0.02,
                        max_weight=max(0.20, 1.0 / int(rb_top)),
                        progress=lambda value: rb_progress.progress(value),
                        transaction_costs=rb_assumption,
                    )
                rb_progress.empty()
                st.session_state["robustness_run"] = (rb_results, rb_skipped, rb_info)
            except Exception as exc:
                st.error(f"Robustness run failed: {exc}")

    if "robustness_run" in st.session_state:
        rb_results, rb_skipped, rb_info = st.session_state["robustness_run"]
        st.caption(
            f"Downloaded history: {rb_info['first_price']} to {rb_info['last_price']} · "
            f"Universe: {rb_info['asset_columns']} tickers with some data · "
            f"Candidate windows: {rb_info['candidate_windows']}"
        )
        if not rb_results.empty and "Net Total Return" in rb_results.columns:
            st.subheader("Transaction costs — gross vs net")
            st.caption(
                "Cost-adjusted results are computed from actual changes in target holdings at each "
                "rebalance. FTSE MIB remains an index reference without simulated ETF fees or tracking error. "
                "All figures are historical scenario estimates, not executable trade confirmations."
            )
            rb_cost_rows = rb_results[rb_results["Model"] != "FTSE MIB"].copy()
            rb_cost_rows["Return drag (pp)"] = (
                rb_cost_rows["Total Return"] - rb_cost_rows["Net Total Return"]
            ) * 100
            rb_cost_summary = rb_cost_rows.groupby("Model").agg(
                Windows=("Total Return", "count"),
                Gross_Return=("Total Return", "mean"),
                Net_Return=("Net Total Return", "mean"),
                Mean_Cost_EUR=("Transaction Costs EUR", "mean"),
                Mean_Orders=("Orders", "mean"),
                Return_Drag_pp=("Return drag (pp)", "mean"),
                Net_Sharpe=("Net Sharpe Ratio", "mean"),
                Net_Drawdown=("Net Max Drawdown", "mean"),
            ).reindex(["CAPM", "FF3", "FF5", "Equal Weight"])
            st.dataframe(rb_cost_summary.style.format({
                "Gross_Return": "{:+.2%}", "Net_Return": "{:+.2%}",
                "Mean_Cost_EUR": "€{:,.2f}", "Mean_Orders": "{:.1f}",
                "Return_Drag_pp": "{:.2f}", "Net_Sharpe": "{:.2f}",
                "Net_Drawdown": "{:.2%}",
            }), use_container_width=True)
            rb_cost_chart = rb_cost_summary[["Gross_Return", "Net_Return"]] * 100
            st.bar_chart(rb_cost_chart, y_label="Average annual OOS return (%)")
            st.download_button(
                "Download transaction cost comparison (CSV)",
                rb_cost_rows.to_csv(index=False).encode("utf-8"),
                file_name="rolling_robustness_transaction_costs.csv", mime="text/csv",
                key="rb_cost_download",
            )
        if not rb_results.empty:
            # The benchmark is an explicit fifth strategy, not a portfolio model.
            # Keep only OOS dates shared by all five strategies for the primary comparison.
            rb_models = ["CAPM", "FF3", "FF5", "Equal Weight", "FTSE MIB"]
            rb_available = rb_results[rb_results["Model"].isin(rb_models)].copy()
            rb_complete_dates = (
                rb_available.groupby("OOS Start")["Model"]
                .agg(lambda x: set(rb_models).issubset(set(x)))
            )
            rb_complete_dates = rb_complete_dates[rb_complete_dates].index
            rb_common = rb_available[rb_available["OOS Start"].isin(rb_complete_dates)].copy()
            st.subheader("Like-for-like comparison (same OOS windows)")
            st.caption(
                f"{len(rb_complete_dates)} common windows with all five strategies; "
                "missing-model windows are excluded only from this comparison, not from the raw results."
            )
            if not rb_common.empty:
                rb_common["Positive Window"] = rb_common["Total Return"] > 0
                summary = rb_common.groupby("Model").agg(
                    Windows=("Total Return", "count"),
                    Median_Return=("Total Return", "median"),
                    Mean_Return=("Total Return", "mean"),
                    Positive_Rate=("Positive Window", "mean"),
                    Median_Sharpe=("Sharpe Ratio", "median"),
                    Worst_Drawdown=("Max Drawdown", "min"),
                )
                benchmark_returns = rb_common.loc[
                    rb_common["Model"] == "FTSE MIB", ["OOS Start", "Total Return"]
                ].set_index("OOS Start")["Total Return"]
                rb_common["Benchmark OOS Return"] = rb_common["OOS Start"].map(benchmark_returns)
                rb_common["Outperformance"] = rb_common["Total Return"] > rb_common["Benchmark OOS Return"]
                rb_common["Excess OOS Return"] = rb_common["Total Return"] - rb_common["Benchmark OOS Return"]
                extra = rb_common.groupby("Model").agg(
                    Benchmark_Win_Rate=("Outperformance", "mean"),
                    Mean_Excess_Return=("Excess OOS Return", "mean"),
                )
                summary = summary.join(extra).reindex(rb_models)
                st.dataframe(summary.style.format({
                    "Median_Return": "{:.2%}", "Mean_Return": "{:.2%}",
                    "Positive_Rate": "{:.1%}", "Benchmark_Win_Rate": "{:.1%}",
                    "Mean_Excess_Return": "{:+.2%}",
                    "Median_Sharpe": "{:.2f}", "Worst_Drawdown": "{:.2%}",
                }), use_container_width=True)
                # Risk diagnostics are descriptive and use the same completed OOS
                # windows for all five strategies. No financial engine changes.
                st.subheader("Risk-adjusted performance — common windows")
                st.caption(
                    "Sharpe and drawdown statistics summarize individual OOS windows, "
                    "not one continuously invested multi-year portfolio. "
                    "Higher Sharpe and less negative drawdown are preferable."
                )
                rb_risk_summary = rb_common.groupby("Model").agg(
                    Windows=("Sharpe Ratio", "count"),
                    Mean_Return=("Total Return", "mean"),
                    Mean_Sharpe=("Sharpe Ratio", "mean"),
                    Median_Sharpe=("Sharpe Ratio", "median"),
                    Mean_Drawdown=("Max Drawdown", "mean"),
                    Median_Drawdown=("Max Drawdown", "median"),
                    Worst_Drawdown=("Max Drawdown", "min"),
                ).reindex(rb_models)
                st.dataframe(
                    rb_risk_summary.style.format({
                        "Mean_Return": "{:+.2%}",
                        "Mean_Sharpe": "{:.2f}",
                        "Median_Sharpe": "{:.2f}",
                        "Mean_Drawdown": "{:.2%}",
                        "Median_Drawdown": "{:.2%}",
                        "Worst_Drawdown": "{:.2%}",
                    }),
                    use_container_width=True,
                )
                rb_risk_col1, rb_risk_col2 = st.columns(2)
                with rb_risk_col1:
                    st.markdown("**Annual OOS Sharpe Ratio**")
                    st.line_chart(
                        rb_common.pivot_table(
                            index="OOS Start", columns="Model", values="Sharpe Ratio"
                        ).reindex(columns=rb_models),
                        y_label="Sharpe Ratio",
                    )
                with rb_risk_col2:
                    st.markdown("**Annual OOS Maximum Drawdown**")
                    st.line_chart(
                        rb_common.pivot_table(
                            index="OOS Start", columns="Model", values="Max Drawdown"
                        ).reindex(columns=rb_models) * 100,
                        y_label="Maximum Drawdown (%)",
                    )

                st.subheader("Paired comparison vs Equal Weight")
                st.caption(
                    "Each cell counts windows in which the model beats Equal Weight "
                    "on the SAME OOS dates. Ties do not count as wins."
                )
                rb_return_wide = rb_common.pivot(index="OOS Start", columns="Model", values="Total Return")
                rb_sharpe_wide = rb_common.pivot(index="OOS Start", columns="Model", values="Sharpe Ratio")
                rb_drawdown_wide = rb_common.pivot(index="OOS Start", columns="Model", values="Max Drawdown")
                rb_pairs = []
                for rb_model in ["CAPM", "FF3", "FF5", "FTSE MIB"]:
                    rb_valid = (
                        rb_return_wide[[rb_model, "Equal Weight"]].notna().all(axis=1)
                        & rb_sharpe_wide[[rb_model, "Equal Weight"]].notna().all(axis=1)
                        & rb_drawdown_wide[[rb_model, "Equal Weight"]].notna().all(axis=1)
                    )
                    rb_n = int(rb_valid.sum())
                    if not rb_n:
                        continue
                    rb_pairs.append({
                        "Model": rb_model,
                        "Compared windows": rb_n,
                        "Higher return": int((rb_return_wide.loc[rb_valid, rb_model] > rb_return_wide.loc[rb_valid, "Equal Weight"]).sum()),
                        "Higher Sharpe": int((rb_sharpe_wide.loc[rb_valid, rb_model] > rb_sharpe_wide.loc[rb_valid, "Equal Weight"]).sum()),
                        "Smaller drawdown": int((rb_drawdown_wide.loc[rb_valid, rb_model] > rb_drawdown_wide.loc[rb_valid, "Equal Weight"]).sum()),
                        "Mean return difference (pp)": float((rb_return_wide.loc[rb_valid, rb_model] - rb_return_wide.loc[rb_valid, "Equal Weight"]).mean() * 100),
                        "Mean Sharpe difference": float((rb_sharpe_wide.loc[rb_valid, rb_model] - rb_sharpe_wide.loc[rb_valid, "Equal Weight"]).mean()),
                        "Mean drawdown difference (pp)": float((rb_drawdown_wide.loc[rb_valid, rb_model] - rb_drawdown_wide.loc[rb_valid, "Equal Weight"]).mean() * 100),
                    })
                if rb_pairs:
                    rb_pair_df = pd.DataFrame(rb_pairs).set_index("Model")
                    for rb_col in ["Higher return", "Higher Sharpe", "Smaller drawdown"]:
                        rb_pair_df[rb_col] = rb_pair_df.apply(
                            lambda r: f"{int(r[rb_col])}/{int(r['Compared windows'])}", axis=1
                        )
                    st.dataframe(rb_pair_df.style.format({
                        "Mean return difference (pp)": "{:+.2f}",
                        "Mean Sharpe difference": "{:+.2f}",
                        "Mean drawdown difference (pp)": "{:+.2f}",
                    }), use_container_width=True)
                    st.caption(
                        "A positive drawdown difference means a shallower loss than Equal Weight; "
                        "a negative difference means a deeper loss."
                    )

                st.subheader("Stress windows — weakest FTSE MIB returns")
                st.caption(
                    "The three lowest-return benchmark windows among the common dates. "
                    "These are selected mechanically, not predefined crisis labels. "
                    "With overlapping windows, observations are not independent."
                )
                rb_stress_dates = rb_return_wide["FTSE MIB"].nsmallest(3).index
                rb_stress = rb_common[rb_common["OOS Start"].isin(rb_stress_dates)].copy()
                rb_stress["OOS Start"] = rb_stress["OOS Start"].astype(str)
                rb_stress = rb_stress[["OOS Start", "Model", "Total Return", "Sharpe Ratio", "Max Drawdown"]]
                rb_stress["Model"] = pd.Categorical(rb_stress["Model"], categories=rb_models, ordered=True)
                rb_stress = rb_stress.sort_values(["OOS Start", "Model"])
                st.dataframe(rb_stress.style.format({
                    "Total Return": "{:+.2%}",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                }), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download risk comparison (CSV)",
                    rb_risk_summary.reset_index().to_csv(index=False).encode("utf-8"),
                    file_name="rolling_robustness_risk_comparison.csv", mime="text/csv",
                    key="rb_risk_export",
                )

                st.subheader("OOS return by start date — common windows")
                chart_data = rb_common.pivot_table(
                    index="OOS Start", columns="Model", values="Total Return"
                ).reindex(columns=rb_models)
                st.line_chart(chart_data * 100, y_label="Total Return (%)")
                st.subheader("Excess return vs FTSE MIB — common windows")
                excess_chart = rb_common[rb_common["Model"] != "FTSE MIB"].pivot_table(
                    index="OOS Start", columns="Model", values="Excess OOS Return"
                ).reindex(columns=rb_models[:-1])
                st.bar_chart(excess_chart * 100, y_label="Excess Return (percentage points)")
            else:
                st.warning("No windows contain all five strategies. Review skipped-window diagnostics.")
            st.subheader("All completed windows (unfiltered)")
            st.dataframe(rb_results, use_container_width=True, hide_index=True)
            st.download_button("Download robustness results (CSV)", rb_results.to_csv(index=False).encode("utf-8"),
                               file_name="rolling_robustness_results.csv", mime="text/csv", key="rb_export")
        else:
            st.warning("No valid OOS windows were completed. Inspect the data coverage and skipped-window diagnostics.")
        if not rb_skipped.empty:
            with st.expander(f"Skipped windows / model failures ({len(rb_skipped)})"):
                st.dataframe(rb_skipped, use_container_width=True, hide_index=True)
                st.download_button("Download diagnostics (CSV)", rb_skipped.to_csv(index=False).encode("utf-8"),
                                   file_name="rolling_robustness_diagnostics.csv", mime="text/csv", key="rb_errors")


def _drbt_pretest_returns(prices, weights_history, test_start, required):
    """Construct pre-OOS constant-mix return proxy using first investable weights.

    This is NOT the historical optimized strategy: weights are those chosen at t0.
    Historical prices are only used for a conditional volatility estimate at t0.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Missing historical price matrix; rerun Standard Walk-Forward.")
    if not isinstance(weights_history, pd.DataFrame) or weights_history.empty:
        raise ValueError("Missing initial strategy weights; rerun Standard Walk-Forward.")
    w = pd.to_numeric(weights_history.iloc[0], errors="coerce").fillna(0.0)
    w = w[w > 1e-10]
    w = w[w.index.isin(prices.columns)]
    if w.empty:
        raise ValueError("Initial strategy holdings have no matching historical price columns.")
    history = prices.loc[prices.index < test_start, w.index].apply(pd.to_numeric, errors="coerce")
    # Do not forward-fill stale data across missing sessions; use complete dates.
    history = history.dropna(how="any")
    history = history.loc[(history > 0).all(axis=1)]
    if len(history) < required + 1:
        raise ValueError(f"Only {max(0, len(history)-1)} complete pre-test price returns for initial holdings; "
                         f"need {required}. Check instrument listing dates or extend history.")
    rets = history.pct_change().dropna()
    w = w / w.sum()
    return rets.mul(w, axis=1).sum(axis=1).tail(required)


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

        # Optional short-horizon evaluation: keep the rolling estimation
        # window unchanged, but test only the most recent N trading days.
        standard_oos_options = {
            "Remaining historical period": None,
            "1 month": 21,
            "2 months": 42,
            "3 months": 63,
            "6 months": 126,
        }
        standard_oos_label = st.selectbox(
            "Out-of-sample test period",
            list(standard_oos_options),
            index=0,
            key="standard_oos_period",
        )
        standard_oos_days = standard_oos_options[standard_oos_label]
        available_oos_days = standard_dataset_years * 252 - standard_lookback_days
        if standard_oos_days is not None and standard_oos_days > available_oos_days:
            st.warning("The selected dataset and estimation window do not "
                       "leave enough history for this test period.")
        st.info(
            f"Historical dataset: {standard_dataset_years} years\n\n"
            f"Estimation window: {standard_lookback_label}\n\n"
            f"Out-of-sample: {standard_oos_label if standard_oos_days is not None else f'approximately {available_oos_days / 252:g} years'} "
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
            "1 month": 21,
            "2 months": 42,
            "3 months": 63,
            "6 months": 126,
            **{
                f"{year} year" + ("s" if year != 1 else ""): 252 * year
                for year in range(1, 11)
            },
        }

        fixed_backtest_label = (
            st.selectbox(
                "Backtest period",
                list(
                    fixed_backtest_options.keys()
                ),
                index=4,  # Keep the previous default: 1 year.
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
            f"{required_years:.2f} years "
            f"of historical data required."
        )


    st.subheader("Transaction Costs")
    bt_cost_choice = st.selectbox(
        "Execution cost scenario",
        ["None (legacy / gross)", "Optimistic", "Base", "Conservative"],
        index=2, key="bt_transaction_cost_choice",
        help="Costs charged on every actual purchase, sale and rebalance. "
             "Commission per order plus half-spread and slippage on traded notional.",
    )
    bt_cost_assumptions = TRANSACTION_COST_PRESETS.get(bt_cost_choice)
    if bt_cost_assumptions is not None:
        st.caption(
            f"Commission: {bt_cost_assumptions.commission_rate*100:.2f}% "
            f"(minimum €{bt_cost_assumptions.min_commission:.2f}/order); "
            f"half-spread: {bt_cost_assumptions.half_spread_bps:g} bps; "
            f"slippage: {bt_cost_assumptions.slippage_bps:g} bps. "
            "Italian transaction tax and capital gains taxes excluded."
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
            shared_ff5_factors = None
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

                        if standard_oos_days is not None:
                            required_rows = standard_lookback_days + standard_oos_days + 1
                            if len(standard_prices) < required_rows:
                                raise ValueError(
                                    "Not enough historical observations for the "
                                    "selected short out-of-sample period."
                                )
                            # Use the MOST RECENT test window, not the first
                            # months after the beginning of the historical dataset.
                            standard_prices = standard_prices.tail(required_rows)

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
                        if shared_ff5_factors is None:
                            shared_ff5_factors = load_europe_ff5()
                        standard_ff5_factors = shared_ff5_factors
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
                                transaction_costs=bt_cost_assumptions,
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
                                ff5_factors=standard_ff5_factors,
                                eurusd_prices=standard_eurusd,
                            )

                            standard_result["drbt_price_history"] = standard_prices
                            try:
                                standard_result["ff3_performance_attribution"] = compute_ff3_attribution(
                                    standard_result["ff3_bt"],
                                    standard_result["ff3_factor_exposure_history"],
                                    standard_ff3_factors, standard_eurusd,
                                )
                            except (ValueError, KeyError, TypeError) as attribution_error:
                                standard_result["ff3_attribution_error"] = str(attribution_error)

                            if policy == "fixed":
                                for model in ("capm", "ff3", "ff5"):
                                    try:
                                        standard_result[f"{model}_same_initial_buy_hold"] = (
                                            _build_same_initial_model_buy_hold(
                                                standard_prices, standard_result, model
                                            )
                                        )
                                    except ValueError as comparison_error:
                                        standard_result[f"{model}_comparison_error"] = str(comparison_error)

                            standard_result["lookback_label"] = standard_lookback_label
                            standard_result["period_label"] = standard_oos_label
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
                        if shared_ff5_factors is None:
                            shared_ff5_factors = load_europe_ff5()
                        fixed_ff5_factors = shared_ff5_factors
                        fixed_ff3_factors = shared_ff3_factors
                        fixed_eurusd = shared_eurusd
                        fixed_result = (
                            run_fixed_horizon_engine(
                                tickers=tickers,
                                transaction_costs=bt_cost_assumptions,
                                prices=fixed_prices,
                                estimation_days=
                                    fixed_lookback_days,
                                backtest_days=
                                    fixed_backtest_days,
                                ff3_factors=fixed_ff3_factors,
                                ff5_factors=fixed_ff5_factors,
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



    # =====================================================
    # DYNAMIC RISK BACKTEST (ADDITIVE; FINANCIAL ENGINES UNCHANGED)
    # =====================================================
    st.divider()
    st.subheader("Dynamic Risk Backtesting — multi-strategy exposure overlay")
    st.caption(
        "Experimental risk overlay on existing net backtest equity curves. "
        "The original strategies and their backtesting engines are unchanged; "
        "additional exposure trades are estimated separately."
    )
    drbt_strategy_keys = {
        "Maximum Sharpe": "max_sharpe_bt",
        "CAPM Maximum Sharpe": "capm_bt",
        "FF3 Maximum Sharpe": "ff3_bt",
        "FF5 Maximum Sharpe": "ff5_bt",
        "Minimum Volatility": "min_vol_bt",
        "Risk Parity": "risk_parity_bt",
        "HRP": "hrp_bt",
        "Equal Weight": "equal_weight_bt",
    }
    drbt_sources = {}
    if isinstance(standard_result, dict):
        if any(k in standard_result for k in drbt_strategy_keys.values()):
            drbt_sources["Standard · Fixed"] = standard_result
        else:
            for policy, result in standard_result.items():
                if isinstance(result, dict) and any(k in result for k in drbt_strategy_keys.values()):
                    drbt_sources[f"Standard · {str(policy).title()}"] = result
    if isinstance(fixed_result, dict) and any(k in fixed_result for k in drbt_strategy_keys.values()):
        drbt_sources["Fixed Horizon"] = fixed_result

    if not drbt_sources:
        st.info("Run a standard backtest first to compare volatility overlays across strategies.")
    else:
        drbt_source = st.selectbox("Backtest source", list(drbt_sources), key="drbt_source", help='Seleziona il backtest già eseguito da usare come riferimento: Standard Fixed, Standard Dynamic oppure Fixed Horizon, se disponibili. Il motore di ottimizzazione originale non cambia.')
        drbt_base = drbt_sources[drbt_source]
        drbt_available = {name: key for name, key in drbt_strategy_keys.items()
                          if key in drbt_base and isinstance(drbt_base[key], (pd.DataFrame, pd.Series))
                          and len(drbt_base[key]) > 0}
        drbt_strategies = st.multiselect(
            "Portfolio strategies", list(drbt_available),
            default=list(drbt_available), key="drbt_strategies",
            help='Seleziona uno, più o tutti gli otto portafogli. Ogni modello di volatilità viene confrontato con lo Standard della stessa strategia.'
        )
        drbt_a, drbt_b, drbt_c, drbt_d = st.columns(4)
        with drbt_a:
            drbt_target = st.number_input("Target volatility (%)", 5., 50., 15., 1., key="drbt_target", help='Obiettivo di volatilità annualizzata del portafoglio. L’esposizione obiettivo è il minimo tra 100% e target diviso volatilità prevista. Non garantisce che la volatilità realizzata rispetti il target.')
        with drbt_b:
            drbt_horizon = st.selectbox("Forecast horizon (sessions)", [5, 21, 63], index=1, key="drbt_horizon", help='Numero di sedute a cui si riferisce la previsione di volatilità: 5 circa una settimana, 21 circa un mese, 63 circa un trimestre. Non è la frequenza dei ribilanciamenti.')
        with drbt_c:
            drbt_update = st.selectbox("Forecast refresh (sessions)", [1, 5, 21], index=1, key="drbt_update", help='Ogni quante sedute si aggiorna la previsione: 1 ogni seduta, 5 circa ogni settimana, 21 circa ogni mese. Le decisioni hanno effetto dalla seduta successiva.')
        with drbt_d:
            drbt_train = st.number_input("Pre-test volatility estimation (sessions)", 252, 1000, 252, 21, key="drbt_train", help='Sedute precedenti al test usate per inizializzare la volatilità: 252 sono circa un anno di Borsa. Nessuna seduta fuori campione viene esclusa. La storia è una proxy calcolata con i pesi iniziali.')
        drbt_e, drbt_f, drbt_g = st.columns(3)
        with drbt_e:
            drbt_band = st.number_input("Execute exposure change from (pp)", 5., 50., 10., 1., key="drbt_band", help='Soglia minima in punti percentuali per ridurre l’esposizione. Esempio: partendo dal 100%, con soglia 15 pp si interviene se l’esposizione obiettivo è al massimo 85%.')
        with drbt_f:
            drbt_review = st.number_input("Review from (pp)", 0., 20., 5., 1., key="drbt_review", help='Soglia di attenzione in punti percentuali. Se lo scostamento supera questa soglia il sistema segnala REVIEW, senza necessariamente eseguire un’operazione. Deve essere inferiore alla soglia di esecuzione.')
        with drbt_g:
            drbt_cash = st.number_input("Cash annual yield (%)", 0., 15., 0., 0.5, key="drbt_cash", help='Rendimento annualizzato attribuito alla quota in liquidità. Con 0% il capitale non investito non genera interessi nella simulazione.')
        drbt_reentry = st.number_input(
            "Restore exposure from (pp)", 1.0, float(drbt_band),
            min(5.0, float(drbt_band)), 1.0, key="drbt_reentry",
            help='Soglia minima in punti percentuali per aumentare l’esposizione quando la volatilità prevista scende. Una soglia inferiore a quella di vendita permette un rientro più rapido.'
        )
        drbt_models = st.multiselect(
            "Forecast models", ["EWMA", "GARCH(1,1)", "GJR-GARCH", "EGARCH"],
            default=["EWMA", "GARCH(1,1)"], key="drbt_models"
        , help='Modelli di previsione: EWMA pesa maggiormente i rendimenti recenti; GARCH(1,1) stima volatilità persistente; GJR-GARCH ed EGARCH considerano anche reazioni asimmetriche agli shock.')
        drbt_cost_choice = st.selectbox(
            "Incremental exposure trading costs", ["Optimistic", "Base", "Conservative"],
            index=1, key="drbt_cost_choice"
        , help='Scenario di commissioni, spread e slippage per le operazioni aggiuntive di gestione del rischio. I costi già inclusi nello Standard non vengono duplicati.')
        st.caption("Volatility is initialized from pre-test asset returns with each strategy's first weights. "
                   "All OOS dates are retained. This is a fixed-weight pre-test volatility proxy, not a reconstructed "
                   "historical strategy. Forecasts use lagged portfolio returns; exposure decisions take effect next session. "
                   "Only incremental overlay costs are charged. No taxes or full constituent-level execution.")
        if st.button("Run Dynamic Risk Backtest", type="primary", key="drbt_run"):
            if not drbt_strategies or not drbt_models:
                st.warning("Select at least one portfolio strategy and one forecast model.")
            elif drbt_review >= drbt_band:
                st.warning("The review band must be smaller than the execution band.")
            else:
                try:
                    from src.dynamic_risk_backtest import run_exposure_overlay
                    drbt_cost = TRANSACTION_COST_PRESETS[drbt_cost_choice]
                    metrics_parts, curve_parts, log_parts, errors = [], {}, [], []
                    for strategy_name in drbt_strategies:
                        strategy_key = drbt_available[strategy_name]
                        weight_key = strategy_key.replace("_bt", "_weights_history")
                        history = drbt_base.get(weight_key)
                        holdings = (int((history.iloc[-1] > 1e-10).sum())
                                    if isinstance(history, pd.DataFrame) and not history.empty else 20)
                        try:
                            with st.spinner(f"Evaluating {strategy_name}..."):
                                pretest = _drbt_pretest_returns(
                                    drbt_base.get("drbt_price_history"), history,
                                    drbt_base[strategy_key].index[0], int(drbt_train))
                                met, curves, log = run_exposure_overlay(
                                    drbt_base[strategy_key], models=drbt_models, train=int(drbt_train),
                                    pretest_returns=pretest,
                                    horizon=int(drbt_horizon), target_vol=float(drbt_target),
                                    trade_band_pp=float(drbt_band), review_band_pp=float(drbt_review),
                                    update_every=int(drbt_update), capital=float(drbt_base["capital"]),
                                    commission_rate=drbt_cost.commission_rate,
                                    min_commission=drbt_cost.min_commission,
                                    half_spread_bps=drbt_cost.half_spread_bps,
                                    slippage_bps=drbt_cost.slippage_bps,
                                    estimated_holdings=holdings, cash_annual_yield=float(drbt_cash)/100,
                                    reentry_band_pp=float(drbt_reentry),
                                )
                            met = met.reset_index().rename(columns={"Strategy": "Risk Model"})
                            met.insert(0, "Portfolio Strategy", strategy_name)
                            standard = met.iloc[0]
                            met["Δ CAGR (pp)"] = (met["CAGR"] - standard["CAGR"]) * 100
                            met["Δ Sharpe"] = met["Sharpe (approx.)"] - standard["Sharpe (approx.)"]
                            met["Δ Max Drawdown (pp)"] = (met["Max Drawdown"] - standard["Max Drawdown"]) * 100
                            metrics_parts.append(met)
                            curve_parts[strategy_name] = curves
                            if not log.empty:
                                log.insert(0, "Portfolio Strategy", strategy_name)
                                log_parts.append(log)
                        except Exception as exc:
                            errors.append(f"{strategy_name}: {exc}")
                    if not metrics_parts:
                        raise ValueError("No strategy could be evaluated. " + "; ".join(errors))
                    st.session_state["drbt_result"] = {
                        "metrics": pd.concat(metrics_parts, ignore_index=True),
                        "curves": curve_parts,
                        "log": pd.concat(log_parts, ignore_index=True) if log_parts else pd.DataFrame(),
                        "source": drbt_source, "warmup": int(drbt_train),
                        "errors": errors,
                    }
                except Exception as exc:
                    st.error(f"Dynamic Risk Backtest failed: {exc}")
        saved = st.session_state.get("drbt_result")
        if saved:
            st.caption(f"Source: {saved['source']}. Pre-test estimation: {saved['warmup']} sessions "
                       "per strategy (no OOS sessions excluded).")
            for err in saved.get("errors", []):
                st.warning(f"Skipped {err}")
            table = saved["metrics"].copy()
            for col in ["CAGR", "Max Drawdown", "Annual volatility", "Total return"]:
                table[col] = (table[col] * 100).round(2)
            table = table.rename(columns={"CAGR": "CAGR (%)", "Max Drawdown": "Max Drawdown (%)",
                                          "Annual volatility": "Annual volatility (%)", "Total return": "Total return (%)"})
            # Color semantics: green = improvement versus this portfolio's Standard;
            # red = deterioration; neutral for turnover, cash and trading costs.
            def _risk_comparison_style(frame):
                from pandas.api.types import is_numeric_dtype
                styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
                positive_good = {"CAGR (%)", "Sharpe (approx.)", "Max Drawdown (%)",
                                 "Total return (%)", "Δ CAGR (pp)", "Δ Sharpe",
                                 "Δ Max Drawdown (pp)"}
                negative_good = {"Annual volatility (%)"}
                for _, grp in frame.groupby("Portfolio Strategy", sort=False):
                    base_rows = grp[grp["Risk Model"] == "Standard"]
                    if base_rows.empty:
                        continue
                    base = base_rows.iloc[0]
                    for idx, row in grp.iterrows():
                        if row["Risk Model"] == "Standard":
                            styles.loc[idx, "Risk Model"] = "font-weight: 700; background-color: rgba(125,125,125,0.13)"
                            continue
                        for col in positive_good | negative_good:
                            if col not in frame.columns or not is_numeric_dtype(frame[col]):
                                continue
                            baseline = 0.0 if col.startswith("Δ ") else base[col]
                            val = row[col]
                            if pd.isna(val) or pd.isna(baseline):
                                continue
                            diff = float(val) - float(baseline)
                            if col in negative_good:
                                diff = -diff
                            if abs(diff) < 1e-8:
                                continue
                            styles.loc[idx, col] = (
                                "background-color: rgba(34,197,94,0.23); color: #16a34a; font-weight: 650"
                                if diff > 0 else
                                "background-color: rgba(239,68,68,0.19); color: #dc2626; font-weight: 650"
                            )
                return styles

            st.caption("Color legend: 🟢 improvement vs the same strategy's Standard · "
                       "🔴 deterioration · gray = baseline. "
                       "Higher (less negative) drawdown is better; lower volatility is better. "
                       "Trading costs and number of trades remain neutral.")
            st.dataframe(table.style.apply(_risk_comparison_style, axis=None)
                         .format(precision=2, na_rep="—"),
                         use_container_width=True, hide_index=True)
            chart_strategy = st.selectbox("Equity curve comparison", list(saved["curves"]), key="drbt_chart_strategy")
            comparison_curves = saved["curves"][chart_strategy].copy()
            fig = go.Figure()
            line_colors = {"Standard": "#38bdf8", "EWMA": "#2563eb", "GARCH(1,1)": "#fb7185",
                           "GJR-GARCH": "#a78bfa", "EGARCH": "#f59e0b"}
            for model_name in comparison_curves.columns:
                is_standard = model_name == "Standard"
                fig.add_trace(go.Scatter(
                    x=comparison_curves.index,
                    y=comparison_curves[model_name],
                    mode="lines", name=model_name,
                    line=dict(color=line_colors.get(model_name),
                              width=3.2 if is_standard else 1.9,
                              dash="solid" if is_standard else "dash"),
                ))
            fig.update_layout(title=f"{chart_strategy}: standard vs risk-managed — aligned dates",
                              xaxis_title="Date", yaxis_title="Portfolio value (€)",
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Daily euro difference against the same strategy's Standard curve.
            # No interpolation or forward filling: only matched observations are compared.
            st.markdown("##### Difference vs Standard (€)")
            st.caption("Above zero = risk-managed portfolio ahead of Standard; "
                       "below zero = behind Standard. Daily differences, not returns.")
            if "Standard" in comparison_curves.columns:
                baseline = pd.to_numeric(comparison_curves["Standard"], errors="coerce")
                delta_fig = go.Figure()
                plotted = False
                for model_name in comparison_curves.columns:
                    if model_name == "Standard":
                        continue
                    candidate = pd.to_numeric(comparison_curves[model_name], errors="coerce")
                    aligned = pd.concat([baseline.rename("standard"),
                                         candidate.rename("candidate")], axis=1).dropna()
                    if aligned.empty:
                        continue
                    difference = aligned["candidate"] - aligned["standard"]
                    delta_fig.add_trace(go.Scatter(
                        x=difference.index, y=difference,
                        mode="lines", name=model_name,
                        line=dict(color=line_colors.get(model_name), width=2),
                        hovertemplate="%{x|%d %b %Y}<br>Difference: €%{y:+,.2f}<extra>" + model_name + "</extra>",
                    ))
                    plotted = True
                delta_fig.add_hline(y=0, line_color="#cbd5e1", line_width=1.5,
                                    line_dash="dash", annotation_text="Standard = €0",
                                    annotation_position="top left")
                delta_fig.update_layout(
                    title=f"{chart_strategy}: incremental value vs Standard",
                    xaxis_title="Date", yaxis_title="Difference (€)",
                    hovermode="x unified", height=360,
                )
                if plotted:
                    st.plotly_chart(delta_fig, use_container_width=True)
                else:
                    st.info("No aligned risk-managed curves available for this strategy.")
            else:
                st.info("Standard equity curve is unavailable for this comparison.")
            with st.expander("Exposure decisions and incremental costs"):
                st.dataframe(saved["log"], use_container_width=True)
            st.download_button("Download dynamic risk metrics CSV",
                               saved["metrics"].to_csv(index=False).encode(),
                               "dynamic_risk_metrics_all_strategies.csv", "text/csv", key="drbt_dl_metrics")
            st.download_button("Download dynamic risk decisions CSV",
                               saved["log"].to_csv(index=False).encode(),
                               "dynamic_risk_decisions_all_strategies.csv", "text/csv", key="drbt_dl_log")
            st.warning("Experimental exposure overlay: uses each strategy's existing net equity curve as "
                       "a risky sleeve. It does not replay constituent orders or jointly simulate cash "
                       "and portfolio optimization. Compare only aligned samples; validate before live use.")

# =========================================================
# SAVED EXPERIMENTS TAB
# =========================================================

# =========================================================
# MONTE CARLO SIMULATION (INDEPENDENT TAB)
# =========================================================
with monte_carlo_tab:
    st.header("Monte Carlo Simulation")
    st.caption(
        "Simulazioni statistiche condizionate ai rendimenti fuori campione del backtest. "
        "Non sono previsioni certe, né simulazioni degli ordini sui singoli titoli."
    )
    mc_keys = {
        "Maximum Sharpe": "max_sharpe_bt",
        "CAPM Maximum Sharpe": "capm_bt",
        "FF3 Maximum Sharpe": "ff3_bt",
        "FF5 Maximum Sharpe": "ff5_bt",
        "Minimum Volatility": "min_vol_bt",
        "Risk Parity": "risk_parity_bt",
        "HRP": "hrp_bt",
        "Equal Weight": "equal_weight_bt",
    }
    mc_sources = {}
    mc_standard = st.session_state.get("standard_backtest_results")
    mc_fixed = st.session_state.get("fixed_backtest_results")
    if isinstance(mc_standard, dict):
        if "max_sharpe_bt" in mc_standard:
            mc_sources["Standard · Fixed"] = mc_standard
        else:
            for mc_policy, mc_result in mc_standard.items():
                if isinstance(mc_result, dict) and "max_sharpe_bt" in mc_result:
                    mc_sources[f"Standard · {str(mc_policy).title()}"] = mc_result
    if isinstance(mc_fixed, dict) and "max_sharpe_bt" in mc_fixed:
        mc_sources["Fixed Horizon"] = mc_fixed

    if not mc_sources:
        st.info("Esegui prima un backtest nella scheda Backtesting. Monte Carlo utilizza le sue curve storiche nette.")
    else:
        mc_source_name = st.selectbox(
            "Fonte dei rendimenti", list(mc_sources), key="mc_source",
            help="Seleziona il backtest da cui estrarre i rendimenti giornalieri storici. "
                 "Le simulazioni partono da questi rendimenti, non dai rendimenti attesi dei modelli fattoriali."
        )
        mc_source = mc_sources[mc_source_name]
        mc_available = {
            name: eq for name, key in mc_keys.items()
            if (eq := mc_extract_equity(mc_source, key)) is not None
        }
        if not mc_available:
            st.warning("Nessuna curva valida trovata nel backtest selezionato.")
        else:
            mc_selected = st.multiselect(
                "Strategie di portafoglio", list(mc_available),
                default=list(mc_available), key="mc_strategies",
                help="Seleziona uno o più portafogli da simulare. Le strategie vengono confrontate "
                     "usando gli stessi shock casuali, preservando la dipendenza storica tra di esse."
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                mc_years = st.selectbox("Orizzonte (anni)", [1, 3, 5], key="mc_years",
                    help="Numero di anni futuri simulati: 252 sedute per anno.")
            with c2:
                mc_paths = st.selectbox("Numero simulazioni", [500, 1000, 2000, 3000], index=1, key="mc_paths",
                    help="Numero di traiettorie casuali. Un numero maggiore rende più stabili le stime, "
                         "ma aumenta i tempi di calcolo.")
            with c3:
                mc_capital = st.number_input("Capitale iniziale (€)", min_value=100.0,
                    value=10000.0, step=1000.0, key="mc_capital",
                    help="Capitale ipotetico da cui partono tutte le simulazioni. "
                         "Le curve storiche vengono convertite in rendimenti prima della simulazione.")
            with c4:
                mc_method = st.selectbox("Metodo di simulazione",
                    ["Block bootstrap", "IID bootstrap", "Multivariate Gaussian"], key="mc_method",
                    help="Block bootstrap: estrae blocchi consecutivi di sedute storiche e conserva parte "
                         "della dipendenza temporale. IID: estrae singole sedute indipendenti. "
                         "Gaussian: simula rendimenti normali multivariati con media e covarianza storiche.")
            c5, c6, c7 = st.columns(3)
            with c5:
                mc_block = st.selectbox("Lunghezza blocco (sedute)", [3, 5, 10, 21], index=1,
                    disabled=mc_method != "Block bootstrap", key="mc_block",
                    help="Nel block bootstrap, numero di sedute consecutive estratte insieme. "
                         "Blocchi più lunghi preservano più struttura temporale, ma riducono la varietà dei campioni.")
            with c6:
                mc_drift = st.number_input("Aggiustamento rendimento annuo (pp)",
                    min_value=-30.0, max_value=30.0, value=0.0, step=1.0, key="mc_drift",
                    help="Stress test sulla media dei rendimenti: -5 sottrae circa 5 punti percentuali "
                         "all'anno al rendimento medio giornaliero utilizzato nelle simulazioni. "
                         "Non modifica la volatilità o le correlazioni stimate.")
            with c7:
                mc_seed = st.number_input("Seed casuale", min_value=0, max_value=1000000,
                    value=42, step=1, key="mc_seed",
                    help="Permette di riprodurre gli stessi scenari casuali con gli stessi parametri.")

            if mc_selected:
                try:
                    mc_returns_preview = mc_aligned_returns({name: mc_available[name] for name in mc_selected})
                    st.caption(
                        f"Campione storico comune: {mc_returns_preview.index.min():%d/%m/%Y} – "
                        f"{mc_returns_preview.index.max():%d/%m/%Y} · "
                        f"{len(mc_returns_preview)} rendimenti giornalieri · {len(mc_selected)} strategie."
                    )
                    if len(mc_returns_preview) < 252:
                        st.warning("Campione inferiore a un anno: le code della distribuzione e i risultati "
                                   "su 3–5 anni possono essere molto instabili.")
                    elif mc_years * 252 > len(mc_returns_preview) * 3:
                        st.warning("L'orizzonte simulato è molto più lungo dello storico osservato: "
                                   "gli scenari ripetono necessariamente caratteristiche del campione disponibile.")
                except ValueError as mc_error:
                    st.warning(str(mc_error))

            if st.button("Run Monte Carlo Simulation", type="primary", key="mc_run", disabled=not mc_selected):
                try:
                    mc_returns = mc_aligned_returns({name: mc_available[name] for name in mc_selected})
                    with st.spinner("Simulazione degli scenari in corso..."):
                        mc_metrics, mc_percentiles, mc_finals, mc_trajectories = mc_simulate(
                            mc_returns, years=int(mc_years), n_paths=int(mc_paths),
                            initial_capital=float(mc_capital), method=mc_method,
                            block_size=int(mc_block), seed=int(mc_seed),
                            annual_drift_adjustment=float(mc_drift) / 100,
                            return_paths=True,
                        )
                    st.session_state["mc_results"] = {
                        "metrics": mc_metrics, "percentiles": mc_percentiles,
                        "finals": mc_finals, "trajectories": mc_trajectories, "source": mc_source_name,
                        "strategies": tuple(mc_selected), "years": mc_years,
                        "paths": mc_paths, "method": mc_method, "capital": mc_capital,
                        "seed": mc_seed, "drift": mc_drift, "block": mc_block,
                        "sample": len(mc_returns),
                    }
                except Exception as mc_error:
                    st.error(f"Monte Carlo Simulation failed: {mc_error}")

            mc_saved = st.session_state.get("mc_results")
            if mc_saved:
                st.divider()
                st.subheader("Monte Carlo — Results")
                st.caption(
                    f"Fonte: {mc_saved['source']} · {mc_saved['years']} anni · "
                    f"{mc_saved['paths']:,} simulazioni · {mc_saved['method']} · "
                    f"{mc_saved['sample']} rendimenti storici. "
                    "Risultati riferiti alla configurazione dell'ultima esecuzione."
                )
                if mc_saved['source'] != mc_source_name or tuple(mc_selected) != mc_saved['strategies']:
                    st.warning("La selezione è cambiata: premi Run Monte Carlo Simulation per aggiornare i risultati.")
                # Dark, understated per-column gradients consistent with the other
                # Quant Portfolio Optimizer tables (no bright matplotlib palettes).
                mc_green_cols = ["Median final (€)", "Mean final (€)",
                                 "P5 final (€)", "P95 final (€)"]
                mc_red_cols = ["Probability of loss (%)", "VaR 95% (€)",
                               "ES 95% (€)", "Probability MDD ≤ -20% (%)"]

                def _mc_dark_gradient(column, low_rgb, high_rgb):
                    numeric = pd.to_numeric(column, errors="coerce")
                    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
                    if finite.empty:
                        return ["" for _ in column]
                    lo, hi = float(finite.min()), float(finite.max())
                    styles = []
                    for val in numeric:
                        if pd.isna(val) or not np.isfinite(val):
                            styles.append("")
                            continue
                        # Keep a visible, dark base and scale the saturation per metric.
                        t = (float(val) - lo) / (hi - lo) if hi > lo else 0.5
                        t = 0.12 + 0.88 * t
                        rgb = tuple(round(a + (b - a) * t)
                                    for a, b in zip(low_rgb, high_rgb))
                        styles.append(
                            f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); "
                            "color: #f2f4f7; font-weight: 500;"
                        )
                    return styles

                mc_styled = mc_saved["metrics"].style.format({
                    "Median final (€)": "€{:,.2f}", "Mean final (€)": "€{:,.2f}",
                    "P5 final (€)": "€{:,.2f}", "P95 final (€)": "€{:,.2f}",
                    "Probability of loss (%)": "{:.1f}%", "VaR 95% (€)": "€{:,.2f}",
                    "ES 95% (€)": "€{:,.2f}", "Probability MDD ≤ -20% (%)": "{:.1f}%",
                })
                for mc_col in mc_green_cols:
                    mc_styled = mc_styled.apply(
                        _mc_dark_gradient, subset=[mc_col], axis=0,
                        low_rgb=(18, 28, 25), high_rgb=(29, 111, 68),
                    )
                for mc_col in mc_red_cols:
                    mc_styled = mc_styled.apply(
                        _mc_dark_gradient, subset=[mc_col], axis=0,
                        low_rgb=(28, 22, 26), high_rgb=(124, 38, 49),
                    )
                st.dataframe(mc_styled, use_container_width=True)
                st.caption("VaR 95%: perdita rispetto al capitale iniziale al 5° percentile. "
                           "ES 95%: perdita media nei peggiori 5% degli scenari. "
                           "Le perdite sono riportate come zero se il corrispondente risultato è positivo. "
                           "MDD: massimo drawdown all'interno di ciascuna traiettoria.")
                mc_view = st.selectbox("Strategia da visualizzare", list(mc_saved["percentiles"]),
                                       key="mc_view_strategy")
                mc_p = mc_saved["percentiles"][mc_view]
                mc_show_paths = st.checkbox("Mostra tutte le traiettorie simulate", value=False,
                    key="mc_show_all_paths",
                    help="Mostra ogni percorso Monte Carlo come linea sottile e trasparente dietro il ventaglio. "
                         "Per mantenere il grafico fluido, la visualizzazione campiona le sedute, "
                         "ma i calcoli utilizzano tutte le osservazioni.")
                mc_fig = go.Figure()
                if mc_show_paths:
                    mc_paths_array = mc_saved.get("trajectories", {}).get(mc_view)
                    if mc_paths_array is None:
                        st.info("Riesegui la simulazione per visualizzare le singole traiettorie.")
                    else:
                        # One WebGL trace with NaN separators avoids thousands of traces.
                        import numpy as _mc_np
                        mc_stride = max(1, int(_mc_np.ceil((mc_paths_array.shape[1] - 1) / 300)))
                        mc_idx = _mc_np.unique(_mc_np.r_[_mc_np.arange(0, mc_paths_array.shape[1], mc_stride),
                                                        mc_paths_array.shape[1] - 1])
                        mc_sample = mc_paths_array[:, mc_idx]
                        mc_x = _mc_np.broadcast_to(mc_idx, mc_sample.shape)
                        mc_fig.add_trace(go.Scattergl(
                            x=_mc_np.column_stack([mc_x, _mc_np.full((len(mc_x), 1), _mc_np.nan)]).ravel(),
                            y=_mc_np.column_stack([mc_sample, _mc_np.full((len(mc_sample), 1), _mc_np.nan)]).ravel(),
                            mode="lines", line=dict(color="rgba(125,190,235,0.10)", width=0.6),
                            name=f"{len(mc_sample):,} simulazioni", hoverinfo="skip"))
                
                mc_fig.add_trace(go.Scatter(x=mc_p.index, y=mc_p["P95"],
                    line=dict(width=0), name="95° percentile", hovertemplate="Seduta %{x}: €%{y:,.0f}<extra></extra>"))
                mc_fig.add_trace(go.Scatter(x=mc_p.index, y=mc_p["P5"],
                    fill="tonexty", fillcolor="rgba(56,189,248,0.07)" if mc_show_paths else "rgba(56,189,248,0.12)",
                    line=dict(width=0), name="5°–95° percentile"))
                mc_fig.add_trace(go.Scatter(x=mc_p.index, y=mc_p["P75"],
                    line=dict(width=0), name="75° percentile"))
                mc_fig.add_trace(go.Scatter(x=mc_p.index, y=mc_p["P25"],
                    fill="tonexty", fillcolor="rgba(56,189,248,0.13)" if mc_show_paths else "rgba(56,189,248,0.22)",
                    line=dict(width=0), name="25°–75° percentile"))
                mc_fig.add_trace(go.Scatter(x=mc_p.index, y=mc_p["Median"],
                    line=dict(color="#38bdf8", width=3), name="Mediana"))
                mc_fig.add_hline(y=float(mc_saved["capital"]), line_dash="dash",
                    line_color="rgba(255,255,255,0.55)", annotation_text="Capitale iniziale")
                mc_fig.update_layout(title=f"{mc_view} — ventaglio delle traiettorie simulate",
                    xaxis_title="Sedute future", yaxis_title="Valore portafoglio (€)",
                    template="plotly_dark", hovermode="x unified")
                st.plotly_chart(mc_fig, use_container_width=True, key="mc_fan_chart")
                mc_hist = go.Figure()
                for mc_name in mc_saved["finals"].columns:
                    mc_hist.add_trace(go.Histogram(x=mc_saved["finals"][mc_name],
                        name=mc_name, opacity=0.55, nbinsx=55))
                mc_hist.add_vline(x=float(mc_saved["capital"]), line_dash="dash",
                    line_color="white")
                mc_hist.update_layout(title="Distribuzione dei valori finali",
                    xaxis_title="Valore finale (€)", yaxis_title="Numero di scenari",
                    barmode="overlay", template="plotly_dark")
                st.plotly_chart(mc_hist, use_container_width=True, key="mc_distribution")
                st.download_button("Download Monte Carlo Metrics (CSV)",
                    mc_saved["metrics"].to_csv().encode("utf-8"),
                    file_name="monte_carlo_metrics.csv", mime="text/csv", key="mc_download_metrics")
                st.download_button("Download Terminal Values (CSV)",
                    mc_saved["finals"].to_csv(index=False).encode("utf-8"),
                    file_name="monte_carlo_terminal_values.csv", mime="text/csv", key="mc_download_finals")
                st.info("Le simulazioni ripetono o parametrizzano la distribuzione osservata nel campione: "
                        "non modellano automaticamente cambiamenti strutturali, fiscalità, ordini futuri "
                        "o la riottimizzazione futura dei portafogli.")


            st.divider()
            st.subheader("Monte Carlo — Robustness & Method Comparison")
            st.caption(
                "Confronta gli stessi portafogli su campioni storici differenti e con tre metodi: "
                "Block bootstrap, IID bootstrap e normale multivariata. "
                "Questa analisi non cambia le simulazioni principali."
            )
            rb1, rb2, rb3 = st.columns(3)
            with rb1:
                mc_rb_paths = st.selectbox("Simulazioni per combinazione", [300, 500, 1000], index=1,
                    key="mc_rb_paths", help="Numero di scenari per ogni coppia campione/metodo. "
                    "Più scenari riducono il rumore Monte Carlo, ma aumentano i tempi di calcolo.")
            with rb2:
                mc_rb_min = st.selectbox("Minimo sedute per sottocampione", [60, 100, 126, 252], index=1,
                    key="mc_rb_min", help="I sottocampioni troppo corti sono esclusi. "
                    "Con un anno di storico, le stime delle code restano comunque fragili.")
            with rb3:
                mc_rb_seed = st.number_input("Seed del confronto", min_value=0, max_value=1000000,
                    value=42, step=1, key="mc_rb_seed", help="Rende replicabile il confronto statistico.")
            st.caption("Campioni: intero periodo, prima metà, seconda metà e ultimo 75% "
                       "(quando contengono abbastanza osservazioni). Sono parzialmente sovrapposti: "
                       "non costituiscono test indipendenti né una validazione walk-forward.")
            if st.button("Run Robustness Comparison", type="primary", key="mc_rb_run",
                         disabled=not mc_selected):
                try:
                    mc_rb_returns = mc_aligned_returns({name: mc_available[name] for name in mc_selected})
                    n_rb = len(mc_rb_returns)
                    candidate_samples = {
                        "Intero periodo": mc_rb_returns,
                        "Prima metà": mc_rb_returns.iloc[:n_rb // 2],
                        "Seconda metà": mc_rb_returns.iloc[n_rb // 2:],
                        "Ultimo 75%": mc_rb_returns.iloc[n_rb // 4:],
                    }
                    samples = {k: v for k, v in candidate_samples.items()
                               if len(v) >= max(40, int(mc_rb_min))}
                    methods = ["Block bootstrap", "IID bootstrap", "Multivariate Gaussian"]
                    rows = []
                    with st.spinner(f"Confronto di {len(samples) * len(methods)} configurazioni in corso..."):
                        for sample_name, sample_returns in samples.items():
                            for method_idx, method_name in enumerate(methods):
                                m, _, _ = mc_simulate(
                                    sample_returns, years=int(mc_years), n_paths=int(mc_rb_paths),
                                    initial_capital=float(mc_capital), method=method_name,
                                    block_size=int(mc_block),
                                    seed=int(mc_rb_seed) + method_idx,
                                    annual_drift_adjustment=float(mc_drift) / 100,
                                    return_paths=False,
                                )
                                for strategy_name, vals in m.iterrows():
                                    rows.append({
                                        "Strategy": strategy_name, "Sample": sample_name,
                                        "Method": method_name, "Observations": len(sample_returns),
                                        "Start": sample_returns.index[0].strftime("%Y-%m-%d"),
                                        "End": sample_returns.index[-1].strftime("%Y-%m-%d"),
                                        **vals.to_dict(),
                                    })
                    st.session_state["mc_rb_results"] = {
                        "table": pd.DataFrame(rows), "source": mc_source_name,
                        "strategies": tuple(mc_selected), "years": int(mc_years),
                        "capital": float(mc_capital), "drift": float(mc_drift),
                        "block": int(mc_block), "paths": int(mc_rb_paths),
                        "min_obs": int(mc_rb_min), "seed": int(mc_rb_seed),
                    }
                except Exception as rb_error:
                    st.error(f"Monte Carlo robustness comparison failed: {rb_error}")

            mc_rb_saved = st.session_state.get("mc_rb_results")
            if mc_rb_saved:
                rb_table = mc_rb_saved["table"]
                if (mc_rb_saved["source"] != mc_source_name or
                        mc_rb_saved["strategies"] != tuple(mc_selected) or
                        mc_rb_saved["years"] != int(mc_years) or
                        mc_rb_saved["capital"] != float(mc_capital) or
                        mc_rb_saved["drift"] != float(mc_drift) or
                        mc_rb_saved["block"] != int(mc_block) or
                        mc_rb_saved["paths"] != int(mc_rb_paths) or
                        mc_rb_saved["min_obs"] != int(mc_rb_min) or
                        mc_rb_saved["seed"] != int(mc_rb_seed)):
                    st.warning("I parametri sono cambiati: riesegui Run Robustness Comparison per aggiornare la tabella.")
                st.caption(f"{len(rb_table)} risultati · "
                           f"{rb_table['Sample'].nunique()} campioni · "
                           f"{rb_table['Method'].nunique()} metodi · "
                           f"{mc_rb_saved['paths']} scenari per combinazione.")
                rb_strategy = st.selectbox("Strategia per il confronto", rb_table["Strategy"].unique(),
                    key="mc_rb_view_strategy")
                rb_metric_options = ["Median final (€)", "Probability of loss (%)",
                                     "VaR 95% (€)", "ES 95% (€)",
                                     "Probability MDD ≤ -20% (%)"]
                rb_metric = st.selectbox("Metrica da confrontare", rb_metric_options,
                    key="mc_rb_metric")
                rb_view = rb_table.loc[rb_table["Strategy"] == rb_strategy]
                rb_pivot = rb_view.pivot(index="Sample", columns="Method", values=rb_metric)
                rb_order = [x for x in ["Intero periodo", "Prima metà", "Seconda metà", "Ultimo 75%"]
                            if x in rb_pivot.index]
                rb_pivot = rb_pivot.reindex(rb_order)
                rb_fig = go.Figure()
                for rb_method in rb_pivot.columns:
                    rb_fig.add_trace(go.Bar(name=rb_method, x=rb_pivot.index,
                                            y=rb_pivot[rb_method]))
                rb_fig.update_layout(title=f"{rb_strategy} — {rb_metric}",
                    xaxis_title="Campione storico", yaxis_title=rb_metric,
                    barmode="group", template="plotly_dark")
                st.plotly_chart(rb_fig, use_container_width=True, key="mc_rb_chart")
                st.dataframe(rb_pivot.style.format("{:,.2f}"), use_container_width=True)
                st.caption("La differenza tra campioni misura la sensibilità al periodo storico; "
                           "la differenza tra metodi misura la sensibilità alle ipotesi distributive. "
                           "Con pochi rendimenti, VaR, ES e probabilità di drawdown possono cambiare molto.")
                st.download_button("Download Monte Carlo Robustness (CSV)",
                    rb_table.to_csv(index=False).encode("utf-8"),
                    file_name="monte_carlo_robustness_comparison.csv",
                    mime="text/csv", key="mc_rb_download")


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


# =========================================================
# VOLATILITY ANALYTICS & FORECASTING (ISOLATED SECTION)
# =========================================================
with volatility_tab:
    st.header("Volatility Analytics & Forecasting")
    st.caption("Historical and forward-looking risk estimates; not predictions of price direction.")
    v_mode = st.radio("Analysis mode", ["Portfolio Volatility", "Asset Volatility"], horizontal=True, key="vol_analysis_mode")
    if v_mode == "Portfolio Volatility":
        v_strategy = st.selectbox("Portfolio strategy", ["FF3 Maximum Sharpe", "FF5 Maximum Sharpe", "CAPM Maximum Sharpe", "Maximum Sharpe", "Equal Weight"], key="vol_portfolio_strategy")
        v_state_map = {
            "FF3 Maximum Sharpe": ("ff3_results", "weights"),
            "FF5 Maximum Sharpe": ("ff5_results", "weights"),
            "CAPM Maximum Sharpe": ("factor_model_results", "weights"),
            "Maximum Sharpe": ("optimization_results", "max_weights"),
            "Equal Weight": ("optimization_results", "max_weights"),
        }
        v_key, v_weight_key = v_state_map[v_strategy]
        v_state = st.session_state.get(v_key)
        if v_state is None or v_weight_key not in v_state:
            st.info("First run the corresponding Portfolio Optimization or Factor Model to generate current target weights.")
        else:
            v_weights = pd.Series(v_state[v_weight_key], dtype=float)
            v_weights = v_weights[v_weights > 1e-10]
            if v_strategy == "Equal Weight":
                v_weights = pd.Series(1.0/len(v_state["prices"].columns), index=v_state["prices"].columns)
            st.caption(f"{len(v_weights)} holdings from current {v_strategy} allocation. Fixed-weight, daily-rebalanced risk proxy (not an executed backtest).")
            v_c1, v_c2, v_c3, v_c4 = st.columns(4)
            with v_c1:
                v_period = st.selectbox("Historical period", ["1y", "2y", "3y", "5y", "10y"], index=2, key="vol_port_period")
            with v_c2:
                v_window = st.selectbox("Rolling window (sessions)", [21,63,126,252], index=1, key="vol_port_window")
            with v_c3:
                v_horizon = st.selectbox("Forecast horizon (sessions)", [5,21,63,126], index=1, key="vol_port_horizon")
            with v_c4:
                v_decay = st.slider("EWMA decay", 0.80, 0.99, 0.94, 0.01, key="vol_port_decay")
            v_b1,v_b2 = st.columns(2)
            with v_b1:
                v_train = st.number_input("OOS training sessions", min_value=252, max_value=1250, value=500, step=25, key="vol_port_train")
            with v_b2:
                v_stride = st.selectbox("OOS step (sessions)", [5,21,63], index=1, key="vol_port_stride")
            if st.button("Run Portfolio Volatility Analysis", type="primary", key="vol_port_run"):
                try:
                    with st.spinner("Downloading constituent prices and estimating portfolio risk..."):
                        v_px = load_volatility_prices(list(v_weights.index), v_period)
                        v_returns = portfolio_log_returns(v_px, v_weights)
                        v_rolling = v_returns.rolling(v_window).std()*np.sqrt(252)*100
                        from src.volatility_forecasting import ewma_variance
                        v_ewma = pd.Series(np.sqrt(ewma_variance(v_returns.to_numpy(),v_decay)*252)*100,index=v_returns.index)
                        v_hist = pd.DataFrame({"Rolling":v_rolling,"EWMA":v_ewma})
                        v_models = ["Rolling", "EWMA", "GARCH(1,1)", "GJR-GARCH", "EGARCH"] if ARCH_AVAILABLE else ["Rolling", "EWMA"]
                        v_paths, v_errors = {}, {}
                        for v_m in v_models:
                            try:
                                v_paths[v_m] = forecast_volatility_path(v_returns, v_horizon, v_m, v_window, v_decay)
                                if v_m not in ("Rolling","EWMA"):
                                    v_hist[v_m] = filtered_garch_volatility(v_returns, v_m)
                            except Exception as exc:
                                v_errors[v_m] = str(exc)
                        v_contrib, v_port_sigma = portfolio_risk_contributions(v_px, v_weights)
                        st.session_state["vol_port_results"] = dict(history=v_hist,paths=v_paths,errors=v_errors,returns=v_returns,contrib=v_contrib,sigma=v_port_sigma,weights=v_weights,params=dict(horizon=v_horizon,window=v_window,decay=v_decay,train=v_train,stride=v_stride,strategy=v_strategy,period=v_period))
                        st.session_state.pop("vol_port_validation", None)
                except Exception as exc:
                    st.error(f"Portfolio volatility analysis failed: {exc}")
            v_result = st.session_state.get("vol_port_results")
            if v_result:
                v_p = v_result["params"]
                if v_p["strategy"] != v_strategy:
                    st.info("Displayed results refer to the previously analyzed strategy. Run again to update.")
                st.subheader("Portfolio risk overview")
                st.metric("Realized portfolio volatility (annualized)", f"{v_result['sigma']:.2f}%")
                v_forecasts = pd.DataFrame([{"Model":m,"Forecast volatility (%)":float(np.sqrt(np.mean((path.to_numpy()/100)**2))*100)} for m,path in v_result["paths"].items()])
                st.dataframe(v_forecasts.style.format({"Forecast volatility (%)":"{:.2f}"}),use_container_width=True,hide_index=True)
                st.caption("Forecast summary is sqrt(mean of predicted daily variances), annualized. In-sample GARCH curves are filtered estimates, NOT historical out-of-sample forecasts.")
                # Advisory-only Dynamic Risk Management; no orders or backtest changes.
                from src.risk_alerts import risk_alert
                st.subheader("Dynamic Risk Management — volatility alerts")
                st.caption("Traffic-light alerts are based on forecast annualized volatility, not on predicted price direction. Thresholds are provisional and configurable.")
                dr_c1, dr_c2, dr_c3 = st.columns(3)
                with dr_c1:
                    dr_model = st.selectbox("Risk signal model", list(v_result["paths"].keys()),
                                            index=list(v_result["paths"].keys()).index("EWMA") if "EWMA" in v_result["paths"] else 0,
                                            key="dr_risk_model")
                with dr_c2:
                    dr_target = st.number_input("Target annual volatility (%)", min_value=1.0, max_value=50.0,
                                                value=15.0, step=1.0, key="dr_target")
                with dr_c3:
                    dr_current = st.number_input("Current invested exposure (%)", min_value=0.0, max_value=100.0,
                                                 value=100.0, step=5.0, key="dr_current")
                with st.expander("Alert thresholds and rebalancing bands", expanded=False):
                    dr_t1, dr_t2, dr_t3 = st.columns(3)
                    with dr_t1:
                        dr_caution = st.number_input("Yellow from (%)", 1.0, 100.0, 20.0, 1.0, key="dr_caution")
                    with dr_t2:
                        dr_warning = st.number_input("Orange from (%)", 1.0, 100.0, 25.0, 1.0, key="dr_warning")
                    with dr_t3:
                        dr_critical = st.number_input("Red from (%)", 1.0, 100.0, 30.0, 1.0, key="dr_critical")
                    dr_b1, dr_b2 = st.columns(2)
                    with dr_b1:
                        dr_review = st.number_input("Review exposure gap (pp)", 0.0, 50.0, 5.0, 1.0, key="dr_review")
                    with dr_b2:
                        dr_rebalance = st.number_input("Rebalance review gap (pp)", 1.0, 75.0, 10.0, 1.0, key="dr_rebalance")
                try:
                    dr_forecast = float(v_forecasts.set_index("Model").loc[dr_model, "Forecast volatility (%)"])
                    dr = risk_alert(dr_forecast, dr_target, dr_current, dr_caution,
                                    dr_warning, dr_critical, dr_review, dr_rebalance)
                    dr_cols = st.columns(4)
                    dr_kpis = [
                        ("FORECAST RISK", f"{dr['forecast']:.2f}%", dr['status'], dr['color']),
                        ("TARGET EXPOSURE", f"{dr['suggested_exposure']:.1f}%", "FF3 risky allocation", "#3b82f6"),
                        ("SUGGESTED CASH", f"{dr['suggested_cash']:.1f}%", "Before trading costs", "#64748b"),
                        ("REBALANCING", f"{dr['gap_pp']:.1f} pp", dr['action'],
                         "#ef4444" if dr['action']=="REBALANCE REVIEW" else "#eab308" if dr['action']=="MONITOR / REVIEW" else "#22c55e"),
                    ]
                    for dr_col, (dr_title, dr_value, dr_label, dr_color) in zip(dr_cols, dr_kpis):
                        with dr_col:
                            st.markdown(
                                f'<div style="background:#151b25;border:1px solid {dr_color};border-left:5px solid {dr_color};'
                                f'border-radius:12px;padding:15px 12px;min-height:126px">'
                                f'<div style="font-size:12px;color:#b8c2d0;font-weight:700">{dr_title}</div>'
                                f'<div style="font-size:27px;color:#fff;font-weight:750;margin:7px 0">{dr_value}</div>'
                                f'<div style="font-size:12px;color:{dr_color};font-weight:700">{dr_label}</div></div>',
                                unsafe_allow_html=True)
                    if dr['status']=="CRITICAL":
                        st.error(f"RED ALERT: {dr_model} forecasts {dr_forecast:.2f}% annual volatility (threshold {dr_critical:.1f}%). Review risk exposure; no trade is executed.")
                    elif dr['status']=="HIGH":
                        st.warning(f"HIGH RISK: {dr_model} forecasts {dr_forecast:.2f}% (orange threshold {dr_warning:.1f}%).")
                    elif dr['status']=="ELEVATED":
                        st.warning(f"ELEVATED RISK: {dr_model} forecasts {dr_forecast:.2f}% (yellow threshold {dr_caution:.1f}%).")
                    else:
                        st.success(f"NORMAL RISK: {dr_model} forecast {dr_forecast:.2f}% is below the yellow threshold ({dr_caution:.1f}%).")
                    st.caption(
                        f"Exposure gap = |current {dr_current:.1f}% − proposed {dr['suggested_exposure']:.1f}%|. "
                        f"Review from {dr_review:.1f} pp; rebalance review from {dr_rebalance:.1f} pp. "
                        "Exposure is calculated as min(100%, target volatility / forecast volatility). "
                        "The dashboard does not know your broker positions: enter current exposure manually. "
                        "Signals are advisory, not automatic sell orders. No transaction costs or cash yield are included in this indication."
                    )
                except (ValueError, KeyError) as dr_exc:
                    st.warning(f"Risk alerts unavailable: {dr_exc}")

                v_fig=go.Figure()
                for v_m in v_result["history"].columns:
                    v_fig.add_trace(go.Scatter(x=v_result["history"].index,y=v_result["history"][v_m],name=v_m))
                v_fig.update_layout(title="Portfolio — historical annualized volatility",yaxis_title="Volatility (%)",hovermode="x unified")
                st.plotly_chart(v_fig,use_container_width=True)
                v_fig2=go.Figure()
                for v_m,v_path in v_result["paths"].items():
                    v_fig2.add_trace(go.Scatter(x=v_path.index,y=v_path.values,name=v_m))
                v_fig2.update_layout(title=f"Forecast volatility path — next {v_p['horizon']} sessions",xaxis_title="Trading sessions ahead",yaxis_title="Annualized volatility (%)",hovermode="x unified")
                st.plotly_chart(v_fig2,use_container_width=True)
                st.subheader("Portfolio risk contributions")
                st.dataframe(v_result["contrib"].style.format("{:.2f}"),use_container_width=True)
                st.caption("Euler contributions based on historical covariance. Assumes target weights are held fixed and rebalanced daily; excludes costs, FX conversions and future changes in weights.")
                if v_result["errors"]:
                    st.warning("Some models could not be estimated: " + "; ".join(f"{k}: {v}" for k,v in v_result["errors"].items()))
                if st.button("Validate all models out of sample",key="vol_port_validate"):
                    with st.spinner("Rolling-origin validation across the same dates (may take several minutes)..."):
                        try:
                            v_validation,v_scores,v_errors=compare_volatility_models(v_result["returns"],v_p["horizon"],v_p["train"],v_p["stride"],v_p["window"],v_p["decay"])
                            st.session_state["vol_port_validation"]=(v_validation,v_scores,v_errors)
                        except Exception as exc:
                            st.error(f"Portfolio volatility validation failed: {exc}")
                if st.session_state.get("vol_port_validation") is not None:
                    v_validation,v_scores,v_errors=st.session_state["vol_port_validation"]
                    st.subheader("Out-of-sample model comparison")
                    if v_scores.empty:
                        st.info("Not enough common forecast windows. Increase history or shorten training period.")
                    else:
                        st.dataframe(v_scores,hide_index=True,use_container_width=True)
                        st.caption("Lower QLIKE and variance MSE are better; all models use identical forecast origins. QLIKE can be negative. The number of OOS windows is shown explicitly.")
                        st.dataframe(v_validation.tail(30),use_container_width=True)
                        st.download_button("Download portfolio forecast validation CSV",v_validation.to_csv(index=False).encode(),"portfolio_volatility_validation.csv","text/csv",key="vol_port_csv")
                    if v_errors:
                        st.warning("Validation model errors: "+"; ".join(f"{k}: {v}" for k,v in v_errors.items()))
    else:
        v_source = st.radio("Asset source", ["Current asset selection", "Manual selection"], horizontal=True, key="vol_source")
        v_universe = market_instruments.drop_duplicates(subset="ticker").copy()
        v_name_to_ticker = dict(zip(v_universe["name"].astype(str) + " (" + v_universe["ticker"].astype(str) + ")", v_universe["ticker"].astype(str)))
        if v_source == "Current asset selection":
            # Prefer the current sidebar selection when it contains assets.
            # A completed Quant Screening must also be available here when the
            # sidebar remains in Manual Selection with no manually chosen tickers.
            v_sidebar_tickers = list(dict.fromkeys(tickers))
            v_screening_state = st.session_state.get("screening_results")
            v_screening_valid = (
                isinstance(v_screening_state, dict)
                and v_screening_state.get("market") == market
            )
            v_market_tickers = set(v_universe["ticker"].astype(str))
            v_quant_tickers = (
                [t for t in st.session_state.get("quant_selected_tickers", [])
                 if t in v_market_tickers]
                if v_screening_valid else []
            )
            v_tickers = v_sidebar_tickers if v_sidebar_tickers else list(dict.fromkeys(v_quant_tickers))
            v_origin = (
                "current sidebar selection" if v_sidebar_tickers
                else "latest Quant Selection screening" if v_quant_tickers
                else "current selection"
            )
            st.caption(f"{len(v_tickers)} assets from {v_origin}.")
        else:
            v_choices = st.multiselect("Assets to analyze", options=list(v_name_to_ticker), key="vol_manual_assets")
            v_tickers = [v_name_to_ticker[x] for x in v_choices]
        v_c1, v_c2, v_c3, v_c4 = st.columns(4)
        with v_c1:
            v_period = st.selectbox("Historical period", ["1y", "2y", "3y", "5y", "10y"], index=2, key="vol_period")
        with v_c2:
            v_window = st.selectbox("Rolling window (sessions)", [21, 63, 126, 252], index=1, key="vol_window")
        with v_c3:
            v_horizon = st.selectbox("Forecast horizon (sessions)", [5, 21, 63, 126], index=1, key="vol_horizon")
        with v_c4:
            v_model = st.selectbox("Forecast model", ["Rolling", "EWMA", "GARCH(1,1)", "GJR-GARCH", "EGARCH"], index=2, key="vol_model")
        v_d1, v_d2, v_d3 = st.columns(3)
        with v_d1:
            v_lambda = st.slider("EWMA decay", 0.80, 0.99, 0.94, 0.01, key="vol_decay")
        with v_d2:
            v_train = st.number_input("Forecast backtest training sessions", min_value=252, max_value=1250, value=500, step=25, key="vol_train")
        with v_d3:
            v_stride = st.selectbox("Backtest step (sessions)", [5, 21, 63], index=1, key="vol_stride")
        if v_model in ("GARCH(1,1)", "GJR-GARCH", "EGARCH") and not ARCH_AVAILABLE:
            st.warning("GARCH models require the 'arch' package. Add arch to requirements.txt and redeploy; Rolling/EWMA work without it.")
        if st.button("Run Volatility Analysis", type="primary", key="vol_run"):
            if not v_tickers:
                st.warning("Select one or more assets, or run Quant Selection first.")
            elif len(v_tickers) > 30:
                st.warning("Choose at most 30 assets per analysis to keep computation manageable.")
            else:
                with st.spinner("Downloading adjusted prices and estimating volatility..."):
                    try:
                        v_prices = load_volatility_prices(v_tickers, v_period)
                        v_results = {}
                        for v_ticker in v_tickers:
                            if v_ticker not in v_prices or v_prices[v_ticker].dropna().shape[0] < max(v_window + 5, 90):
                                continue
                            v_results[v_ticker] = analyze_volatility(v_prices[v_ticker], v_window, v_horizon, v_model, v_lambda)
                        st.session_state["vol_results"] = v_results
                        st.session_state["vol_run_params"] = dict(model=v_model, horizon=v_horizon, train=v_train, stride=v_stride, window=v_window, decay=v_lambda)
                    except Exception as e:
                        st.error(f"Volatility analysis failed: {e}")
        v_results = st.session_state.get("vol_results", {})
        if v_results:
            v_summary = pd.DataFrame([dict(Asset=k, **r["summary"]) for k, r in v_results.items()]).set_index("Asset")
            st.subheader("Asset volatility overview")
            st.dataframe(v_summary.style.format({"Rolling volatility (%)": "{:.2f}", "EWMA volatility (%)": "{:.2f}", "Forecast volatility (%)": "{:.2f}", "Volatility percentile (%)": "{:.1f}"}), use_container_width=True)
            st.download_button("Download volatility summary CSV", v_summary.to_csv().encode("utf-8"), "volatility_summary.csv", "text/csv", key="vol_export")
            v_asset = st.selectbox("Inspect asset", list(v_results), key="vol_inspect")
            v_result = v_results[v_asset]
            v_series = v_result["history"]
            v_fig = go.Figure()
            v_fig.add_trace(go.Scatter(x=v_series.index, y=v_series["Rolling"], name="Rolling annualized volatility"))
            v_fig.add_trace(go.Scatter(x=v_series.index, y=v_series["EWMA"], name="EWMA annualized volatility"))
            v_fig.update_layout(title=f"{v_asset} — Historical volatility", yaxis_title="Annualized volatility (%)", xaxis_title="Date", hovermode="x unified")
            st.plotly_chart(v_fig, use_container_width=True)
            st.metric(f"{st.session_state['vol_run_params']['model']} forecast ({st.session_state['vol_run_params']['horizon']} sessions)", f"{v_result['summary']['Forecast volatility (%)']:.2f}%")
            st.caption("Forecast is the annualized volatility implied by the average predicted daily variance over the chosen horizon. It does not predict return direction.")
            if st.button("Validate forecast out of sample", key="vol_validate"):
                with st.spinner("Rolling-origin forecast validation (no future observations in estimation)..."):
                    try:
                        p = st.session_state["vol_run_params"]
                        v_validation = backtest_volatility(v_result["returns"], p["model"], p["horizon"], p["train"], p["stride"], p["window"], p["decay"])
                        st.session_state["vol_validation"] = (v_asset, v_validation)
                    except Exception as e:
                        st.error(f"Forecast validation failed: {e}")
            v_stored = st.session_state.get("vol_validation")
            if v_stored and v_stored[0] == v_asset:
                v_validation = v_stored[1]
                if v_validation.empty:
                    st.info("Insufficient history for the chosen training window and forecast horizon. Increase the historical period.")
                else:
                    st.subheader("Out-of-sample forecast accuracy")
                    v_err = pd.DataFrame({"QLIKE": v_validation.attrs.get("qlike", {}), "MSE (variance)": v_validation.attrs.get("mse", {})})
                    st.dataframe(v_err, use_container_width=True)
                    st.caption("Lower QLIKE and MSE are better. All forecasts are compared to the same realized horizon variance. This is model validation, not a trading signal.")
                    st.dataframe(v_validation.tail(30), use_container_width=True)
                    st.download_button("Download forecast validation CSV", v_validation.to_csv().encode("utf-8"), "volatility_forecast_validation.csv", "text/csv", key="vol_valid_export")
