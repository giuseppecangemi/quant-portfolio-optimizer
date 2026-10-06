import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from src.data import get_prices
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
    efficient_frontier,
)
from src.visualization import plot_efficient_frontier
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


# =========================================================
# DATA
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
INSTRUMENTS_FILE = PROJECT_DIR / "data" / "instruments.csv"

instruments = pd.read_csv(INSTRUMENTS_FILE)

FTSE_MIB_TICKER = "FTSEMIB.MI"


# =========================================================
# HELPERS
# =========================================================

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
):
    """
    Fixed-Horizon Backtest.

    L'ottimizzazione viene eseguita UNA SOLA VOLTA alla fine
    del periodo di estimation.

    I pesi ottenuti vengono poi mantenuti invariati durante
    tutto il periodo di backtest.
    """

    prices = (
        prices
        .sort_index()
        .dropna()
    )

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

    # Minimum Volatility calcolato una sola volta
    min_vol_weights = (
        optimize_minimum_volatility(
            expected_returns,
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

    min_vol_bt = build_buy_and_hold(
        min_vol_weights
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

    ftse_prices = get_prices(
        [FTSE_MIB_TICKER],
        period=benchmark_period,
    )

    ftse_mib_bt = (
        build_benchmark_backtest(
            ftse_prices,
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

    min_bt_metrics = (
        calculate_backtest_metrics(
            min_vol_bt,
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

        "min_vol_bt":
            min_vol_bt,

        "equal_weight_bt":
            equal_weight_bt,

        "ftse_mib_bt":
            ftse_mib_bt,

        "max_sharpe_weights_history":
            max_sharpe_weights_history,

        "min_vol_weights_history":
            min_vol_weights_history,

        "equal_weight_weights_history":
            equal_weight_weights_history,

        "max_bt_metrics":
            max_bt_metrics,

        "min_bt_metrics":
            min_bt_metrics,

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
):

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
    )

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
    )

    # -----------------------------------------------------
    # FIXED-HORIZON TRIM
    # -----------------------------------------------------

    if fixed_backtest_days is not None:

        max_sharpe_bt = (
            max_sharpe_bt
            .iloc[:fixed_backtest_days + 1]
        )

        min_vol_bt = (
            min_vol_bt
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

        min_vol_weights_history = (
            min_vol_weights_history.loc[
                min_vol_weights_history.index
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

    ftse_prices = get_prices(
        [FTSE_MIB_TICKER],
        period=benchmark_period,
    )

    ftse_mib_bt = (
        build_benchmark_backtest(
            ftse_prices,
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

    min_bt_metrics = (
        calculate_backtest_metrics(
            min_vol_bt,
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

        "min_vol_bt":
            min_vol_bt,

        "equal_weight_bt":
            equal_weight_bt,

        "ftse_mib_bt":
            ftse_mib_bt,

        "max_sharpe_weights_history":
            max_sharpe_weights_history,

        "min_vol_weights_history":
            min_vol_weights_history,

        "equal_weight_weights_history":
            equal_weight_weights_history,

        "max_bt_metrics":
            max_bt_metrics,

        "min_bt_metrics":
            min_bt_metrics,

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

            "Minimum Volatility":
                result["min_bt_metrics"],

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

    portfolio_fig.update_yaxes(
        tickprefix="€",
        tickformat=",.0f",
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

    min_drawdown = (
        calculate_drawdown(
            result["min_vol_bt"]
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
            x=min_drawdown.index,
            y=min_drawdown,
            mode="lines",
            name="Minimum Volatility",
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
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode="x unified",
        yaxis_tickformat=".0%",
        legend_title="Strategy",
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
            "Minimum Volatility",
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
        == "Minimum Volatility"
    ):

        selected_weights_history = (
            result[
                "min_vol_weights_history"
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

    final_col1, final_col2, final_col3, final_col4 = (
        st.columns(4)
    )

    final_max = (
        result["max_sharpe_bt"]
        ["Portfolio Value"]
        .iloc[-1]
    )

    final_min = (
        result["min_vol_bt"]
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
        "Minimum Volatility",
        f"€{final_min:,.2f}",
        delta=(
            f"{final_min / capital - 1:.2%}"
        ),
    )

    final_col3.metric(
        "Equal Weight",
        f"€{final_equal:,.2f}",
        delta=(
            f"{final_equal / capital - 1:.2%}"
        ),
    )

    final_col4.metric(
        "FTSE MIB",
        f"€{final_ftse:,.2f}",
        delta=(
            f"{final_ftse / capital - 1:.2%}"
        ),
    )


# =========================================================
# SESSION STATE
# =========================================================

if "optimization_results" not in st.session_state:
    st.session_state.optimization_results = None

if "standard_backtest_results" not in st.session_state:
    st.session_state.standard_backtest_results = None

if "fixed_backtest_results" not in st.session_state:
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
# INSTRUMENTS
# ---------------------------------------------------------

selected_names = (
    st.sidebar.multiselect(
        "Select instruments",
        options=market_instruments[
            "name"
        ].tolist(),
        default=[],
        placeholder=(
            "Type to search a company..."
        ),
    )
)

selected_instruments = (
    market_instruments[
        market_instruments[
            "name"
        ].isin(selected_names)
    ]
)

tickers = (
    selected_instruments[
        "ticker"
    ].tolist()
)

if selected_instruments.empty:

    st.sidebar.caption(
        "Nessuno strumento selezionato."
    )

else:

    st.sidebar.caption(
        f"{len(selected_instruments)} "
        f"strumenti selezionati"
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
# MAIN NAVIGATION
# =========================================================

optimization_tab, backtest_tab = st.tabs(
    [
        "Portfolio Optimization",
        "Backtesting",
    ]
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
                [
                    "1y",
                    "2y",
                    "5y",
                    "10y",
                ],
                index=2,
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

                        "frontier":
                            frontier,

                        "min_table":
                            min_table,

                        "max_table":
                            max_table,

                        "min_total_invested":
                            min_total_invested,

                        "min_cash":
                            min_cash,

                        "max_total_invested":
                            max_total_invested,

                        "max_cash":
                            max_cash,
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
                [
                    "1y",
                    "2y",
                    "5y",
                    "10y",
                ],
                index=2,
                key="standard_period",
            )
        )

        standard_lookback_options = {
            "6 months": 126,
            "1 year": 252,
            "2 years": 504,
            "3 years": 756,
            "5 years": 1260,
        }

        standard_lookback_label = (
            st.selectbox(
                "Estimation window",
                list(
                    standard_lookback_options.keys()
                ),
                index=1,
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

        st.info(
            "Example\n\n"
            "Historical dataset: 5 years\n\n"
            "Estimation window: 1 year\n\n"
            "→ approximately 4 years of "
            "walk-forward backtest."
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
            "1 year": 252,
            "2 years": 504,
            "3 years": 756,
            "5 years": 1260,
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
            "1 year": 252,
            "2 years": 504,
            "3 years": 756,
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

            # =================================================
            # STANDARD WALK-FORWARD
            # =================================================

            if run_standard:

                try:

                    with st.spinner(
                        "Running Standard Walk-Forward backtest..."
                    ):

                        standard_prices = (
                            get_prices(
                                tickers,
                                period=standard_period,
                            )
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

                        standard_result = (
                            run_backtest_engine(
                                tickers=tickers,
                                prices=standard_prices,
                                lookback_days=
                                    standard_lookback_days,
                                rebalance_days=
                                    standard_rebalance_days,
                                risk_free_rate=
                                    risk_free_rate,
                                max_weight=
                                    max_weight,
                                capital=
                                    capital,
                                fixed_backtest_days=
                                    None,
                                benchmark_period=
                                    standard_period,
                            )
                        )

                        standard_result[
                            "lookback_label"
                        ] = (
                            standard_lookback_label
                        )

                        standard_result[
                            "period_label"
                        ] = (
                            "Remaining historical period"
                        )

                        standard_result[
                            "rebalance_label"
                        ] = (
                            standard_rebalance_label
                        )

                        standard_result[
                            "historical_dataset"
                        ] = (
                            standard_period
                        )

                        st.session_state[
                            "standard_backtest_results"
                        ] = standard_result

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

                        fixed_prices = (
                            get_fixed_horizon_prices(
                                tickers,
                                fixed_lookback_days,
                                fixed_backtest_days,
                            )
                        )

                        fixed_result = (
                            run_fixed_horizon_engine(
                                tickers=tickers,
                                prices=fixed_prices,
                                estimation_days=
                                    fixed_lookback_days,
                                backtest_days=
                                    fixed_backtest_days,
                                risk_free_rate=
                                    risk_free_rate,
                                max_weight=
                                    max_weight,
                                capital=
                                    capital,
                                benchmark_period=
                                    "10y",
                            )
                        )

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

        display_backtest_results(
            result=standard_result,

            title=(
                "Standard Walk-Forward Backtest"
            ),

            description=(
                "The historical dataset contains both "
                "the initial estimation window and the "
                "subsequent out-of-sample simulation. "
                "At every rebalance the estimation "
                "window rolls forward using only "
                "information available at that date."
            ),

            lookback_label=
                standard_result[
                    "lookback_label"
                ],

            backtest_period_label=
                standard_result[
                    "period_label"
                ],

            rebalance_label=
                standard_result[
                    "rebalance_label"
                ],

            result_key="standard",
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