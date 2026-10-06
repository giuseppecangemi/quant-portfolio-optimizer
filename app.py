import streamlit as st
import pandas as pd

from src.data import get_prices
from src.returns import calculate_returns, annualized_returns
from src.risk import (
    annualized_volatility,
    covariance_matrix,
    portfolio_return,
    portfolio_volatility
)
from src.optimization import (
    optimize_minimum_volatility,
    optimize_maximum_sharpe,
    efficient_frontier
)
from src.visualization import plot_efficient_frontier


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Quant Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)


# --------------------------------------------------
# Load instruments catalog
# --------------------------------------------------

INSTRUMENTS_FILE = "data/instruments.csv"

instruments = pd.read_csv(
    INSTRUMENTS_FILE
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("Quant Portfolio Optimizer")

st.write(
    """
    Quantitative portfolio analysis and optimization
    based on Modern Portfolio Theory.
    """
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Portfolio Parameters")


# --------------------------------------------------
# Market and instruments
# --------------------------------------------------

market = st.sidebar.selectbox(
    "Market",
    sorted(
        instruments["market"].unique()
    )
)

market_instruments = instruments[
    instruments["market"] == market
]

# Mappa nome -> ticker, per mostrare "Nome (TICKER)" nel menu
ticker_by_name = dict(
    zip(
        market_instruments["name"],
        market_instruments["ticker"]
    )
)

selected_names = st.sidebar.multiselect(
    "Select instruments",
    options=market_instruments["name"].tolist(),
    default=[],
    format_func=lambda name: f"{name} ({ticker_by_name[name]})",
    placeholder="Scrivi per cercare (nome o ticker)..."
)

selected_instruments = market_instruments[
    market_instruments["name"].isin(
        selected_names
    )
]

tickers = selected_instruments[
    "ticker"
].tolist()

if selected_instruments.empty:

    st.sidebar.caption(
        "Nessuno strumento selezionato."
    )

else:

    st.sidebar.caption(
        f"{len(selected_instruments)} "
        f"strumenti selezionati"
    )

# --------------------------------------------------
# Historical period
# --------------------------------------------------

period = st.sidebar.selectbox(
    "Historical period",
    ["1y", "2y", "5y", "10y"],
    index=0
)


# --------------------------------------------------
# Risk-free rate
# --------------------------------------------------

risk_free_rate = st.sidebar.number_input(
    "Risk-free rate",
    min_value=0.0,
    max_value=0.20,
    value=0.0,
    step=0.005,
    format="%.3f"
)


# --------------------------------------------------
# Investment capital
# --------------------------------------------------

capital = st.sidebar.number_input(
    "Investment capital",
    min_value=100.0,
    value=10000.0,
    step=1000.0,
    format="%.2f"
)


# --------------------------------------------------
# Fractional shares
# --------------------------------------------------

st.sidebar.subheader("Quote frazionarie")

use_fractional_shares = st.sidebar.toggle(
    "Utilizza quote frazionarie",
    value=False
)

st.sidebar.info(
    "Attivando questa opzione, il portafoglio può utilizzare "
    "quote frazionarie (es. 3,42 azioni) per avvicinarsi "
    "maggiormente all'allocazione ottimale. "
    "Se disattivata, vengono utilizzate solo quote intere "
    "e il capitale non investito rimane come liquidità residua."
)


# --------------------------------------------------
# Maximum asset weight
# --------------------------------------------------

st.sidebar.subheader("Diversificazione")

use_max_weight = st.sidebar.toggle(
    "Limita peso massimo per asset",
    value=False
)

if use_max_weight:

    max_weight = st.sidebar.slider(
        "Peso massimo per singolo asset",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
        format="%d%%"
    )

    max_weight = max_weight / 100

    st.sidebar.info(
        f"Ogni singolo asset potrà rappresentare al massimo "
        f"il {max_weight:.0%} del portafoglio."
    )

else:

    max_weight = 1.0


# --------------------------------------------------
# Run analysis
# --------------------------------------------------

run_analysis = st.sidebar.button(
    "Run analysis"
)


# --------------------------------------------------
# Main analysis
# --------------------------------------------------

if run_analysis:

    if len(tickers) < 2:

        st.error(
            "Seleziona almeno due strumenti."
        )

        st.stop()

    try:

        # --------------------------------------------------
        # Download prices
        # --------------------------------------------------

        with st.spinner(
            "Downloading market data..."
        ):

            prices = get_prices(
                tickers,
                period=period
            )

        # --------------------------------------------------
        # Returns
        # --------------------------------------------------

        returns = calculate_returns(
            prices
        )

        expected_returns = annualized_returns(
            returns
        )

        # --------------------------------------------------
        # Risk
        # --------------------------------------------------

        volatility = annualized_volatility(
            returns
        )

        covariance = covariance_matrix(
            returns
        )

        # --------------------------------------------------
        # Optimization
        # --------------------------------------------------

        min_weights = optimize_minimum_volatility(
            expected_returns,
            covariance,
            max_weight=max_weight
        )

        max_weights = optimize_maximum_sharpe(
            expected_returns,
            covariance,
            risk_free_rate,
            max_weight=max_weight
        )

        # --------------------------------------------------
        # Portfolio metrics
        # --------------------------------------------------

        min_return = portfolio_return(
            min_weights.values,
            expected_returns
        )

        min_volatility = portfolio_volatility(
            min_weights.values,
            covariance
        )

        min_sharpe = (
            (min_return - risk_free_rate)
            / min_volatility
            if min_volatility > 0
            else 0
        )

        max_return = portfolio_return(
            max_weights.values,
            expected_returns
        )

        max_volatility = portfolio_volatility(
            max_weights.values,
            covariance
        )

        max_sharpe = (
            (max_return - risk_free_rate)
            / max_volatility
            if max_volatility > 0
            else 0
        )

        # --------------------------------------------------
        # Efficient frontier
        # --------------------------------------------------

        frontier = efficient_frontier(
            expected_returns,
            covariance,
            max_weight=max_weight,
            points=100
        )

        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        st.header("Portfolio Results")

        col1, col2 = st.columns(2)

        # ==================================================
        # Minimum Volatility
        # ==================================================

        with col1:

            st.subheader(
                "Minimum Volatility"
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            metric_col1.metric(
                "Expected Return",
                f"{min_return:.2%}"
            )

            metric_col2.metric(
                "Volatility",
                f"{min_volatility:.2%}"
            )

            metric_col3.metric(
                "Sharpe Ratio",
                f"{min_sharpe:.2f}"
            )

            st.write(
                "Portfolio allocation"
            )

            min_weights_display = (
                min_weights
                .rename("Weight")
                .to_frame()
            )

            # --------------------------------------------------
            # Target amount
            # --------------------------------------------------

            min_weights_display[
                "Target Amount"
            ] = (
                min_weights * capital
            )

            # --------------------------------------------------
            # Latest available market price
            # --------------------------------------------------

            min_weights_display[
                "Price"
            ] = prices.iloc[-1]

            # --------------------------------------------------
            # Quantity
            # --------------------------------------------------

            if use_fractional_shares:

                min_weights_display[
                    "Quantity"
                ] = (
                    min_weights_display[
                        "Target Amount"
                    ]
                    / min_weights_display["Price"]
                )

            else:

                min_weights_display[
                    "Quantity"
                ] = (
                    min_weights_display[
                        "Target Amount"
                    ]
                    / min_weights_display["Price"]
                ).astype(int)

            # --------------------------------------------------
            # Actual amount invested
            # --------------------------------------------------

            min_weights_display[
                "Invested"
            ] = (
                min_weights_display["Quantity"]
                * min_weights_display["Price"]
            )

            # --------------------------------------------------
            # Capital summary
            # --------------------------------------------------

            min_total_invested = (
                min_weights_display[
                    "Invested"
                ].sum()
            )

            min_cash = (
                capital
                - min_total_invested
            )

            # --------------------------------------------------
            # Formatting
            # --------------------------------------------------

            min_weights_display[
                "Weight"
            ] = (
                min_weights_display["Weight"]
                .map(
                    lambda x: f"{x:.2%}"
                )
            )

            min_weights_display[
                "Target Amount"
            ] = (
                min_weights_display[
                    "Target Amount"
                ]
                .map(
                    lambda x: f"€{x:,.2f}"
                )
            )

            min_weights_display[
                "Price"
            ] = (
                min_weights_display["Price"]
                .map(
                    lambda x: f"€{x:,.2f}"
                )
            )

            if use_fractional_shares:

                min_weights_display[
                    "Quantity"
                ] = (
                    min_weights_display[
                        "Quantity"
                    ]
                    .map(
                        lambda x: f"{x:.4f}"
                    )
                )

            else:

                min_weights_display[
                    "Quantity"
                ] = (
                    min_weights_display[
                        "Quantity"
                    ]
                    .map(
                        lambda x: f"{int(x)}"
                    )
                )

            min_weights_display[
                "Invested"
            ] = (
                min_weights_display["Invested"]
                .map(
                    lambda x: f"€{x:,.2f}"
                )
            )

            st.dataframe(
                min_weights_display,
                use_container_width=True
            )

            # --------------------------------------------------
            # Capital summary
            # --------------------------------------------------

            summary_col1, summary_col2 = st.columns(2)

            summary_col1.metric(
                "Total Invested",
                f"€{min_total_invested:,.2f}"
            )

            summary_col2.metric(
                "Residual Cash",
                f"€{min_cash:,.2f}"
            )

        # ==================================================
        # Maximum Sharpe
        # ==================================================

        with col2:

            st.subheader(
                "Maximum Sharpe"
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            metric_col1.metric(
                "Expected Return",
                f"{max_return:.2%}"
            )

            metric_col2.metric(
                "Volatility",
                f"{max_volatility:.2%}"
            )

            metric_col3.metric(
                "Sharpe Ratio",
                f"{max_sharpe:.2f}"
            )

            st.write(
                "Portfolio allocation"
            )

            max_weights_display = (
                max_weights
                .rename("Weight")
                .to_frame()
            )

            # --------------------------------------------------
            # Target amount
            # --------------------------------------------------

            max_weights_display[
                "Target Amount"
            ] = (
                max_weights * capital
            )

            # --------------------------------------------------
            # Latest available market price
            # --------------------------------------------------

            max_weights_display[
                "Price"
            ] = prices.iloc[-1]

            # --------------------------------------------------
            # Quantity
            # --------------------------------------------------

            if use_fractional_shares:

                max_weights_display[
                    "Quantity"
                ] = (
                    max_weights_display[
                        "Target Amount"
                    ]
                    / max_weights_display["Price"]
                )

            else:

                max_weights_display[
                    "Quantity"
                ] = (
                    max_weights_display[
                        "Target Amount"
                    ]
                    / max_weights_display["Price"]
                ).astype(int)

            # --------------------------------------------------
            # Actual amount invested
            # --------------------------------------------------

            max_weights_display[
                "Invested"
            ] = (
                max_weights_display["Quantity"]
                * max_weights_display["Price"]
            )

            # --------------------------------------------------
            # Capital summary
            # --------------------------------------------------

            max_total_invested = (
                max_weights_display[
                    "Invested"
                ].sum()
            )

            max_cash = (
                capital
                - max_total_invested
            )

            # --------------------------------------------------
            # Formatting
            # --------------------------------------------------

            max_weights_display[
                "Weight"
            ] = (
                max_weights_display["Weight"]
                .map(
                    lambda x: f"{x:.2%}"
                )
            )

            max_weights_display[
                "Target Amount"
            ] = (
                max_weights_display[
                    "Target Amount"
                ]
                .map(
                    lambda x: f"€{x:,.2f}"
                )
            )

            max_weights_display[
                "Price"
            ] = (
                max_weights_display["Price"]
                .map(
                    lambda x: f"€{x:,.2f}"
                )
            )

            if use_fractional_shares:

                max_weights_display[
                    "Quantity"
                ] = (
                    max_weights_display[
                        "Quantity"
                    ]
                    .map(
                        lambda x: f"{x:.4f}"
                    )
                )

            else:

                max_weights_display[
                    "Quantity"
                ] = (
                    max_weights_display[
                        "Quantity"
                    ]
                    .map(
                        lambda x: f"{int(x)}"
                    )
                )

            max_weights_display[
                "Invested"
            ] = (
                max_weights_display["Invested"]
                .map(
                    lambda x: f"€{x:,.2f}"
                )
            )

            st.dataframe(
                max_weights_display,
                use_container_width=True
            )

            # --------------------------------------------------
            # Capital summary
            # --------------------------------------------------

            summary_col1, summary_col2 = st.columns(2)

            summary_col1.metric(
                "Total Invested",
                f"€{max_total_invested:,.2f}"
            )

            summary_col2.metric(
                "Residual Cash",
                f"€{max_cash:,.2f}"
            )

        # ==================================================
        # Efficient Frontier
        # ==================================================

        st.header(
            "Efficient Frontier"
        )

        fig = plot_efficient_frontier(
            frontier,
            expected_returns,
            covariance,
            min_weights,
            max_weights,
            risk_free_rate
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==================================================
        # Historical Asset Statistics
        # ==================================================

        st.header(
            "Asset Statistics"
        )

        asset_statistics = pd.DataFrame({
            "Expected Return": expected_returns,
            "Volatility": volatility
        })

        st.dataframe(
            asset_statistics.style.format(
                {
                    "Expected Return": "{:.2%}",
                    "Volatility": "{:.2%}"
                }
            ),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Analysis failed: {e}"
        )

else:

    st.info(
        "Configure the portfolio parameters in the sidebar "
        "and click **Run analysis**."
    )
