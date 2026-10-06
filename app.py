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
# Custom metric cards
# --------------------------------------------------

def colored_metric(label, value, color):

    colors = {
        "green": {
            "bg": "rgba(34, 197, 94, 0.09)",
            "border": "rgba(34, 197, 94, 0.12)",
        },
        "red": {
            "bg": "rgba(239, 68, 68, 0.09)",
            "border": "rgba(239, 68, 68, 0.12)",
        },
        "blue": {
            "bg": "rgba(56, 189, 248, 0.09)",
            "border": "rgba(56, 189, 248, 0.12)",
        },
        "purple": {
            "bg": "rgba(168, 85, 247, 0.09)",
            "border": "rgba(168, 85, 247, 0.12)",
        },
        "yellow": {
            "bg": "rgba(245, 158, 11, 0.09)",
            "border": "rgba(245, 158, 11, 0.12)",
        },
    }

    c = colors[color]

    html = (
        f'<div style="'
        f'background:{c["bg"]};'
        f'border:1px solid {c["border"]};'
        f'border-radius:10px;'
        f'padding:14px 16px;'
        f'min-height:88px;'
        f'box-sizing:border-box;'
        f'">'
        f'<div style="'
        f'font-size:0.78rem;'
        f'opacity:0.72;'
        f'margin-bottom:6px;'
        f'">'
        f'{label}'
        f'</div>'
        f'<div style="'
        f'font-size:1.65rem;'
        f'font-weight:600;'
        f'line-height:1.2;'
        f'">'
        f'{value}'
        f'</div>'
        f'</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Portfolio allocation title
# --------------------------------------------------

def portfolio_allocation_title():

    st.markdown(
        """
        <div style="
            margin-top: 16px;
            margin-bottom: 8px;
            font-size: 0.95rem;
            font-weight: 500;
        ">
            Portfolio allocation
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Portfolio table styling
# --------------------------------------------------

def style_portfolio_table(df, use_fractional_shares):

    styled_df = df.style


    # --------------------------------------------------
    # Generic heatmap function
    # --------------------------------------------------

    def make_heatmap(column, rgb, min_alpha=0.06, max_alpha=0.32):

        positive_values = df.loc[
            df[column] > 0,
            column
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
                + (max_alpha - min_alpha) * intensity
            )

            return (
                f"background-color: rgba({rgb}, {alpha:.3f}); "
                "color: #ffffff; "
                "font-weight: 600;"
            )

        return color_cell


    # --------------------------------------------------
    # Weight heatmap
    # Purple
    # --------------------------------------------------

    weight_color = make_heatmap(
        "Weight",
        "168, 85, 247"
    )

    if weight_color is not None:

        styled_df = styled_df.map(
            weight_color,
            subset=["Weight"]
        )


    # --------------------------------------------------
    # Price heatmap
    # Green
    # --------------------------------------------------

    price_color = make_heatmap(
        "Price",
        "34, 197, 94"
    )

    if price_color is not None:

        styled_df = styled_df.map(
            price_color,
            subset=["Price"]
        )


    # --------------------------------------------------
    # Quantity heatmap
    # Blue
    # --------------------------------------------------

    quantity_color = make_heatmap(
        "Quantity",
        "56, 189, 248"
    )

    if quantity_color is not None:

        styled_df = styled_df.map(
            quantity_color,
            subset=["Quantity"]
        )


    # --------------------------------------------------
    # Quantity = 0
    # Entire row red
    # --------------------------------------------------

    def highlight_zero_quantity(row):

        if row["Quantity"] == 0:

            return [
                "background-color: rgba(239, 68, 68, 0.13); "
                "color: #ffb4b4;"
            ] * len(row)

        return [""] * len(row)


    styled_df = styled_df.apply(
        highlight_zero_quantity,
        axis=1
    )


    # --------------------------------------------------
    # Number formatting
    # --------------------------------------------------

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
            "Invested": "€{:,.2f}"
        }
    )

    return styled_df


# --------------------------------------------------
# Load instruments catalog
# --------------------------------------------------

INSTRUMENTS_FILE = "data/instruments.csv"

instruments = pd.read_csv(
    INSTRUMENTS_FILE
)


# --------------------------------------------------
# Maps
# --------------------------------------------------

name_by_ticker = dict(
    zip(
        instruments["ticker"],
        instruments["name"]
    )
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


ticker_by_name_market = dict(
    zip(
        market_instruments["name"],
        market_instruments["ticker"]
    )
)


selected_names = st.sidebar.multiselect(
    "Select instruments",
    options=market_instruments["name"].tolist(),
    default=[],
    format_func=lambda name: (
        f"{name} ({ticker_by_name_market[name]})"
    ),
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

        st.header(
            "Portfolio Results"
        )

        col1, col2 = st.columns(2)


        # ==================================================
        # Minimum Volatility
        # ==================================================

        with col1:

            st.subheader(
                "Minimum Volatility"
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)


            with metric_col1:

                colored_metric(
                    "Expected Return",
                    f"{min_return:.2%}",
                    "green"
                )


            with metric_col2:

                colored_metric(
                    "Volatility",
                    f"{min_volatility:.2%}",
                    "red"
                )


            with metric_col3:

                colored_metric(
                    "Sharpe Ratio",
                    f"{min_sharpe:.2f}",
                    "blue"
                )


            portfolio_allocation_title()


            min_weights_display = (
                min_weights
                .rename("Weight")
                .to_frame()
            )


            min_weights_display.insert(
                0,
                "Instrument",
                min_weights_display.index.map(
                    lambda ticker: name_by_ticker.get(
                        ticker,
                        ticker
                    )
                )
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
            # Price
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
            # Invested
            # --------------------------------------------------

            min_weights_display[
                "Invested"
            ] = (
                min_weights_display["Quantity"]
                * min_weights_display["Price"]
            )


            # --------------------------------------------------
            # Sort
            # --------------------------------------------------

            min_weights_display = (
                min_weights_display
                .sort_values(
                    by="Invested",
                    ascending=False
                )
                .reset_index(drop=True)
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
            # Styled table
            # --------------------------------------------------

            min_table = style_portfolio_table(
                min_weights_display,
                use_fractional_shares
            )


            st.dataframe(
                min_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Instrument": st.column_config.TextColumn(
                        "Instrument",
                        width="small",
                        help="Nome completo dello strumento"
                    ),
                    "Weight": st.column_config.Column(
                        "Weight",
                        width="small"
                    ),
                    "Target Amount": st.column_config.Column(
                        "Target Amount",
                        width="small"
                    ),
                    "Price": st.column_config.Column(
                        "Price",
                        width="small"
                    ),
                    "Quantity": st.column_config.Column(
                        "Quantity",
                        width="small"
                    ),
                    "Invested": st.column_config.Column(
                        "Invested",
                        width="small"
                    ),
                }
            )


            # --------------------------------------------------
            # Capital cards
            # --------------------------------------------------

            summary_col1, summary_col2 = st.columns(2)


            with summary_col1:

                colored_metric(
                    "Total Invested",
                    f"€{min_total_invested:,.2f}",
                    "purple"
                )


            with summary_col2:

                colored_metric(
                    "Residual Cash",
                    f"€{min_cash:,.2f}",
                    "yellow"
                )


        # ==================================================
        # Maximum Sharpe
        # ==================================================

        with col2:

            st.subheader(
                "Maximum Sharpe"
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)


            with metric_col1:

                colored_metric(
                    "Expected Return",
                    f"{max_return:.2%}",
                    "green"
                )


            with metric_col2:

                colored_metric(
                    "Volatility",
                    f"{max_volatility:.2%}",
                    "red"
                )


            with metric_col3:

                colored_metric(
                    "Sharpe Ratio",
                    f"{max_sharpe:.2f}",
                    "blue"
                )


            portfolio_allocation_title()


            max_weights_display = (
                max_weights
                .rename("Weight")
                .to_frame()
            )


            max_weights_display.insert(
                0,
                "Instrument",
                max_weights_display.index.map(
                    lambda ticker: name_by_ticker.get(
                        ticker,
                        ticker
                    )
                )
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
            # Price
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
            # Invested
            # --------------------------------------------------

            max_weights_display[
                "Invested"
            ] = (
                max_weights_display["Quantity"]
                * max_weights_display["Price"]
            )


            # --------------------------------------------------
            # Sort
            # --------------------------------------------------

            max_weights_display = (
                max_weights_display
                .sort_values(
                    by="Invested",
                    ascending=False
                )
                .reset_index(drop=True)
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
            # Styled table
            # --------------------------------------------------

            max_table = style_portfolio_table(
                max_weights_display,
                use_fractional_shares
            )


            st.dataframe(
                max_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Instrument": st.column_config.TextColumn(
                        "Instrument",
                        width="small",
                        help="Nome completo dello strumento"
                    ),
                    "Weight": st.column_config.Column(
                        "Weight",
                        width="small"
                    ),
                    "Target Amount": st.column_config.Column(
                        "Target Amount",
                        width="small"
                    ),
                    "Price": st.column_config.Column(
                        "Price",
                        width="small"
                    ),
                    "Quantity": st.column_config.Column(
                        "Quantity",
                        width="small"
                    ),
                    "Invested": st.column_config.Column(
                        "Invested",
                        width="small"
                    ),
                }
            )


            # --------------------------------------------------
            # Capital cards
            # --------------------------------------------------

            summary_col1, summary_col2 = st.columns(2)


            with summary_col1:

                colored_metric(
                    "Total Invested",
                    f"€{max_total_invested:,.2f}",
                    "purple"
                )


            with summary_col2:

                colored_metric(
                    "Residual Cash",
                    f"€{max_cash:,.2f}",
                    "yellow"
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
        # Asset Statistics
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