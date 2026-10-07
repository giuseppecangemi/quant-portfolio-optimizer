import numpy as np
import pandas as pd

from src.factors import capm_expected_returns

from src.optimization import (
    optimize_minimum_volatility,
    optimize_maximum_sharpe,
    optimize_risk_parity,
    optimize_hrp,
)


def backtest_portfolio(
    prices: pd.DataFrame,
    strategy: str = "max_sharpe",
    initial_capital: float = 10000.0,
    lookback_days: int = 252,
    rebalance_days: int = 63,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    market_prices: pd.DataFrame | pd.Series | None = None,
):
    """
    Esegue un backtest walk-forward realistico.

    Il portafoglio viene ribilanciato solo alle date previste.
    Tra due ribilanciamenti le quantità degli asset rimangono
    costanti e i pesi cambiano con il movimento dei prezzi.

    Returns
    -------
    backtest : pd.DataFrame
        Serie storica con:
        - Portfolio Value
        - Daily Return

    weights_history : pd.DataFrame
        Pesi target calcolati a ogni data di ribilanciamento.
    """

    valid_strategies = [
        "max_sharpe",
        "capm_max_sharpe",
        "min_volatility",
        "risk_parity",
        "hrp",
        "equal_weight",
    ]

    if strategy not in valid_strategies:
        raise ValueError(
            "Strategy must be 'max_sharpe', 'capm_max_sharpe', "
            "'min_volatility', 'risk_parity', 'hrp' "
            "or 'equal_weight'."
        )

    if prices.empty:
        raise ValueError(
            "Il DataFrame dei prezzi è vuoto."
        )

    prices = (
        prices
        .sort_index()
        .dropna()
    )

    if len(prices) <= lookback_days:
        raise ValueError(
            "Non ci sono abbastanza dati storici "
            "per il lookback selezionato."
        )

    n_assets = prices.shape[1]

    if max_weight < 1 / n_assets:
        raise ValueError(
            "Il peso massimo per asset è troppo basso "
            "rispetto al numero di asset."
        )

    portfolio_values = []
    weights_history = []

    current_portfolio_value = float(
        initial_capital
    )

    # Primo giorno in cui possiamo costruire il portafoglio.
    start_position = lookback_days

    # =====================================================
    # WALK-FORWARD BACKTEST
    # =====================================================

    for rebalance_start in range(
        start_position,
        len(prices) - 1,
        rebalance_days
    ):

        rebalance_date = (
            prices.index[rebalance_start]
        )

        # -------------------------------------------------
        # 1. DATI DISPONIBILI PRIMA DEL RIBILANCIAMENTO
        # -------------------------------------------------

        historical_prices = prices.iloc[
            rebalance_start - lookback_days:
            rebalance_start
        ]

        historical_returns = (
            historical_prices
            .pct_change()
            .dropna()
        )

        expected_returns = (
            historical_returns.mean()
            * 252
        )

        covariance = (
            historical_returns.cov()
            * 252
        )

        # -------------------------------------------------
        # 2. CALCOLO PESI TARGET
        # -------------------------------------------------

        if strategy == "max_sharpe":

            target_weights = (
                optimize_maximum_sharpe(
                    expected_returns,
                    covariance,
                    risk_free_rate=
                        risk_free_rate,
                    max_weight=
                        max_weight
                )
            )

        elif strategy == "capm_max_sharpe":

            if market_prices is None:
                raise ValueError(
                    "market_prices è obbligatorio per la strategia CAPM."
                )

            market_series = (
                market_prices.iloc[:, 0]
                if isinstance(market_prices, pd.DataFrame)
                else market_prices
            )

            historical_market_prices = (
                market_series
                .reindex(historical_prices.index)
                .dropna()
            )

            market_returns = (
                historical_market_prices
                .pct_change()
                .dropna()
            )

            capm_returns = capm_expected_returns(
                historical_returns,
                market_returns,
                risk_free_rate=risk_free_rate,
            )

            target_weights = (
                optimize_maximum_sharpe(
                    capm_returns,
                    covariance,
                    risk_free_rate=risk_free_rate,
                    max_weight=max_weight,
                )
            )

        elif strategy == "min_volatility":

            target_weights = (
                optimize_minimum_volatility(
                    expected_returns,
                    covariance,
                    max_weight=
                        max_weight
                )
            )

        elif strategy == "risk_parity":

            target_weights = (
                optimize_risk_parity(
                    covariance,
                    max_weight=
                        max_weight
                )
            )

        elif strategy == "hrp":

            target_weights = (
                optimize_hrp(
                    covariance,
                    max_weight=
                        max_weight
                )
            )

        else:

            target_weights = pd.Series(
                np.ones(n_assets) / n_assets,
                index=prices.columns
            )

        # Sicurezza numerica.
        target_weights = (
            target_weights
            .reindex(prices.columns)
            .fillna(0.0)
        )

        target_weights[
            target_weights.abs() < 1e-10
        ] = 0.0

        target_weights = (
            target_weights
            / target_weights.sum()
        )

        # Salviamo i pesi scelti.
        weights_row = (
            target_weights.copy()
        )

        weights_row.name = (
            rebalance_date
        )

        weights_history.append(
            weights_row
        )

        # -------------------------------------------------
        # 3. ACQUISTO DEL PORTAFOGLIO
        # -------------------------------------------------

        rebalance_prices = (
            prices.iloc[rebalance_start]
        )

        allocated_capital = (
            current_portfolio_value
            * target_weights
        )

        # Nel backtest utilizziamo quote frazionarie
        # per misurare la strategia teorica senza
        # distorsioni dovute all'arrotondamento.
        quantities = (
            allocated_capital
            / rebalance_prices
        )

        # -------------------------------------------------
        # 4. HOLD FINO AL PROSSIMO RIBILANCIAMENTO
        # -------------------------------------------------

        rebalance_end = min(
            rebalance_start
            + rebalance_days,
            len(prices) - 1
        )

        holding_prices = prices.iloc[
            rebalance_start:
            rebalance_end + 1
        ]

        period_values = (
            holding_prices
            * quantities
        ).sum(axis=1)

        # Evitiamo di duplicare la data iniziale
        # del periodo precedente.
        if portfolio_values:
            period_values = (
                period_values.iloc[1:]
            )

        portfolio_values.append(
            period_values
        )

        current_portfolio_value = float(
            period_values.iloc[-1]
        )

    # =====================================================
    # 5. UNIONE DELLA SERIE STORICA
    # =====================================================

    if not portfolio_values:
        raise ValueError(
            "Il backtest non ha prodotto risultati."
        )

    portfolio_value = pd.concat(
        portfolio_values
    )

    portfolio_value = portfolio_value[
        ~portfolio_value.index.duplicated(
            keep="first"
        )
    ]

    daily_return = (
        portfolio_value
        .pct_change()
        .fillna(0.0)
    )

    backtest = pd.DataFrame(
        {
            "Portfolio Value":
                portfolio_value,

            "Daily Return":
                daily_return,
        }
    )

    weights_history = pd.DataFrame(
        weights_history
    )

    weights_history.index.name = (
        "Rebalance Date"
    )

    return backtest, weights_history


def calculate_backtest_metrics(
    backtest: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calcola le principali metriche del backtest.
    """

    if backtest.empty:
        raise ValueError(
            "Il backtest è vuoto."
        )

    daily_returns = (
        backtest["Daily Return"]
        .dropna()
    )

    portfolio_value = (
        backtest["Portfolio Value"]
    )

    if len(portfolio_value) < 2:
        raise ValueError(
            "Il backtest non contiene abbastanza dati."
        )

    # -------------------------------------------------
    # TOTAL RETURN
    # -------------------------------------------------

    initial_value = float(
        portfolio_value.iloc[0]
    )

    final_value = float(
        portfolio_value.iloc[-1]
    )

    total_return = (
        final_value / initial_value
    ) - 1

    # -------------------------------------------------
    # CAGR
    # -------------------------------------------------

    days = (
        portfolio_value.index[-1]
        - portfolio_value.index[0]
    ).days

    years = days / 365.25

    if years > 0:

        cagr = (
            final_value / initial_value
        ) ** (1 / years) - 1

    else:

        cagr = 0.0

    # -------------------------------------------------
    # VOLATILITÀ
    # -------------------------------------------------

    annualized_volatility = (
        daily_returns.std()
        * np.sqrt(252)
    )

    # -------------------------------------------------
    # RENDIMENTO ANNUALIZZATO
    # -------------------------------------------------

    annualized_return = (
        daily_returns.mean()
        * 252
    )

    # -------------------------------------------------
    # SHARPE RATIO
    # -------------------------------------------------

    if annualized_volatility > 0:

        sharpe_ratio = (
            annualized_return
            - risk_free_rate
        ) / annualized_volatility

    else:

        sharpe_ratio = 0.0

    # -------------------------------------------------
    # SORTINO RATIO
    # -------------------------------------------------

    downside_returns = daily_returns[
        daily_returns < 0
    ]

    downside_deviation = (
        downside_returns.std()
        * np.sqrt(252)
    )

    if (
        not np.isnan(
            downside_deviation
        )
        and downside_deviation > 0
    ):

        sortino_ratio = (
            annualized_return
            - risk_free_rate
        ) / downside_deviation

    else:

        sortino_ratio = 0.0

    # -------------------------------------------------
    # MAX DRAWDOWN
    # -------------------------------------------------

    running_max = (
        portfolio_value.cummax()
    )

    drawdown = (
        portfolio_value
        / running_max
    ) - 1

    max_drawdown = float(
        drawdown.min()
    )

    # -------------------------------------------------
    # RISULTATI
    # -------------------------------------------------

    return {
        "Total Return":
            float(total_return),

        "CAGR":
            float(cagr),

        "Volatility":
            float(
                annualized_volatility
            ),

        "Sharpe Ratio":
            float(sharpe_ratio),

        "Sortino Ratio":
            float(sortino_ratio),

        "Max Drawdown":
            max_drawdown,

        "Final Value":
            final_value,
    }