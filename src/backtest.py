import numpy as np

import pandas as pd



from src.factors import capm_expected_returns

from src.screening import screen_prices_point_in_time



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

    selection_top_n: int | None = None,

    selection_policy: str = "fixed",

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


    valid_selection_policies = ["fixed", "dynamic"]

    if selection_policy not in valid_selection_policies:

        raise ValueError(

            "selection_policy must be 'fixed' or 'dynamic'."

        )



    if prices.empty:

        raise ValueError(

            "Il DataFrame dei prezzi è vuoto."

        )



    # Manteniamo i NaN dell'universo completo. Un dropna globale

    # introdurrebbe una selezione implicita dei soli titoli con storia comune.

    prices = prices.sort_index().copy()

    prices = prices.loc[~prices.index.duplicated(keep="last")]

    prices = prices.dropna(axis=1, how="all")



    if len(prices) <= lookback_days:

        raise ValueError(

            "Non ci sono abbastanza dati storici "

            "per il lookback selezionato."

        )



    n_assets = prices.shape[1]



    effective_assets = (

        min(int(selection_top_n), n_assets)

        if selection_top_n is not None

        else n_assets

    )



    if effective_assets < 2:

        raise ValueError("Servono almeno due asset investibili.")



    if max_weight < 1 / effective_assets:

        raise ValueError(

            "Il peso massimo per asset è troppo basso "

            "rispetto al numero di asset investibili."

        )



    portfolio_values = []

    weights_history = []



    current_portfolio_value = float(

        initial_capital

    )



    # Primo giorno in cui possiamo costruire il portafoglio.

    start_position = lookback_days



    # =====================================================

    # POINT-IN-TIME ASSET SELECTION POLICY

    # =====================================================

    # "fixed": lo screening viene eseguito una sola volta a t0.
    # I titoli selezionati restano invariati per tutto il backtest;
    # ai rebalance successivi vengono ricalcolati soltanto i pesi.
    #
    # "dynamic": lo screening viene rieseguito a ogni rebalance.
    # Possono quindi cambiare sia i componenti sia i pesi.

    fixed_selected_tickers = None

    if selection_top_n is not None and selection_policy == "fixed":

        initial_historical_universe = prices.iloc[
            start_position - lookback_days:
            start_position
        ]

        _, fixed_selected_tickers = screen_prices_point_in_time(
            initial_historical_universe,
            top_n=int(selection_top_n),
            min_observations=min(lookback_days, 253),
        )

        initial_rebalance_row = prices.iloc[start_position]

        fixed_selected_tickers = [
            ticker for ticker in fixed_selected_tickers
            if ticker in initial_rebalance_row.index
            and pd.notna(initial_rebalance_row[ticker])
            and initial_rebalance_row[ticker] > 0
        ]

        if len(fixed_selected_tickers) < 2:
            raise ValueError(
                "Point-in-time screening produced fewer than two "
                "tradable assets at the backtest start."
            )



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



        historical_universe = prices.iloc[

            rebalance_start - lookback_days:

            rebalance_start

        ]



        # -------------------------------------------------

        # 1A. POINT-IN-TIME ASSET SELECTION

        # -------------------------------------------------

        if selection_top_n is not None:

            if selection_policy == "fixed":

                selected_tickers = fixed_selected_tickers.copy()

            else:

                _, selected_tickers = screen_prices_point_in_time(

                    historical_universe,

                    top_n=int(selection_top_n),

                    min_observations=min(

                        lookback_days,

                        253,

                    ),

                )



            # I titoli devono avere un prezzo disponibile alla data

            # di rebalance. In modalità fixed questo controllo non

            # modifica la selezione economica: protegge soltanto da

            # dati mancanti/non negoziabili.

            rebalance_row = prices.iloc[rebalance_start]

            tradable_tickers = [

                ticker for ticker in selected_tickers

                if ticker in rebalance_row.index

                and pd.notna(rebalance_row[ticker])

                and rebalance_row[ticker] > 0

            ]



            if len(tradable_tickers) < 2:

                raise ValueError(

                    f"Fewer than two selected assets are tradable "

                    f"on {rebalance_date}."

                )



            selected_tickers = tradable_tickers

            historical_prices = historical_universe[

                selected_tickers

            ].dropna()

        else:

            selected_tickers = list(prices.columns)

            historical_prices = historical_universe.dropna()



        if len(historical_prices) < 2:

            raise ValueError(

                f"Insufficient common history on {rebalance_date}."

            )



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



            selected_count = len(selected_tickers)

            target_weights = pd.Series(

                np.ones(selected_count) / selected_count,

                index=selected_tickers,

            )



        # Sicurezza numerica.

        # Espandiamo i pesi sull'universo completo: gli asset non

        # selezionati alla data t ricevono peso zero.

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



        rebalance_prices = prices.iloc[rebalance_start]



        allocated_capital = (

            current_portfolio_value

            * target_weights

        )



        # Nel backtest utilizziamo quote frazionarie

        # per misurare la strategia teorica senza

        # distorsioni dovute all'arrotondamento.

        quantities = pd.Series(0.0, index=prices.columns)

        active = target_weights > 0

        quantities.loc[active] = (

            allocated_capital.loc[active]

            / rebalance_prices.loc[active]

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

        ].copy()



        # Valutiamo solo le posizioni effettivamente detenute. Per una

        # quotazione sporadicamente mancante usiamo l'ultimo prezzo noto,

        # senza usare dati futuri (solo forward-fill).

        active_tickers = target_weights[target_weights > 0].index.tolist()

        active_holding_prices = holding_prices[active_tickers].ffill()



        if active_holding_prices.isna().any().any():

            raise ValueError(

                f"Missing holding-period prices after rebalance on "

                f"{rebalance_date}."

            )



        period_values = (

            active_holding_prices

            * quantities.reindex(active_tickers)

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