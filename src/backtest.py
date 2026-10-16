import numpy as np
import pandas as pd
from src.factors import capm_expected_returns
from src.fama_french import estimate_ff3_eur_returns
from src.fama_french_5 import estimate_ff5_eur_returns
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
    shared_window_cache: dict | None = None,
    ff3_factors=None,
    eurusd_prices=None,
    ff3_exposure_records=None,
    ff5_factors=None,
    ff5_exposure_records=None,
    transaction_costs=None,
    transaction_log=None,
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
        "ff3_max_sharpe",
        "ff5_max_sharpe",
        "min_volatility",
        "risk_parity",
        "hrp",
        "equal_weight",
    ]
    if strategy not in valid_strategies:
        raise ValueError(
            "la stategia deve essere 'max_sharpe', 'capm_max_sharpe', "
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
    previous_quantities = pd.Series(0.0, index=prices.columns)
    gross_capital = float(initial_capital)
    gross_portfolio_values = []
    gross_prev_quantities = pd.Series(0.0, index=prices.columns)
    cost_rows = []
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
    if shared_window_cache is None:
        shared_window_cache = {}
    if selection_top_n is not None and selection_policy == "fixed":
        initial_historical_universe = prices.iloc[
            start_position - lookback_days:
            start_position
        ]
        fixed_key = ("fixed_selection", start_position, lookback_days, int(selection_top_n))
        if fixed_key not in shared_window_cache:
            _, selected = screen_prices_point_in_time(initial_historical_universe, top_n=int(selection_top_n), min_observations=min(lookback_days, 253))
            shared_window_cache[fixed_key] = list(selected)
        fixed_selected_tickers = shared_window_cache[fixed_key].copy()
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
                selection_key = ("dynamic_selection", rebalance_start, lookback_days, int(selection_top_n))
                if selection_key not in shared_window_cache:
                    _, selected = screen_prices_point_in_time(historical_universe, top_n=int(selection_top_n), min_observations=min(lookback_days, 253))
                    shared_window_cache[selection_key] = list(selected)
                selected_tickers = shared_window_cache[selection_key].copy()
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
        stats_key = ("window_stats", rebalance_start, tuple(historical_prices.columns), len(historical_prices))
        if stats_key not in shared_window_cache:
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
            shared_window_cache[stats_key] = (historical_returns, expected_returns, covariance)
        historical_returns, expected_returns, covariance = shared_window_cache[stats_key]
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
        elif strategy == "ff3_max_sharpe":
            if ff3_factors is None or eurusd_prices is None:
                raise ValueError("FF3 factors and EUR/USD prices are required.")
            ff3_key = ("ff3_expected_returns", rebalance_start,
                       tuple(historical_prices.columns), len(historical_prices))
            if ff3_key not in shared_window_cache:
                shared_window_cache[ff3_key] = estimate_ff3_eur_returns(
                    historical_prices, ff3_factors, eurusd_prices, return_stats=True
                )
            ff3_expected, ff3_stats = shared_window_cache[ff3_key]
            ff3_mu = ff3_expected.reindex(covariance.index)
            if ff3_mu.isna().any():
                raise ValueError(f"Missing FF3 expected returns on {rebalance_date}.")
            target_weights = optimize_maximum_sharpe(
                ff3_mu, covariance, risk_free_rate=risk_free_rate,
                max_weight=max_weight
            )
        elif strategy == "ff5_max_sharpe":
            if ff5_factors is None or eurusd_prices is None:
                raise ValueError("FF5 factors and EUR/USD prices are required.")
            ff5_key = ("ff5_expected_returns", rebalance_start,
                       tuple(historical_prices.columns), len(historical_prices))
            if ff5_key not in shared_window_cache:
                shared_window_cache[ff5_key] = estimate_ff5_eur_returns(
                    historical_prices, ff5_factors, eurusd_prices, return_stats=True
                )
            ff5_expected, ff5_stats = shared_window_cache[ff5_key]
            ff5_mu = ff5_expected.reindex(covariance.index)
            if ff5_mu.isna().any():
                raise ValueError(f"Missing FF5 expected returns on {rebalance_date}.")
            target_weights = optimize_maximum_sharpe(
                ff5_mu, covariance, risk_free_rate=risk_free_rate,
                max_weight=max_weight
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
        # Exposure FF3 ponderate sui pesi target effettivamente investiti.
        if strategy == "ff3_max_sharpe" and ff3_exposure_records is not None:
            aligned = ff3_stats.reindex(target_weights.index)
            exposure = {"Rebalance Date": pd.Timestamp(rebalance_date)}
            for label, beta in (("Market Beta", "Beta MKT"),
                                ("SMB Exposure", "Beta SMB"),
                                ("HML Exposure", "Beta HML")):
                values = aligned[beta].fillna(0.0)
                exposure[label] = float((target_weights * values).sum())
            exposure["Active Assets"] = int((target_weights > 1e-10).sum())
            ff3_exposure_records.append(exposure)
        if strategy == "ff5_max_sharpe" and ff5_exposure_records is not None:
            aligned = ff5_stats.reindex(target_weights.index)
            exposure = {"Rebalance Date": pd.Timestamp(rebalance_date)}
            for label, beta in (("Market Beta", "Beta MKT"),
                                ("SMB Exposure", "Beta SMB"),
                                ("HML Exposure", "Beta HML"),
                                ("RMW Exposure", "Beta RMW"),
                                ("CMA Exposure", "Beta CMA")):
                exposure[label] = float((target_weights * aligned[beta].fillna(0)).sum())
            exposure["Active Assets"] = int((target_weights > 1e-10).sum())
            ff5_exposure_records.append(exposure)
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
        # Le operazioni di trading sono finanziate dal valore effettivo del portafoglio.
        # I costi vengono calcolati in base alle variazioni delle quantità di titoli
        # da acquistare o vendere, non alla variazione dei pesi target.
        execution_cost = 0.0
        if transaction_costs is not None:
            from src.transaction_costs import estimate_order_costs
            executable = target_weights.reindex(prices.columns).fillna(0.0)
            # Risolviamo iterativamente il problema di punto fisso: i costi di transazione
            # riducono il capitale disponibile per acquistare la nuova allocazione target.
            available = float(current_portfolio_value)
            for _ in range(30):
                desired = (available * executable).div(rebalance_prices).fillna(0.0)
                changes = desired - previous_quantities
                execution_cost, detail = estimate_order_costs(
                    changes, rebalance_prices, transaction_costs
                )
                next_available = float(current_portfolio_value) - execution_cost
                if next_available <= 0:
                    raise ValueError("Trading costs exhaust portfolio capital")
                if abs(next_available - available) < 1e-8:
                    available = next_available
                    break
                available = next_available
            current_portfolio_value = available
            cost_rows.append({"Rebalance Date": rebalance_date, **detail,
                              "Total Cost": execution_cost})
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
        previous_quantities = quantities.copy()
        if transaction_costs is not None:
            gross_prev_quantities = (
                gross_capital * target_weights.reindex(prices.columns).fillna(0.0)
            ).div(rebalance_prices).fillna(0.0)
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
        if transaction_costs is not None:
            gross_period = (active_holding_prices * gross_prev_quantities.reindex(active_tickers)).sum(axis=1)
            if gross_portfolio_values:
                gross_period = gross_period.iloc[1:]
            gross_portfolio_values.append(gross_period)
            gross_capital = float(gross_period.iloc[-1])
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
    if transaction_costs is not None:
        gross_series = pd.concat(gross_portfolio_values).reindex(backtest.index)
        backtest["Gross Portfolio Value"] = gross_series
        backtest["Transaction Cost"] = 0.0
        for cost_row in cost_rows:
            date = cost_row["Rebalance Date"]
            if date in backtest.index:
                backtest.loc[date, "Transaction Cost"] += cost_row["Total Cost"]
        backtest.attrs["transaction_log"] = pd.DataFrame(cost_rows)
        # La prima esecuzione dell'ordine rappresenta un costo rispetto al capitale
        # inizialmente investito, anche se il primo valore osservato del portafoglio
        # è già al netto dei costi di transazione.
        backtest.loc[backtest.index[0], "Daily Return"] = (
            backtest["Portfolio Value"].iloc[0] / initial_capital - 1
        )
        if transaction_log is not None:
            transaction_log.extend(cost_rows)
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
