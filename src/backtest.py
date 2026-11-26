from hashlib import sha256

import numpy as np
import pandas as pd
from src.eur_risk_free import ESTR_ONLY, validate_mode
from src.factors import capm_expected_returns
from src.fama_french import estimate_ff3_eur_returns
from src.fama_french_5 import estimate_ff5_eur_returns
from src.factor_currency import estimate_factor_returns, factors_for_backtest_signal
from src.transaction_costs import fund_orders
from src.risk import portfolio_return, portfolio_volatility
from src.portfolio_context import check_minimum, portfolio_settings
from src.screening import screen_prices_point_in_time, screen_tradable_point_in_time, screen_quant_point_in_time
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
    historical_fundamentals=None,
    selection_audit=None,
    active_factors=("Momentum", "Risk"),
    market="Borsa Italiana",
    capture_initial_ex_ante=False,
    input_estimator=None,
    eur_rf_mode=ESTR_ONLY,
    selection_prices=None,
    manual_assets=None,
    strict_estimation=False,
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
    validate_mode(eur_rf_mode)
    valid_strategies = [
        "ff3_factor_cov_max_sharpe",
        "ff5_factor_cov_max_sharpe",
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
    from src.hybrid_portfolio import eligible_manual, manual_records, native_ranking_prices
    configured = portfolio_settings()
    if configured is not None:
        if selection_prices is None:
            selection_prices = native_ranking_prices(prices)
        if manual_assets is None:
            manual_assets = manual_records(configured)
    ranking_prices = prices if selection_prices is None else selection_prices.sort_index()
    manual_assets = {} if manual_assets is None else manual_assets
    if len(prices) <= lookback_days:
        raise ValueError(
            "Non ci sono abbastanza dati storici "
            "per il lookback selezionato."
        )
    n_assets = prices.shape[1]
    effective_assets = (
        min(int(selection_top_n) + len(manual_assets), n_assets)
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
    factor_risk_history = {}
    estimator_failure = None
    portfolio_values = []
    weights_history = []
    previous_quantities = pd.Series(0.0, index=prices.columns)
    position_marks = []
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
    # Reuse cached estimates only for the same inputs and market.
    def digest(frame):
        if frame is None:
            return None
        labels = tuple(frame.columns) if isinstance(frame, pd.DataFrame) else frame.name
        return labels, sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).digest()

    fingerprint = (digest(prices), digest(ranking_prices) if selection_prices is not None else None)
    factor_fingerprints = (market, digest(ff3_factors), digest(ff5_factors), digest(eurusd_prices))
    if (shared_window_cache.get("_dataset") != fingerprint
            or shared_window_cache.get("_fundamentals") is not historical_fundamentals):
        shared_window_cache.clear()
        shared_window_cache["_dataset"] = fingerprint
        shared_window_cache["_fundamentals"] = historical_fundamentals
    def ranking_inputs(position):
        decision = prices.index[position-1]
        history = ranking_prices.loc[:decision].tail(lookback_days)
        return history, history.iloc[-1], decision

    if selection_top_n is not None and selection_policy == "fixed":
        fixed_key = ("fixed_selection_tplus1_v4", tuple(prices.columns), prices.index[start_position], lookback_days, int(selection_top_n), tuple(active_factors))
        if fixed_key not in shared_window_cache:
            ranking, selected = screen_quant_point_in_time(*ranking_inputs(start_position), historical_fundamentals, top_n=int(selection_top_n), min_observations=max(252, min(lookback_days, 253)), active_factors=active_factors)
            shared_window_cache[fixed_key] = (list(selected), ranking.copy())
        cached_selection = shared_window_cache[fixed_key]
        fixed_selected_tickers = list(cached_selection[0])
        if selection_audit is not None:
            selection_audit[str(prices.index[start_position].date())] = cached_selection[1].copy()
        if len(fixed_selected_tickers) < 2 and not manual_assets:
            raise ValueError("Point-in-time screening produced fewer than two assets.")
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
        # 1. Decisione alla chiusura di t-1, esecuzione alla chiusura di t.
        #    Il prezzo di t non entra nello screening né nella stima.
        #    DATI DISPONIBILI PRIMA DEL RIBILANCIAMENTO
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
                selection_key = ("dynamic_selection_tplus1_v4", tuple(prices.columns), prices.index[rebalance_start], lookback_days, int(selection_top_n), tuple(active_factors))
                if selection_key not in shared_window_cache:
                    ranking, selected = screen_quant_point_in_time(*ranking_inputs(rebalance_start), historical_fundamentals, top_n=int(selection_top_n), min_observations=max(252, min(lookback_days, 253)), active_factors=active_factors)
                    shared_window_cache[selection_key] = (list(selected), ranking.copy())
                selected_tickers = list(shared_window_cache[selection_key][0])
                if selection_audit is not None:
                    selection_audit[str(prices.index[rebalance_start].date())] = shared_window_cache[selection_key][1].copy()
            if len(selected_tickers) < 2 and not manual_assets:
                raise ValueError(f"Fewer than two selected assets on {rebalance_date}.")
            historical_prices = historical_universe[
                selected_tickers
            ].dropna()
        else:
            selected_tickers = list(prices.columns)
            historical_prices = historical_universe.dropna()
        if manual_assets:
            decision = historical_universe.index[-1]
            selected_tickers, manual_status = eligible_manual(historical_universe, selected_tickers, manual_assets,
                                                               ranked=selection_top_n is not None)
            if len(selected_tickers) < 2:
                raise ValueError(f"Fewer than two eligible assets on {decision:%Y-%m-%d}.")
            if selection_audit is not None and str(rebalance_date.date()) in selection_audit:
                selection_audit[str(rebalance_date.date())].attrs["manual_status"] = manual_status
        factor_prices = historical_universe[selected_tickers]
        if manual_assets:
            historical_prices = factor_prices.dropna()
        if input_estimator is None:
            if len(historical_prices) < 2:
                raise ValueError(
                    f"Insufficient common history on {rebalance_date}."
                )
            stats_key = ("window_stats", lookback_days, rebalance_start, tuple(historical_prices.columns), len(historical_prices))
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
            initial_mu = expected_returns
            risk_cov = covariance
        factor_components = None
        factor_risk_error = None

        def load_factor_components(model, use_factor_cov):
            from src.factor_covariance import estimate_factor_covariance

            key = ("factor_risk", eur_rf_mode, model, factor_fingerprints, market, rebalance_start,
                   factor_prices.index[0], factor_prices.index[-1], tuple(factor_prices.columns))
            if not capture_initial_ex_ante:
                return (estimate_factor_covariance(factor_prices, signal_factors, model=model,
                        eurusd_prices=eurusd_prices, market=market, eur_rf_mode=eur_rf_mode) if use_factor_cov else None), None
            if key not in shared_window_cache:
                shared_window_cache[key] = estimate_factor_covariance(
                    factor_prices, signal_factors, model=model, eurusd_prices=eurusd_prices,
                    market=market, return_components=True, eur_rf_mode=eur_rf_mode,
                )
            return shared_window_cache[key]

        if input_estimator is not None:
            try:
                target_weights, initial_mu, risk_cov = input_estimator(factor_prices)
            except (ValueError, ImportError, RuntimeError, np.linalg.LinAlgError) as exc:
                estimator_failure = f"Rebalance {rebalance_date}: {exc}"
                if strict_estimation or not portfolio_values:
                    raise ValueError(estimator_failure) from exc
                break
        elif strategy == "max_sharpe":
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
            initial_mu = capm_returns
            target_weights = (
                optimize_maximum_sharpe(
                    capm_returns,
                    covariance,
                    risk_free_rate=risk_free_rate,
                    max_weight=max_weight,
                )
            )
        elif strategy in ("ff3_max_sharpe", "ff3_factor_cov_max_sharpe"):
            if ff3_factors is None or (market != "USA" and eurusd_prices is None):
                raise ValueError("FF3 factors and EUR/USD prices are required.")
            signal_factors = factors_for_backtest_signal(ff3_factors, factor_prices.index[-1])
            ff3_key = ("ff3_expected_returns", eur_rf_mode, factor_fingerprints, lookback_days, rebalance_start,
                       tuple(historical_prices.columns), len(historical_prices))
            if ff3_key not in shared_window_cache:
                shared_window_cache[ff3_key] = estimate_factor_returns(
                    factor_prices, signal_factors, eurusd_prices,
                    market=market, model="ff3", return_stats=True, eur_rf_mode=eur_rf_mode
                )
            ff3_expected, ff3_stats = shared_window_cache[ff3_key]
            ff3_mu = ff3_expected.reindex(covariance.index)
            if ff3_mu.isna().any():
                raise ValueError(f"Missing FF3 expected returns on {rebalance_date}.")
            initial_mu = ff3_mu
            use_factor_cov = strategy == "ff3_factor_cov_max_sharpe"
            try:
                factor_cov, factor_components = load_factor_components("ff3", use_factor_cov)
                if use_factor_cov:
                    risk_cov = factor_cov
            except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
                if use_factor_cov:
                    raise
                factor_risk_error = str(exc)
            target_weights = optimize_maximum_sharpe(
                ff3_mu, risk_cov, risk_free_rate=risk_free_rate,
                max_weight=max_weight
            )
        elif strategy in ("ff5_max_sharpe", "ff5_factor_cov_max_sharpe"):
            if ff5_factors is None or (market != "USA" and eurusd_prices is None):
                raise ValueError("FF5 factors and EUR/USD prices are required.")
            signal_factors = factors_for_backtest_signal(ff5_factors, factor_prices.index[-1])
            ff5_key = ("ff5_expected_returns", eur_rf_mode, factor_fingerprints, lookback_days, rebalance_start,
                       tuple(historical_prices.columns), len(historical_prices))
            if ff5_key not in shared_window_cache:
                shared_window_cache[ff5_key] = estimate_factor_returns(
                    factor_prices, signal_factors, eurusd_prices,
                    market=market, model="ff5", return_stats=True, eur_rf_mode=eur_rf_mode
                )
            ff5_expected, ff5_stats = shared_window_cache[ff5_key]
            ff5_mu = ff5_expected.reindex(covariance.index)
            if ff5_mu.isna().any():
                raise ValueError(f"Missing FF5 expected returns on {rebalance_date}.")
            initial_mu = ff5_mu
            use_factor_cov = strategy == "ff5_factor_cov_max_sharpe"
            try:
                factor_cov, factor_components = load_factor_components("ff5", use_factor_cov)
                if use_factor_cov:
                    risk_cov = factor_cov
            except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
                if use_factor_cov:
                    raise
                factor_risk_error = str(exc)
            target_weights = optimize_maximum_sharpe(
                ff5_mu, risk_cov, risk_free_rate=risk_free_rate,
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
        if configured:
            check_minimum(target_weights.reindex(selected_tickers))
        if capture_initial_ex_ante and rebalance_start == start_position:
            labels = risk_cov.index
            weights = target_weights.reindex(labels).to_numpy()
            expected = portfolio_return(weights, initial_mu.reindex(labels))
            volatility = portfolio_volatility(weights, risk_cov.loc[labels, labels])
            initial_ex_ante_kpis = {
                "estimation_end": historical_universe.index[-1].date().isoformat(),
                "expected_return": expected,
                "volatility": volatility,
                "sharpe": (expected - risk_free_rate) / volatility if volatility > 0 else 0.0,
            }

        if factor_components is not None:
            from src.factor_covariance import factor_risk_record

            try:
                factor_risk_history[rebalance_date.date().isoformat()] = factor_risk_record(
                    target_weights, factor_components, risk_cov, factor_prices.index[-1], use_factor_cov,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                factor_risk_error = str(exc)
        if factor_risk_error is not None:
            factor_risk_history[rebalance_date.date().isoformat()] = {"error": factor_risk_error}

        # Exposure FF3 ponderate sui pesi target effettivamente investiti.
        if strategy in ("ff3_max_sharpe", "ff3_factor_cov_max_sharpe") and ff3_exposure_records is not None:
            aligned = ff3_stats.reindex(target_weights.index)
            exposure = {"Rebalance Date": pd.Timestamp(rebalance_date)}
            for label, beta in (("Market Beta", "Beta MKT"),
                                ("SMB Exposure", "Beta SMB"),
                                ("HML Exposure", "Beta HML")):
                values = aligned[beta].fillna(0.0)
                exposure[label] = float((target_weights * values).sum())
            exposure["Active Assets"] = int((target_weights > 1e-10).sum())
            ff3_exposure_records.append(exposure)
        if strategy in ("ff5_max_sharpe", "ff5_factor_cov_max_sharpe") and ff5_exposure_records is not None:
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
        executable = target_weights.gt(0) | previous_quantities.ne(0)
        if not np.isfinite(rebalance_prices[executable]).all() or rebalance_prices[executable].le(0).any():
            raise ValueError(f"Missing or invalid execution price on {rebalance_date}")
        if transaction_costs is not None:
            pretrade_value = current_portfolio_value
            current_portfolio_value, execution_cost, detail = fund_orders(
                pretrade_value, target_weights, previous_quantities,
                rebalance_prices, transaction_costs,
            )
            cost_rows.append({"Rebalance Date": rebalance_date, **detail,
                              "Total Cost": execution_cost,
                              "Executed Turnover": detail["Traded Notional"] / pretrade_value})
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
        if capture_initial_ex_ante:
            position_marks.append(active_holding_prices.mul(quantities.reindex(active_tickers)))
        # Evitiamo di duplicare la data iniziale
        # del periodo precedente.
        if portfolio_values:
            portfolio_values[-1].iloc[-1] = period_values.iloc[0]
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
    if factor_risk_history:
        backtest.attrs["factor_risk_history"] = factor_risk_history
    if estimator_failure is not None:
        backtest.attrs["estimator_failure"] = estimator_failure
    backtest.attrs["initial_capital"] = float(initial_capital)
    if capture_initial_ex_ante:
        backtest.attrs["initial_ex_ante_kpis"] = initial_ex_ante_kpis
        marks = pd.concat(position_marks).fillna(0.)
        backtest.attrs["risk_position_marks"] = marks.loc[~marks.index.duplicated(keep="last")]
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
        .copy()
    )
    # Il rendimento della prima seduta deve includere i costi iniziali.
    if "initial_capital" in backtest.attrs and len(daily_returns):
        initial_capital = float(backtest.attrs["initial_capital"])
        daily_returns.iloc[0] = float(backtest["Portfolio Value"].iloc[0]) / initial_capital - 1

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
    # La prima osservazione può essere già al netto dei costi di esecuzione.
    # Per i rendimenti del backtest la base è il capitale iniziale versato.
    initial_value = float(backtest.attrs.get("initial_capital", portfolio_value.iloc[0]))
    if not np.isfinite(initial_value) or initial_value <= 0:
        raise ValueError("Initial capital must be finite and positive")
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
    daily_target = risk_free_rate / 252
    downside = np.minimum(daily_returns - daily_target, 0.0)
    downside_deviation = float(np.sqrt(252 * np.mean(downside ** 2)))
    sortino_ratio = ((annualized_return - risk_free_rate) / downside_deviation
                     if downside_deviation > 0 else 0.0)
    # -------------------------------------------------
    # MAX DRAWDOWN
    # -------------------------------------------------
    running_max = (
        portfolio_value.cummax().clip(lower=initial_value)
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
