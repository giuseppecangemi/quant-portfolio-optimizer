import time

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# CONFIGURAZIONE
# =========================================================

MIN_PRICE_OBSERVATIONS = 80
PRICE_BATCH_SIZE = 10
PRICE_BATCH_RETRIES = 2
PRICE_RETRY_SLEEP_SECONDS = 2.0

# I fondamentali Yahoo possono essere incompleti/intermittenti.
# Non determinano mai, da soli, la validità del ticker.
FUNDAMENTAL_PAUSE_SECONDS = 0.03


# =========================================================
# UTILITIES
# =========================================================

def _clean_universe(universe: pd.DataFrame) -> pd.DataFrame:
    required = {"name", "ticker"}

    if not required.issubset(universe.columns):
        raise ValueError(
            "Universe must contain at least 'name' and 'ticker' columns."
        )

    cleaned = universe.copy()

    cleaned["name"] = cleaned["name"].astype(str).str.strip()
    cleaned["ticker"] = cleaned["ticker"].astype(str).str.strip()

    cleaned = cleaned[
        cleaned["ticker"].ne("")
        & cleaned["ticker"].str.lower().ne("nan")
    ].copy()

    # Evita di scaricare due volte lo stesso ticker.
    cleaned = cleaned.drop_duplicates(subset=["ticker"], keep="first")

    return cleaned.reset_index(drop=True)


def _chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _extract_close_frame(
    downloaded: pd.DataFrame,
    requested_tickers: list[str],
) -> pd.DataFrame:
    """
    Normalizza l'output di yf.download() in un DataFrame:
        index = date
        columns = ticker
        values = adjusted Close

    Gestisce sia download multi-ticker sia single-ticker.
    """
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()

    # MultiIndex: formato tipico di yf.download con più ticker.
    if isinstance(downloaded.columns, pd.MultiIndex):
        level0 = downloaded.columns.get_level_values(0)
        level1 = downloaded.columns.get_level_values(1)

        if "Close" in level0:
            close = downloaded["Close"].copy()
        elif "Close" in level1:
            close = downloaded.xs("Close", axis=1, level=1).copy()
        else:
            return pd.DataFrame()

        if isinstance(close, pd.Series):
            close = close.to_frame()

        # Normalizza i nomi delle colonne.
        close.columns = [str(col) for col in close.columns]
        return close.apply(pd.to_numeric, errors="coerce")

    # Single ticker.
    if "Close" not in downloaded.columns:
        return pd.DataFrame()

    close = downloaded["Close"]

    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            return pd.DataFrame()

    ticker = requested_tickers[0]
    return pd.DataFrame(
        {ticker: pd.to_numeric(close, errors="coerce")}
    )


# =========================================================
# PRICE DATA LAYER
# =========================================================

def _download_price_batch(
    tickers: list[str],
    period: str,
) -> tuple[pd.DataFrame, str]:
    """
    Scarica un blocco di ticker in UNA chiamata Yahoo.
    Fa retry solo a livello di batch.
    """
    last_error = ""

    for attempt in range(PRICE_BATCH_RETRIES + 1):
        try:
            data = yf.download(
                tickers=tickers,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=20,
                group_by="column",
            )

            close = _extract_close_frame(data, tickers)

            if not close.empty:
                return close, "OK"

            last_error = "Empty batch response"

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < PRICE_BATCH_RETRIES:
            time.sleep(PRICE_RETRY_SLEEP_SECONDS * (attempt + 1))

    return pd.DataFrame(), last_error or "Batch download failed"


def download_universe_prices(
    universe: pd.DataFrame,
    period: str = "2y",
    batch_size: int = PRICE_BATCH_SIZE,
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """
    Scarica l'universo a blocchi, non con 163 chiamate singole.

    Importante:
    - un ticker viene considerato valido in base ai prezzi effettivamente ricevuti;
    - un batch fallito NON viene chiamato "invalid ticker";
    - per i ticker mancanti nel batch viene effettuato un solo retry individuale,
      utile a distinguere una risposta batch incompleta da un ticker realmente
      non disponibile.
    """
    cleaned = _clean_universe(universe)
    tickers = cleaned["ticker"].tolist()

    prices: dict[str, pd.Series] = {}
    status: dict[str, str] = {
        ticker: "Not downloaded"
        for ticker in tickers
    }

    for batch in _chunks(tickers, max(1, int(batch_size))):
        close_frame, batch_status = _download_price_batch(batch, period)

        if close_frame.empty:
            for ticker in batch:
                status[ticker] = f"Batch download failed: {batch_status}"
            continue

        for ticker in batch:
            if ticker not in close_frame.columns:
                status[ticker] = "Missing from batch response"
                continue

            series = close_frame[ticker].dropna()

            if len(series) < MIN_PRICE_OBSERVATIONS:
                status[ticker] = (
                    f"Insufficient price history ({len(series)} observations)"
                )
                continue

            prices[ticker] = series.astype(float)
            status[ticker] = "OK"

    # Retry mirato SOLO dei ticker non recuperati dal batch.
    missing = [
        ticker
        for ticker in tickers
        if ticker not in prices
    ]

    for ticker in missing:
        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            close_frame = _extract_close_frame(data, [ticker])

            if close_frame.empty or ticker not in close_frame.columns:
                # Non dichiariamo "invalid ticker": non possiamo dedurlo
                # da una risposta vuota di Yahoo.
                if status[ticker].startswith("Batch download failed"):
                    status[ticker] += " | Individual retry returned no data"
                else:
                    status[ticker] = "No price data after individual retry"
                continue

            series = close_frame[ticker].dropna()

            if len(series) < MIN_PRICE_OBSERVATIONS:
                status[ticker] = (
                    f"Insufficient price history ({len(series)} observations)"
                )
                continue

            prices[ticker] = series.astype(float)
            status[ticker] = "OK"

        except Exception as exc:
            previous = status.get(ticker, "")
            detail = f"Individual retry failed: {type(exc).__name__}"

            status[ticker] = (
                f"{previous} | {detail}"
                if previous
                else detail
            )

    return prices, status


# =========================================================
# FUNDAMENTAL DATA
# =========================================================

def _safe_float(value) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _safe_fundamentals(ticker: str) -> tuple[dict, str]:
    """
    Recupera Value/Quality senza confondere un errore fondamentale
    con un ticker non valido.

    Se Yahoo non restituisce i fondamentali, Momentum e Risk restano
    comunque utilizzabili.
    """
    result = {
        "earnings_yield": np.nan,
        "book_to_market": np.nan,
        "roe": np.nan,
        "debt_to_equity": np.nan,
    }

    try:
        info = yf.Ticker(ticker).info or {}

        trailing_pe = _safe_float(info.get("trailingPE"))
        price_to_book = _safe_float(info.get("priceToBook"))
        roe = _safe_float(info.get("returnOnEquity"))
        debt_to_equity = _safe_float(info.get("debtToEquity"))

        if pd.notna(trailing_pe) and trailing_pe > 0:
            result["earnings_yield"] = 1.0 / trailing_pe

        if pd.notna(price_to_book) and price_to_book > 0:
            result["book_to_market"] = 1.0 / price_to_book

        if pd.notna(roe):
            result["roe"] = roe

        if pd.notna(debt_to_equity):
            # Yahoo normalmente restituisce D/E in percentuale.
            result["debt_to_equity"] = debt_to_equity / 100.0

        available = sum(pd.notna(v) for v in result.values())

        if available == 4:
            return result, "OK"

        if available == 0:
            return result, "No fundamental fields available"

        return result, f"Partial fundamentals ({available}/4)"

    except Exception as exc:
        return result, f"Fundamental request failed: {type(exc).__name__}"


# =========================================================
# SIGNALS
# =========================================================

def _percentile_score(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Percentile cross-sectional 0-100.
    I NaN restano NaN.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    score = numeric.rank(
        pct=True,
        method="average",
        ascending=True,
    ) * 100.0

    if not higher_is_better:
        score = 100.0 - score

    return score.astype(float)


def _calculate_price_signals(close: pd.Series) -> dict:
    """
    Momentum + Risk calcolati esclusivamente dallo storico prezzi.
    """
    close = close.dropna().astype(float)
    returns = close.pct_change().dropna()

    if len(close) >= 252:
        # 252 observations represent the conventional 1Y trading window.
        # Use the oldest available price rather than switching the ENTIRE
        # strategy to 6-1 when a 252-row training slice is supplied.
        # With 253+ observations, preserve the original 12-1 definition.
        anchor = -253 if len(close) >= 253 else -252
        momentum = close.iloc[-22] / close.iloc[anchor] - 1.0
        momentum_window = "12-1" if len(close) >= 253 else "12-1 (252 obs)"

    elif len(close) >= 127:
        momentum = close.iloc[-22] / close.iloc[-127] - 1.0
        momentum_window = "6-1"

    else:
        momentum = close.iloc[-1] / close.iloc[0] - 1.0
        momentum_window = "available"

    volatility = float(returns.std() * np.sqrt(252))

    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "momentum": float(momentum),
        "momentum_window": momentum_window,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "observations": int(len(close)),
    }


# =========================================================
# SCREENING
# =========================================================


def score_quant_candidates(valid: pd.DataFrame, top_n: int, min_components: int = 2, active_factors=("Momentum", "Risk")):
    """Single scoring/ranking engine used by live and point-in-time screening."""
    active_factors = tuple(active_factors)
    if not active_factors or len(set(active_factors)) != len(active_factors) or any(f not in ("Momentum", "Risk", "Value", "Quality") for f in active_factors):
        raise ValueError("Select one or more distinct valid Quant factors")
    valid = valid.copy()
    for col in ("Momentum Score", "Risk Score", "Value Score", "Quality Score", "Quant Score", "Rank"):
        valid[col] = np.nan
    valid["Selected"] = False
    valid["Components Available"] = 0
    # -----------------------------------------------------
    # 3. MOMENTUM SCORE
    # -----------------------------------------------------
    valid["Momentum Score"] = _percentile_score(
        valid["Momentum"],
        higher_is_better=True,
    )

    # -----------------------------------------------------
    # 4. RISK SCORE
    # -----------------------------------------------------
    volatility_score = _percentile_score(
        valid["Volatility"],
        higher_is_better=False,
    )

    drawdown_score = _percentile_score(
        valid["Max Drawdown"],
        higher_is_better=True,
    )

    valid["Risk Score"] = pd.concat(
        [volatility_score, drawdown_score],
        axis=1,
    ).mean(axis=1, skipna=True)

    # -----------------------------------------------------
    # 5. VALUE SCORE
    # -----------------------------------------------------
    earnings_yield_score = _percentile_score(
        valid["Earnings Yield"],
        higher_is_better=True,
    )

    book_to_market_score = _percentile_score(
        valid["Book-to-Market"],
        higher_is_better=True,
    )

    valid["Value Score"] = pd.concat(
        [earnings_yield_score, book_to_market_score],
        axis=1,
    ).mean(axis=1, skipna=True)

    value_has_data = valid[
        ["Earnings Yield", "Book-to-Market"]
    ].notna().any(axis=1)

    valid.loc[~value_has_data, "Value Score"] = np.nan

    # -----------------------------------------------------
    # 6. QUALITY SCORE
    # -----------------------------------------------------
    roe_score = _percentile_score(
        valid["ROE"],
        higher_is_better=True,
    )

    leverage_score = _percentile_score(
        valid["Debt-to-Equity"],
        higher_is_better=False,
    )

    valid["Quality Score"] = pd.concat(
        [roe_score, leverage_score],
        axis=1,
    ).mean(axis=1, skipna=True)

    quality_has_data = valid[
        ["ROE", "Debt-to-Equity"]
    ].notna().any(axis=1)

    valid.loc[~quality_has_data, "Quality Score"] = np.nan

    # -----------------------------------------------------
    # 7. QUANT SCORE
    # -----------------------------------------------------
    component_columns = [
        "Momentum Score",
        "Risk Score",
        "Value Score",
        "Quality Score",
    ]

    component_columns = [f"{factor} Score" for factor in active_factors]
    valid["Components Available"] = (
        valid[component_columns]
        .notna()
        .sum(axis=1)
        .astype(int)
    )

    valid["Quant Score"] = valid[
        component_columns
    ].mean(axis=1, skipna=True)

    insufficient = (
        valid["Components Available"] < int(min_components)
    )

    valid.loc[insufficient, "Status"] = "Excluded"
    valid.loc[
        insufficient,
        "Reason",
    ] = "Insufficient scoring data"

    valid.loc[insufficient, "Quant Score"] = np.nan

    # -----------------------------------------------------
    # 8. RANKING
    # -----------------------------------------------------
    eligible = valid[
        valid["Status"].eq("Candidate")
        & valid["Quant Score"].notna()
    ].copy()

    eligible = eligible.sort_values(
        [
            "Quant Score",
            "Momentum Score",
            "Risk Score",
            "Ticker",
        ],
        ascending=[False, False, False, True],
        na_position="last",
    )

    eligible["Rank"] = np.arange(
        1,
        len(eligible) + 1,
        dtype=int,
    )

    selected_tickers = eligible.head(
        max(0, int(top_n))
    )["Ticker"].tolist()

    valid["Selected"] = valid["Ticker"].isin(
        selected_tickers
    )

    valid.loc[
        valid["Selected"],
        "Status",
    ] = "Selected"

    rank_map = eligible.set_index("Ticker")["Rank"]
    valid["Rank"] = valid["Ticker"].map(rank_map)

    return valid, eligible, selected_tickers

def screen_universe(
    universe: pd.DataFrame,
    period: str = "2y",
    top_n: int = 10,
    min_components: int = 2,
    active_factors=("Momentum", "Risk"),
) -> tuple[pd.DataFrame, list[str]]:
    """
    Screening quantitativo robusto.

    ELIGIBILITY
    -----------
    Un titolo è price-eligible se:
    - il ticker è presente;
    - Yahoo restituisce uno storico prezzi;
    - lo storico contiene almeno MIN_PRICE_OBSERVATIONS.

    La mancanza di fondamentali NON rende il ticker invalido.

    SCORE
    -----
    Momentum:
        momentum 12-1, con fallback 6-1.

    Risk:
        low volatility + drawdown meno severo.

    Value:
        Earnings Yield + Book-to-Market, quando disponibili.

    Quality:
        ROE + basso Debt-to-Equity, quando disponibili.

    Quant Score:
        media delle famiglie di segnali disponibili.

    Per default servono almeno 2 componenti, quindi un titolo con
    prezzi validi dispone già di Momentum + Risk e può essere
    classificato anche se Yahoo non restituisce i fondamentali.
    """
    active_factors = tuple(active_factors)
    if not active_factors or any(f not in ("Momentum", "Risk", "Value", "Quality") for f in active_factors):
        raise ValueError("Invalid Quant factors")
    cleaned = _clean_universe(universe)

    if cleaned.empty:
        return pd.DataFrame(), []

    # -----------------------------------------------------
    # 1. PREZZI: download batch dell'intero universo
    # -----------------------------------------------------
    prices, price_status = download_universe_prices(
        cleaned,
        period=period,
    )

    rows = []

    # -----------------------------------------------------
    # 2. Costruzione dataset base
    # -----------------------------------------------------
    for _, instrument in cleaned.iterrows():
        name = instrument["name"]
        ticker = instrument["ticker"]

        row = {
            "Instrument": name,
            "Ticker": ticker,
            "Status": "Excluded",
            "Reason": "",
            "Price Data Status": price_status.get(
                ticker,
                "Not downloaded",
            ),
            "Fundamental Data Status": "Not requested",
            "Observations": np.nan,
            "Momentum Window": "",
            "Momentum": np.nan,
            "Volatility": np.nan,
            "Max Drawdown": np.nan,
            "Earnings Yield": np.nan,
            "Book-to-Market": np.nan,
            "ROE": np.nan,
            "Debt-to-Equity": np.nan,
        }

        close = prices.get(ticker)

        if close is None:
            row["Reason"] = row["Price Data Status"]
            rows.append(row)
            continue

        try:
            signals = _calculate_price_signals(close)

        except Exception as exc:
            row["Reason"] = (
                f"Price signal calculation failed: {type(exc).__name__}"
            )
            rows.append(row)
            continue

        if "Value" in active_factors or "Quality" in active_factors:
            fundamentals, fundamental_status = _safe_fundamentals(ticker)
        else:
            fundamentals = {"earnings_yield": np.nan, "book_to_market": np.nan,
                            "roe": np.nan, "debt_to_equity": np.nan}
            fundamental_status = "Not requested (Momentum + Risk)"

        row.update({
            "Status": "Candidate",
            "Reason": "",
            "Fundamental Data Status": fundamental_status,
            "Observations": signals["observations"],
            "Momentum Window": signals["momentum_window"],
            "Momentum": signals["momentum"],
            "Volatility": signals["volatility"],
            "Max Drawdown": signals["max_drawdown"],
            "Earnings Yield": fundamentals["earnings_yield"],
            "Book-to-Market": fundamentals["book_to_market"],
            "ROE": fundamentals["roe"],
            "Debt-to-Equity": fundamentals["debt_to_equity"],
        })

        rows.append(row)

        if FUNDAMENTAL_PAUSE_SECONDS > 0:
            time.sleep(FUNDAMENTAL_PAUSE_SECONDS)

    result = pd.DataFrame(rows)

    if result.empty:
        return result, []

    # Tipi espliciti: evita l'errore pandas float64 <- False.
    result["Momentum Score"] = np.nan
    result["Risk Score"] = np.nan
    result["Value Score"] = np.nan
    result["Quality Score"] = np.nan
    result["Components Available"] = 0
    result["Quant Score"] = np.nan
    result["Selected"] = False
    result["Rank"] = np.nan

    valid_mask = result["Status"].eq("Candidate")
    valid = result.loc[valid_mask].copy()

    if valid.empty:
        return result, []

    valid, eligible, selected_tickers = score_quant_candidates(
        valid, top_n=top_n, min_components=min_components, active_factors=active_factors
    )

    # -----------------------------------------------------
    # 9. MERGE NEL RISULTATO COMPLETO
    # -----------------------------------------------------
    computed_columns = [
        "Status",
        "Reason",
        "Momentum Score",
        "Risk Score",
        "Value Score",
        "Quality Score",
        "Components Available",
        "Quant Score",
        "Selected",
        "Rank",
    ]

    valid_by_ticker = valid.set_index("Ticker")

    for idx, ticker in result["Ticker"].items():
        if ticker not in valid_by_ticker.index:
            continue

        for column in computed_columns:
            result.at[idx, column] = valid_by_ticker.at[
                ticker,
                column,
            ]

    result["Selected"] = (
        result["Selected"]
        .fillna(False)
        .astype(bool)
    )

    result["Components Available"] = (
        pd.to_numeric(
            result["Components Available"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    # Prima ranking valido, poi esclusi.
    result["_sort_rank"] = result["Rank"].fillna(
        10**9
    )

    result = (
        result.sort_values(
            ["_sort_rank", "Instrument", "Ticker"],
            ascending=[True, True, True],
        )
        .drop(columns="_sort_rank")
        .reset_index(drop=True)
    )

    return result, selected_tickers


# =========================================================
# POINT-IN-TIME SCREENING FOR BACKTESTS
# =========================================================

def build_price_matrix(
    universe: pd.DataFrame,
    period: str = "10y",
) -> pd.DataFrame:
    """Scarica l'universo e costruisce una matrice prezzi con join esterno.

    A differenza dello screener live, qui preserviamo i NaN: nel backtest ogni
    titolo viene valutato solo con le osservazioni che esistevano alla data
    storica simulata.
    """
    prices, _ = download_universe_prices(universe, period=period)

    if not prices:
        return pd.DataFrame()

    matrix = pd.concat(
        [series.rename(ticker) for ticker, series in prices.items()],
        axis=1,
        join="outer",
    )

    return matrix.sort_index()


def screen_prices_point_in_time(historical_prices, top_n=10, min_observations=252, active_factors=("Momentum", "Risk")):
    """Price-only PIT ranking using the same scoring engine as live selection."""
    if historical_prices is None or historical_prices.empty:
        return pd.DataFrame(), []
    required = max(252, int(min_observations))
    rows = []
    for ticker in historical_prices.columns:
        close = pd.to_numeric(historical_prices[ticker], errors="coerce").dropna()
        if len(close) < required:
            continue
        sig = _calculate_price_signals(close)
        if not sig["momentum_window"].startswith("12-1"):
            continue
        rows.append({"Ticker": str(ticker), "Status": "Candidate", "Reason": "",
                     "Observations": sig["observations"], "Momentum Window": sig["momentum_window"],
                     "Momentum": sig["momentum"], "Volatility": sig["volatility"],
                     "Max Drawdown": sig["max_drawdown"], "Earnings Yield": np.nan,
                     "Book-to-Market": np.nan, "ROE": np.nan, "Debt-to-Equity": np.nan})
    if not rows:
        return pd.DataFrame(), []
    ranked, _, selected = score_quant_candidates(pd.DataFrame(rows), top_n,
                                                   min_components=2,
                                                   active_factors=active_factors)
    return ranked.sort_values("Rank", na_position="last").reset_index(drop=True), selected


def screen_tradable_point_in_time(
    historical_prices,
    execution_prices,
    top_n=10,
    min_observations=80,
):
    """Point-in-time screening with explicit data diagnostics."""

    if historical_prices is None or historical_prices.empty:
        raise ValueError(
            "Point-in-time screening: historical price matrix is empty."
        )

    historical_prices = historical_prices.apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    quotes = pd.to_numeric(
        execution_prices, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    required = max(
        MIN_PRICE_OBSERVATIONS,
        int(min_observations),
    )

    counts = historical_prices.notna().sum()

    valid_history = counts[
        counts >= required
    ].index

    valid_quote = quotes.index[
        quotes.notna() & (quotes > 0)
    ]

    eligible = (
        historical_prices.columns
        .intersection(valid_history)
        .intersection(valid_quote)
    )

    raw_usable_days = int(
        (
            historical_prices.notna().sum(axis=1) >= 2
        ).sum()
    )

    print("\n========== SCREENING DIAGNOSTICS ==========")
    print("Matrix shape:", historical_prices.shape)
    print("Date range:", historical_prices.index.min(),
          historical_prices.index.max())
    print("Required observations:", required)
    print("Max observations per asset:", counts.max())
    print("Median observations per asset:", counts.median())
    print("Assets with sufficient history:", len(valid_history))
    print("Assets with valid execution price:",
          len(valid_quote))
    print("Eligible assets:", len(eligible))
    print("Raw days with >=2 quotes:", raw_usable_days)
    print("Top 10 observation counts:")
    print(counts.sort_values(ascending=False).head(10))
    print("===========================================\n")

    if len(eligible) < 2:
        raise ValueError(
            f"Point-in-time screening: {len(eligible)} eligible assets; "
            f"required={required}; "
            f"max_history={counts.max()}; "
            f"raw_common_days={raw_usable_days}."
        )

    ranked, selected = screen_prices_point_in_time(
        historical_prices.loc[:, eligible],
        top_n=top_n,
        min_observations=required,
    )

    if len(selected) < 2:
        raise ValueError(
            f"Point-in-time screening selected only "
            f"{len(selected)} assets from {len(eligible)} eligible."
        )

    return ranked, selected

# =========================================================
# STRICT HISTORICAL FOUR-FACTOR SCREENING
# =========================================================
HISTORICAL_FUNDAMENTALS_COLUMNS = (
    "ticker", "published_at", "earnings_yield", "book_to_market", "roe", "debt_to_equity"
)

def load_pit_fundamentals(csv_path):
    """Load a *verified* point-in-time fundamentals panel.

    Each row represents values known as of `published_at`, not today's Yahoo info.
    A real historical dataset is REQUIRED. Missing coverage raises an error.
    """
    from pathlib import Path
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Historical Value/Quality dataset not found: {path}. "
            "Four-factor historical backtesting is disabled until point-in-time data are supplied."
        )
    panel = pd.read_csv(path)
    missing = set(HISTORICAL_FUNDAMENTALS_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"Missing point-in-time fundamentals columns: {sorted(missing)}")
    panel['published_at'] = pd.to_datetime(panel['published_at'], errors='raise', utc=True).dt.tz_localize(None)
    panel['ticker'] = panel['ticker'].astype(str).str.strip()
    for field in HISTORICAL_FUNDAMENTALS_COLUMNS[2:]:
        panel[field] = pd.to_numeric(panel[field], errors='coerce')
    if panel.duplicated(['ticker', 'published_at']).any():
        raise ValueError('Duplicate ticker/published_at in historical fundamentals')
    return panel.sort_values(['ticker', 'published_at'])


def screen_quant_point_in_time(historical_prices, execution_prices, as_of_date,
                               fundamentals_panel=None, top_n=50, min_observations=252,
                               min_components=None, active_factors=("Momentum", "Risk")):
    """Historical Quant Selection with identical four-family score to live screening.

    Uses the last record published strictly before the decision date; a date-only
    publication timestamp is conservatively eligible starting the next day.
    Fails closed if 4 families cannot be computed for enough securities.
    """
    active_factors = tuple(active_factors)
    if min_components is None:
        min_components = len(active_factors)
    if not ("Value" in active_factors or "Quality" in active_factors):
        # Historical execution quotes are used ONLY for tradability.
        quotes = pd.to_numeric(execution_prices, errors="coerce")
        eligible = [ticker for ticker in historical_prices.columns
                    if ticker in quotes.index and pd.notna(quotes[ticker]) and quotes[ticker] > 0]
        ranking, selected = screen_prices_point_in_time(
            historical_prices[eligible], top_n=top_n,
            min_observations=max(252, int(min_observations)),
            active_factors=active_factors)
        if len(selected) < int(top_n):
            raise ValueError(f"Only {len(selected)} price-only PIT candidates for Top {top_n}")
        return ranking, selected
    if fundamentals_panel is None:
        raise ValueError("Historical Value/Quality requires point-in-time fundamentals")
    required = max(MIN_PRICE_OBSERVATIONS, int(min_observations))
    as_of = pd.Timestamp(as_of_date).tz_localize(None) if pd.Timestamp(as_of_date).tzinfo else pd.Timestamp(as_of_date)
    # Treat same-day releases as unavailable without verified release time.
    cutoff = as_of.normalize()
    panel = fundamentals_panel.loc[(fundamentals_panel['published_at'] < cutoff) & (fundamentals_panel['decision_date'] == cutoff)].copy()
    panel = panel.sort_values('published_at').drop_duplicates('ticker', keep='last').set_index('ticker')
    quotes = pd.to_numeric(execution_prices, errors='coerce')
    rows=[]
    for ticker in historical_prices.columns:
        series = pd.to_numeric(historical_prices[ticker], errors='coerce').dropna()
        if len(series) < required or ticker not in quotes.index or not np.isfinite(quotes[ticker]) or quotes[ticker] <= 0:
            continue
        if ticker not in panel.index:
            continue
        try:
            sig = _calculate_price_signals(series)
        except (ValueError, TypeError, ZeroDivisionError):
            continue
        f = panel.loc[ticker]
        price = float(series.iloc[-1])
        shares = f.get('shares', np.nan)
        market_cap = shares * price if np.isfinite(shares) and shares > 0 else np.nan
        equity = f.get('equity', np.nan)
        earnings = f.get('annual_earnings', np.nan)
        debt = f.get('debt', np.nan)
        earnings_yield = earnings / market_cap if np.isfinite(market_cap) and market_cap > 0 and np.isfinite(earnings) else np.nan
        book_to_market = equity / market_cap if np.isfinite(market_cap) and market_cap > 0 and np.isfinite(equity) else np.nan
        roe = earnings / equity if np.isfinite(equity) and equity != 0 and np.isfinite(earnings) else np.nan
        debt_to_equity = debt / equity if np.isfinite(equity) and equity != 0 and np.isfinite(debt) else np.nan
        rows.append({'Ticker':ticker, 'Status':'Candidate', 'Reason':'',
                     'Observations':len(series), 'Momentum Window':sig['momentum_window'],
                     'Momentum':sig['momentum'], 'Volatility':sig['volatility'],
                     'Max Drawdown':sig['max_drawdown'], 'Earnings Yield':earnings_yield,
                     'Book-to-Market':book_to_market, 'ROE':roe,
                     'Debt-to-Equity':debt_to_equity, 'Fundamentals Published At':f['published_at'],
                     'Income Filed At':f.get('income_filed_at'), 'Equity Filed At':f.get('equity_filed_at'),
                     'Shares Filed At':f.get('shares_filed_at'), 'Debt Filed At':f.get('debt_filed_at')})
    if not rows:
        raise ValueError('No historical candidates with both prices and published fundamentals; no present-day fallback permitted.')
    ranked, eligible, selected = score_quant_candidates(pd.DataFrame(rows), top_n, min_components, active_factors=active_factors)
    if len(selected) < int(top_n):
        raise ValueError(f'Only {len(selected)} assets passed strict historical four-factor Quant Selection; requested {top_n}.')
    # Return the full scored table, NOT the pre-selection `eligible` copy:
    # `score_quant_candidates` assigns Selected/Status on `valid` after
    # creating `eligible`, so the latter has Selected=False for every row.
    ranked = ranked.sort_values("Rank", na_position="last").reset_index(drop=True)
    return ranked, selected
