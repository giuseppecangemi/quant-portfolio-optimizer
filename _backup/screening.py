import time

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# CONFIGURAZIONE
# =========================================================

MIN_PRICE_OBSERVATIONS = 80
PRICE_BATCH_SIZE = 40
PRICE_BATCH_RETRIES = 2
PRICE_RETRY_SLEEP_SECONDS = 1.0

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
                threads=True,
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

    if len(close) >= 253:
        momentum = close.iloc[-22] / close.iloc[-253] - 1.0
        momentum_window = "12-1"

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

def screen_universe(
    universe: pd.DataFrame,
    period: str = "2y",
    top_n: int = 10,
    min_components: int = 2,
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

        fundamentals, fundamental_status = _safe_fundamentals(ticker)

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
