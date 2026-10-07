import numpy as np
import pandas as pd
import yfinance as yf


def _safe_price_history(ticker: str, period: str = "2y") -> pd.Series | None:
    """
    Scarica lo storico di un singolo ticker.
    Un ticker non valido/non disponibile restituisce None e non blocca lo screening.
    """
    try:
        data = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data is None or data.empty or "Close" not in data:
            return None

        close = data["Close"].squeeze().dropna()

        if len(close) < 80:
            return None

        return close.astype(float)

    except Exception:
        return None


def _safe_fundamentals(ticker: str) -> dict:
    """
    Recupera i fondamentali senza propagare errori Yahoo Finance.
    Campi mancanti restano NaN.
    """
    result = {
        "earnings_yield": np.nan,
        "book_to_market": np.nan,
        "roe": np.nan,
        "debt_to_equity": np.nan,
    }

    try:
        info = yf.Ticker(ticker).info or {}

        trailing_pe = info.get("trailingPE")
        price_to_book = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        debt_to_equity = info.get("debtToEquity")

        if trailing_pe is not None and np.isfinite(trailing_pe) and trailing_pe > 0:
            result["earnings_yield"] = 1.0 / float(trailing_pe)

        if (
            price_to_book is not None
            and np.isfinite(price_to_book)
            and price_to_book > 0
        ):
            result["book_to_market"] = 1.0 / float(price_to_book)

        if roe is not None and np.isfinite(roe):
            result["roe"] = float(roe)

        if debt_to_equity is not None and np.isfinite(debt_to_equity):
            # Yahoo normalmente esprime D/E in percentuale.
            result["debt_to_equity"] = float(debt_to_equity) / 100.0

    except Exception:
        pass

    return result


def _percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Converte una metrica in percentile 0-100.
    I NaN rimangono NaN.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)

    score = numeric.rank(
        pct=True,
        method="average",
        ascending=True,
    ) * 100.0

    if not higher_is_better:
        score = 100.0 - score

    return score


def _calculate_price_signals(close: pd.Series) -> dict:
    """
    Segnali price-based:
    - Momentum 12-1 quando lo storico lo consente.
    - In alternativa momentum 6-1.
    - Volatilità annualizzata.
    - Maximum drawdown.
    """
    returns = close.pct_change().dropna()

    if len(close) >= 253:
        momentum = (close.iloc[-22] / close.iloc[-253]) - 1.0
        momentum_window = "12-1"
    elif len(close) >= 127:
        momentum = (close.iloc[-22] / close.iloc[-127]) - 1.0
        momentum_window = "6-1"
    else:
        momentum = (close.iloc[-1] / close.iloc[0]) - 1.0
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


def screen_universe(
    universe: pd.DataFrame,
    period: str = "2y",
    top_n: int = 10,
    min_components: int = 2,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Screening quantitativo dell'universo.

    Componenti:
    - Momentum: rendimento 12-1 (o fallback 6-1).
    - Risk: combinazione di bassa volatilità e drawdown meno severo.
    - Value: Earnings Yield + Book-to-Market.
    - Quality: ROE + basso Debt-to-Equity.

    Ogni componente viene trasformata in percentile 0-100.
    Il Quant Score è la media dei componenti disponibili.

    Regola di robustezza:
    - ticker senza prezzi validi -> Excluded, ma nessun crash;
    - fondamentali mancanti -> NaN, senza crash;
    - servono almeno `min_components` componenti valide per essere Eligible.
    """
    required = {"name", "ticker"}

    if not required.issubset(universe.columns):
        raise ValueError(
            "Universe must contain at least 'name' and 'ticker' columns."
        )

    rows = []

    for _, instrument in universe.iterrows():
        name = str(instrument["name"])
        ticker = str(instrument["ticker"]).strip()

        row = {
            "Instrument": name,
            "Ticker": ticker,
            "Status": "Excluded",
            "Reason": "",
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

        if not ticker or ticker.lower() == "nan":
            row["Reason"] = "Missing ticker"
            rows.append(row)
            continue

        close = _safe_price_history(ticker, period=period)

        if close is None:
            row["Reason"] = "Invalid ticker / no price data"
            rows.append(row)
            continue

        try:
            price_signals = _calculate_price_signals(close)
        except Exception:
            row["Reason"] = "Insufficient or invalid price history"
            rows.append(row)
            continue

        fundamentals = _safe_fundamentals(ticker)

        row.update({
            "Status": "Candidate",
            "Reason": "",
            "Observations": price_signals["observations"],
            "Momentum Window": price_signals["momentum_window"],
            "Momentum": price_signals["momentum"],
            "Volatility": price_signals["volatility"],
            "Max Drawdown": price_signals["max_drawdown"],
            "Earnings Yield": fundamentals["earnings_yield"],
            "Book-to-Market": fundamentals["book_to_market"],
            "ROE": fundamentals["roe"],
            "Debt-to-Equity": fundamentals["debt_to_equity"],
        })

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:
        return result, []

    valid_mask = result["Status"].eq("Candidate")
    valid = result.loc[valid_mask].copy()

    if valid.empty:
        result["Selected"] = False
        result["Rank"] = np.nan
        return result, []

    # -----------------------------------------------------
    # Momentum score
    # -----------------------------------------------------
    valid["Momentum Score"] = _percentile_score(
        valid["Momentum"],
        higher_is_better=True,
    )

    # -----------------------------------------------------
    # Risk score
    # Bassa volatilità + drawdown meno profondo = meglio.
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
    # Value score
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
    # Quality score
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

    component_columns = [
        "Momentum Score",
        "Risk Score",
        "Value Score",
        "Quality Score",
    ]

    valid["Components Available"] = (
        valid[component_columns].notna().sum(axis=1)
    )

    # Media dei componenti disponibili: nessuna imputazione arbitraria.
    valid["Quant Score"] = valid[
        component_columns
    ].mean(axis=1, skipna=True)

    insufficient = valid["Components Available"] < min_components
    valid.loc[insufficient, "Status"] = "Excluded"
    valid.loc[insufficient, "Reason"] = "Insufficient scoring data"
    valid.loc[insufficient, "Quant Score"] = np.nan

    eligible = valid[
        valid["Status"].eq("Candidate")
        & valid["Quant Score"].notna()
    ].copy()

    eligible = eligible.sort_values(
        ["Quant Score", "Momentum Score"],
        ascending=[False, False],
    )

    eligible["Rank"] = np.arange(1, len(eligible) + 1)
    selected_tickers = eligible.head(max(0, int(top_n)))["Ticker"].tolist()

    valid["Selected"] = valid["Ticker"].isin(selected_tickers)
    valid.loc[valid["Selected"], "Status"] = "Selected"

    rank_map = eligible.set_index("Ticker")["Rank"]
    valid["Rank"] = valid["Ticker"].map(rank_map)

    # Riporta i risultati calcolati nel dataframe completo.
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

    for column in computed_columns:
        if column not in result.columns:
            if column == "Selected":
                result[column] = False
            else:
                result[column] = np.nan

    valid_by_ticker = valid.set_index("Ticker")

    for idx, ticker in result["Ticker"].items():
        if ticker in valid_by_ticker.index:
            for column in computed_columns:
                result.at[idx, column] = valid_by_ticker.at[ticker, column]

    result["Selected"] = result["Selected"].fillna(False).astype(bool)

    # Prima gli eligible ordinati per rank, poi gli esclusi.
    result["_sort_rank"] = result["Rank"].fillna(10**9)
    result = (
        result.sort_values(
            ["_sort_rank", "Instrument"],
            ascending=[True, True],
        )
        .drop(columns="_sort_rank")
        .reset_index(drop=True)
    )

    return result, selected_tickers
