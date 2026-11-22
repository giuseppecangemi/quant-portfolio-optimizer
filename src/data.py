import time
import pandas as pd
import yfinance as yf

# Yahoo may not provide FTSEMIB.MI consistently. ^FTSEMIB is the
# corresponding FTSE MIB index quote; retain original column label.
_INDEX_ALIASES = {"FTSEMIB.MI": ("FTSEMIB.MI", "^FTSEMIB", "^FTMIB")}


def _close_series(frame, ticker, *, preserve_invalid=False):
    if frame is None or frame.empty:
        return None
    close = frame.get("Close")
    if close is None:
        return None
    if isinstance(close, pd.DataFrame):
        if close.empty:
            return None
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if not preserve_invalid:
        close = close[close > 0]
    if len(close) < 2:
        return None
    close.name = ticker
    return close


def _fetch_one(symbol, period, *, preserve_invalid=False):
    errors = []
    for attempt in range(2):
        try:
            frame = yf.download(
                symbol, period=period, auto_adjust=True,
                progress=False, threads=False, timeout=15,
            )
            close = _close_series(frame, symbol, preserve_invalid=preserve_invalid)
            if close is not None:
                return close, errors
            errors.append(f"download({symbol}): empty/insufficient")
        except Exception as exc:
            errors.append(f"download({symbol}): {type(exc).__name__}: {exc}")
        try:
            frame = yf.Ticker(symbol).history(period=period, auto_adjust=True, timeout=15)
            close = _close_series(frame, symbol, preserve_invalid=preserve_invalid)
            if close is not None:
                return close, errors
            errors.append(f"history({symbol}): empty/insufficient")
        except Exception as exc:
            errors.append(f"history({symbol}): {type(exc).__name__}: {exc}")
        if attempt == 0:
            time.sleep(0.7)
    return None, errors


def get_prices(tickers: list[str], period: str = "5y", *, preserve_missing: bool = False) -> pd.DataFrame:
    """Return aligned adjusted close prices, preserving requested ticker labels.

    FTSEMIB.MI fallback symbols refer to the index, not an ETF or proxy.
    Raises an explicit error if no source has sufficient data.
    preserve_missing=True keeps the union of trading dates for backtests;
    the default retains complete-case behavior for existing estimators.
    """
    if not tickers:
        raise ValueError("Ticker list is empty")
    prices = {}
    for ticker in tickers:
        errors = []
        for symbol in _INDEX_ALIASES.get(ticker, (ticker,)):
            close, attempts = _fetch_one(symbol, period)
            errors.extend(attempts)
            if close is not None:
                prices[ticker] = close.rename(ticker)
                break
        else:
            raise ValueError(
                f"Nessun dato trovato per {ticker} ({period}). "
                f"Tentativi: {', '.join(_INDEX_ALIASES.get(ticker, (ticker,)))}. "
                f"Dettagli: {'; '.join(errors[-6:])}"
            )
    result = pd.concat(prices, axis=1).sort_index()
    if not preserve_missing:
        result = result.dropna(how="any")
    if len(result) < 2:
        raise ValueError("Dati storici insufficienti dopo l'allineamento dei ticker")
    return result


def download_adjusted_close_range(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download adjusted prices by absolute dates, preserving partial ticker histories."""
    tickers = list(dict.fromkeys(str(t).strip() for t in tickers if str(t).strip()))
    if not tickers:
        return pd.DataFrame()
    pieces = []
    # Small batches are more tolerant of failures in large market universes.
    for offset in range(0, len(tickers), 25):
        batch = tickers[offset:offset + 25]
        try:
            raw = yf.download(
                tickers=batch, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"), auto_adjust=True,
                progress=False, threads=False, group_by="column", timeout=15,
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" not in raw.columns.get_level_values(0):
                    continue
                close = raw["Close"]
            else:
                if "Close" not in raw:
                    continue
                close = raw[["Close"]].rename(columns={"Close": batch[0]})
            if isinstance(close, pd.Series):
                close = close.to_frame(name=batch[0])
            close.columns = [str(c) for c in close.columns]
            pieces.append(close)
        except Exception:
            continue
    if not pieces:
        return pd.DataFrame()
    frame = pd.concat(pieces, axis=1).sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
