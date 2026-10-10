"""Historical adjusted close prices with defensive FTSE MIB resolution."""
import time
import pandas as pd
import yfinance as yf

# Yahoo may not provide FTSEMIB.MI consistently. ^FTSEMIB is the
# corresponding FTSE MIB index quote; retain original column label.
_INDEX_ALIASES = {"FTSEMIB.MI": ("FTSEMIB.MI", "^FTSEMIB", "^FTMIB")}


def _close_series(frame, ticker):
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
    close = close[close > 0]
    if len(close) < 2:
        return None
    close.name = ticker
    return close


def _fetch_one(symbol, period):
    errors = []
    for attempt in range(2):
        try:
            frame = yf.download(
                symbol, period=period, auto_adjust=True,
                progress=False, threads=False, timeout=15,
            )
            close = _close_series(frame, symbol)
            if close is not None:
                return close, errors
            errors.append(f"download({symbol}): empty/insufficient")
        except Exception as exc:
            errors.append(f"download({symbol}): {type(exc).__name__}: {exc}")
        try:
            frame = yf.Ticker(symbol).history(period=period, auto_adjust=True, timeout=15)
            close = _close_series(frame, symbol)
            if close is not None:
                return close, errors
            errors.append(f"history({symbol}): empty/insufficient")
        except Exception as exc:
            errors.append(f"history({symbol}): {type(exc).__name__}: {exc}")
        if attempt == 0:
            time.sleep(0.7)
    return None, errors


def get_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Return aligned adjusted close prices, preserving requested ticker labels.

    FTSEMIB.MI fallback symbols refer to the index, not an ETF or proxy.
    Raises an explicit error if no source has sufficient data.
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
    result = pd.concat(prices, axis=1).sort_index().dropna(how="any")
    if len(result) < 2:
        raise ValueError("Dati storici insufficienti dopo l'allineamento dei ticker")
    return result
