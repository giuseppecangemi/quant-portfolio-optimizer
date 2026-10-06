import pandas as pd
import yfinance as yf


def get_prices(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """
    Scarica i prezzi storici di una lista di strumenti finanziari.

    Parameters
    ----------
    tickers : list[str]
        Lista dei ticker, ad esempio ["AAPL", "MSFT", "SPY"].
    period : str
        Periodo storico richiesto da Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        DataFrame con i prezzi di chiusura.
    """

    prices = {}

    for ticker in tickers:
        data = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            raise ValueError(f"Nessun dato trovato per {ticker}")

        prices[ticker] = data["Close"].squeeze()

    prices = pd.DataFrame(prices)

    return prices.dropna()