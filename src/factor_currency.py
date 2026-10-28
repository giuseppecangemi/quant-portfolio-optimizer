"""Market-aware Fama-French expected returns. Prices must be native currency."""
import pandas as pd
from src.fama_french import fit_ff3, estimate_ff3_eur_returns
from src.fama_french_5 import fit_ff5, estimate_ff5_eur_returns


def estimate_factor_returns(prices, factors, eurusd_prices=None, *, market, model, return_stats=False):
    if model not in ("ff3", "ff5"):
        raise ValueError(f"Unsupported model: {model}")
    prices = prices.sort_index().dropna()
    if len(prices) < 81:
        raise ValueError("Insufficient asset price observations")
    if market == "USA":
        # US stocks, FF factors, and portfolio covariance are all in USD.
        returns = prices.pct_change().dropna()
        available = factors.loc[factors.index < prices.index[-1]]
        stats = (fit_ff3 if model == "ff3" else fit_ff5)(returns, available)
        column = "FF3 Expected Return USD" if model == "ff3" else "FF5 Expected Return USD"
        expected = stats[column].reindex(prices.columns)
        if expected.isna().any():
            raise ValueError("Missing USD factor expected returns")
        return (expected, stats) if return_stats else expected
    if market == "Borsa Italiana":
        if eurusd_prices is None:
            raise ValueError("EURUSD required for European USD-denominated factors")
        estimator = estimate_ff3_eur_returns if model == "ff3" else estimate_ff5_eur_returns
        return estimator(prices, factors, eurusd_prices, return_stats=return_stats)
    raise ValueError(f"Market {market!r} needs an explicit factor/currency mapping")
