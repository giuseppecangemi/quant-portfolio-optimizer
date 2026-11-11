"""Market-aware Fama-French expected returns. Prices must be native currency."""
import pandas as pd
from src.fama_french import fit_ff3
from src.factor_eur import convert_window_factors
from src.fama_french_5 import fit_ff5


def factors_for_backtest_signal(factors, signal_date):
    """Apply a two-calendar-month publication buffer to backtest estimates."""
    cutoff = pd.Timestamp(signal_date) - pd.DateOffset(months=2)
    return factors.loc[factors.index <= cutoff]


def factor_sample(prices, factors, eurusd_prices, *, market):
    prices = prices.sort_index()
    if market == "Borsa Italiana":
        if eurusd_prices is None:
            raise ValueError("EURUSD required for European USD-denominated factors")
        available = convert_window_factors(factors, eurusd_prices, prices)
    elif market == "USA":
        available = factors.loc[factors.index < prices.index[-1]]
    else:
        raise ValueError(f"Market {market!r} needs an explicit factor/currency mapping")

    if market == "Borsa Italiana":
        previous_session = pd.Series(available.attrs["interval_starts"], dtype="datetime64[ns]").reindex(prices.index)
    else:
        sessions = prices.index.union(factors.index).sort_values()
        previous_session = sessions.to_series().shift().reindex(prices.index)
    previous_price_date = prices.index.to_series().shift()
    returns = prices.pct_change(fill_method=None).loc[previous_price_date.eq(previous_session)]
    returns = returns.dropna()
    common = returns.index.intersection(available.index)
    return returns.loc[common], available.loc[common]


def estimate_factor_returns(prices, factors, eurusd_prices=None, *, market, model, return_stats=False):
    if model not in ("ff3", "ff5"):
        raise ValueError(f"Unsupported model: {model}")
    returns, available = factor_sample(prices, factors, eurusd_prices, market=market)
    stats = (fit_ff3 if model == "ff3" else fit_ff5)(returns, available)
    column = f"{model.upper()} Expected Return USD"
    expected = stats[column].reindex(prices.columns)
    if expected.isna().any():
        raise ValueError("Missing factor expected returns")
    if market == "Borsa Italiana":
        stats = stats.rename(columns={column: column.replace("USD", "EUR"),
                                      "Historical Return USD": "Historical Return EUR"})
    return (expected, stats) if return_stats else expected
