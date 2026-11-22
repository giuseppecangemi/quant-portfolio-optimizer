"""UI-independent preparation for historical portfolio simulations."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.fama_french import load_europe_ff3, load_us_ff3
from src.fama_french_5 import load_europe_ff5, load_us_ff5
from src.transaction_costs import CostAssumptions


@dataclass(frozen=True)
class MarketInputs:
    prices: pd.DataFrame
    benchmark: pd.DataFrame
    market: str
    ff3: pd.DataFrame | None = None
    ff5: pd.DataFrame | None = None
    fx: pd.DataFrame | None = None
    fundamentals: pd.DataFrame | None = None


def prepare_market_inputs(prices, benchmark, *, market, families=(), ff3=None, ff5=None,
                          fx=None, fetch_fx: Callable | None = None, fundamentals=None,
                          factor_loaders=None, allow_download=True, require_benchmark=True):
    """Resolve shared feeds once from frames or date-aware loaders."""
    prices = prices() if callable(prices) else prices
    benchmark = benchmark() if callable(benchmark) else benchmark
    if prices.empty:
        raise ValueError("No historical asset prices were downloaded.")
    if require_benchmark and benchmark.empty:
        raise ValueError("Benchmark historical prices are unavailable.")
    factors = {"FF3": ff3, "FF5": ff5}
    errors = {}
    if families and market != "USA":
        try:
            if fx is None and fetch_fx is not None:
                fx = fetch_fx()
            if fx is None or fx.empty:
                raise ValueError("EUR/USD historical prices are unavailable.")
        except Exception as exc:
            errors = dict.fromkeys(families, str(exc))
    if factor_loaders is None:
        factor_loaders = {
            "FF3": load_us_ff3 if market == "USA" else load_europe_ff3,
            "FF5": load_us_ff5 if market == "USA" else load_europe_ff5,
        }
    for family in families:
        if family not in errors and factors[family] is None:
            try:
                if not allow_download:
                    raise ValueError("Historical factor data missing from frozen inputs.")
                factors[family] = factor_loaders[family]()
            except Exception as exc:
                errors[family] = str(exc)
    return MarketInputs(prices, benchmark, market, factors["FF3"], factors["FF5"], fx, fundamentals), errors


@dataclass(frozen=True)
class WalkForwardWindow:
    prices: pd.DataFrame
    training_dates: pd.DatetimeIndex
    oos_dates: pd.DatetimeIndex
    selection_policy: str
    models: tuple[str, ...]
    rebalance_days: int
    transaction_costs: CostAssumptions | None
    risk_free_rate: float
    max_weight: float
    top_n: int | None
    capital: float

    @property
    def training_start(self):
        return self.training_dates[0]

    @property
    def information_cutoff(self):
        return self.training_dates[-1]

    @property
    def training_end(self):
        return self.information_cutoff

    @property
    def investment_decision_date(self):
        return self.information_cutoff

    @property
    def execution_date(self):
        return self.oos_dates[0]

    @property
    def oos_start(self):
        return self.execution_date

    @property
    def oos_end(self):
        return self.oos_dates[-1]

    @property
    def historical_universe(self):
        return tuple(self.prices.columns)

    @property
    def factor_observation_cutoff(self):
        return self.information_cutoff - pd.DateOffset(months=2)


def prepare_walk_forward_window(prices, lookback_days, *, selection_policy, models,
                                rebalance_days, transaction_costs, risk_free_rate,
                                max_weight, top_n, capital):
    """Resolve training and execution dates using the engine's price conventions."""
    panel = prices.sort_index()
    panel = panel.loc[~panel.index.duplicated(keep="last")].dropna(axis=1, how="all")
    if lookback_days < 1 or len(panel) < lookback_days + 2:
        raise ValueError("Walk-forward requires training prices and at least two OOS sessions.")
    return WalkForwardWindow(panel, panel.index[:lookback_days], panel.index[lookback_days:],
                             selection_policy, tuple(models), rebalance_days, transaction_costs,
                             risk_free_rate, max_weight, top_n, capital)


def benchmark_curve(prices, start, end, capital, *, required_dates=None):
    """Native closes. Optionally require every portfolio date to have a real mark."""
    if prices.empty:
        raise ValueError("Benchmark historical prices are unavailable.")
    series = prices.iloc[:, 0]
    close = series.loc[(series.index >= start) & (series.index <= end)].dropna()
    if required_dates is not None:
        valid_close = close.loc[np.isfinite(close) & close.gt(0)]
        missing = required_dates.difference(valid_close.index)
        if len(missing):
            raise ValueError(f"Benchmark missing close on {missing[0]:%Y-%m-%d} ({len(missing)} OOS dates)")
        close = close.reindex(required_dates)
    if len(close) < 2:
        raise ValueError("Not enough FTSE MIB data for the selected backtest period.")
    nav = (close / close.iloc[0]) * capital
    return pd.DataFrame({"Portfolio Value": nav, "Daily Return": nav.pct_change().fillna(0.)})
