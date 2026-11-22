"""Trailing statistics on valid daily index closes."""
import pandas as pd

PERIODS = {"1M": (1, 0), "3M": (3, 0), "6M": (6, 0), "1Y": (0, 1),
           "3Y": (0, 3), "5Y": (0, 5), "10Y": (0, 10), "MAX": (0, 0), "Custom": (0, 0)}


def date_bounds(index, period, custom=None):
    if custom is not None:
        return tuple(map(pd.Timestamp, custom))
    months, years = PERIODS[period]
    end = index[-1]
    start = max(index[0], end - pd.DateOffset(months=months, years=years)) if months or years else index[0]
    return start, end


def rolling_statistics(close, window):
    if window < 2:
        raise ValueError("A rolling window needs at least two observations.")
    close = close.dropna().sort_index()
    rolling = close.rolling(window, min_periods=window)
    mean, std = rolling.mean(), rolling.std(ddof=1)
    result = pd.DataFrame({
        "Close": close, "Mean": mean, "Std": std,
        "Z": (close - mean).div(std.where(std.gt(0))),
        "Percentile": rolling.rank(method="max", pct=True) * 100,
        "Min": rolling.min(), "Max": rolling.max(),
        "Change": close.diff(), "Change %": close.pct_change(fill_method=None) * 100,
    })
    for k in (1, 2, 3):
        result[f"Upper {k}"] = mean + k * std
        result[f"Lower {k}"] = mean - k * std
    return result


def compare_markets(vix, vstoxx, window):
    common = pd.concat([vix.rename("VIX"), vstoxx.rename("VSTOXX")], axis=1, join="inner").dropna().sort_index()
    return common, common["VIX"].rolling(window).corr(common["VSTOXX"]), common["VSTOXX"] - common["VIX"]
