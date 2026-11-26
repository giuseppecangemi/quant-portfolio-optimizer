"""Technical diagnostics for a synthetic close series."""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.volatility_forecasting import ewma_variance


def rsi(series, window):
    change = series.diff()
    gains = change.clip(lower=0).ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    losses = (-change.clip(upper=0)).ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    return (100 * gains / (gains + losses)).mask((gains + losses).eq(0), 50)


def technical(value, sma, ema, regression, rsi_period, macd, bollinger, *, periods_per_year=252):
    out = pd.DataFrame(index=value.index)
    for n in sma:
        out[f"SMA {n}"] = value.rolling(n).mean()
    for n in ema:
        out[f"EMA {n}"] = value.ewm(span=n, adjust=False, min_periods=n).mean()
    log = np.log(value)
    for n in regression:
        x = np.arange(n, dtype=float) - (n-1)/2
        sxx = x @ x
        roll = log.rolling(n)
        slope = roll.apply(lambda y: x @ (y-y.mean()) / sxx, raw=True)
        sd = roll.std(ddof=0)
        residual = np.sqrt((sd**2 - slope**2 * sxx/n).clip(lower=0))
        line = roll.mean() + slope * (n-1)/2
        out[f"Slope {n}"] = slope
        out[f"Annual slope {n}"] = np.expm1(periods_per_year*slope)
        out[f"R² {n}"] = (slope**2 * sxx / n / sd.where(sd.gt(0))**2).clip(0, 1)
        out[f"Regression {n}"] = np.exp(line)
        out[f"Regression low {n}"] = np.exp(line - 2*residual)
        out[f"Regression high {n}"] = np.exp(line + 2*residual)
    fast, slow, signal = macd
    out["RSI"] = rsi(value, rsi_period)
    out["MACD"] = (value.ewm(span=fast, adjust=False, min_periods=fast).mean()
                   - value.ewm(span=slow, adjust=False, min_periods=slow).mean())
    out["Signal"] = out["MACD"].ewm(span=signal, adjust=False, min_periods=signal).mean()
    out["Histogram"] = out["MACD"] - out["Signal"]
    for n in (21, 63, 126, 252):
        out[f"ROC {n}"] = value.pct_change(n, fill_method=None)
    mean, std = value.rolling(bollinger).mean(), value.rolling(bollinger).std(ddof=0)
    out["Bollinger high"], out["Bollinger low"] = mean + 2*std, mean - 2*std
    return out


def swings(value, sensitivity, distance):
    log = np.log(value.to_numpy())
    changes = np.diff(log)
    if len(changes) < 3:
        return []
    sigma = np.std(changes, ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        return []
    found = []
    for sign in (1, -1):
        points, details = find_peaks(sign * log, prominence=sigma*sensitivity, distance=distance)
        found.extend((i, sign, p) for i, p in zip(points, details["prominences"]))
    return sorted(found)


def price_zones(value, sensitivity=2., distance=7, min_contacts=2, width=1.5, limit=8):
    points = swings(value, sensitivity, distance)
    if not points:
        return pd.DataFrame()
    log = np.log(value.to_numpy())
    sigma = np.std(np.diff(log), ddof=1)
    tolerance = width*sigma
    clusters = []
    for item in sorted(points, key=lambda p: log[p[0]]):
        if not clusters or log[item[0]] - log[clusters[-1][0][0]] > tolerance:
            clusters.append([])
        clusters[-1].append(item)
    rows = []
    current = log[-1]
    for cluster in clusters:
        independent = []
        for item in sorted(cluster):
            if not independent or item[0] - independent[-1][0] >= distance:
                independent.append(item)
        if len(independent) < min_contacts:
            continue
        idx = np.array([i for i, _, _ in independent])
        levels = log[idx]
        center = levels.mean()
        end = len(log)-1
        move = [np.max(np.maximum(0., -sign*(log[i+1:min(end, i+distance)+1]-log[i])))
                for i, sign, _ in independent]
        prominence = np.mean([p for _, _, p in independent]) / sigma
        recency = np.exp(-(end-idx)/(len(log)/2)).mean()
        score = len(idx)*recency*(prominence + np.mean(move)/sigma)/(1+np.std(levels)/sigma)
        middle = np.exp(center)
        rows.append({"Type": "Support" if center < current else "Resistance", "Level": middle,
                     "Lower": np.exp(levels.min()-tolerance/2), "Upper": np.exp(levels.max()+tolerance/2),
                     "Contacts": len(idx), "First": value.index[idx[0]], "Last": value.index[idx[-1]],
                     "Score": score, "Distance": middle-value.iloc[-1], "Distance %": np.expm1(center-current)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.nlargest(limit, "Score").sort_values("Distance", key=abs).reset_index(drop=True)


def trend_labels(value):
    labels = {}
    for label, n in (("Short", 21), ("Medium", 63), ("Long", 252)):
        if len(value) < n+5:
            labels[label] = "Unavailable"
            continue
        recent = value.tail(n)
        ma = value.rolling(n).mean()
        pivots = swings(recent, 1.5, 3)
        highs = [recent.iloc[i] for i, sign, _ in pivots if sign == 1]
        lows = [recent.iloc[i] for i, sign, _ in pivots if sign == -1]
        sequence = (np.sign(highs[-1]-highs[-2])+np.sign(lows[-1]-lows[-2]))/2 if min(len(highs), len(lows)) >= 2 else 0
        x = np.arange(n) - (n-1)/2
        score = sum(np.sign(s) for s in (recent.iloc[-1]-ma.iloc[-1], ma.iloc[-1]-ma.iloc[-6],
                                        recent.iloc[-1]-recent.iloc[0], x @ (np.log(recent)-np.log(recent).mean()))) + sequence
        labels[label] = ("Strong Bullish" if score >= 4 else "Bullish" if score >= 2 else
                         "Strong Bearish" if score <= -4 else "Bearish" if score <= -2 else "Neutral")
    return labels


def risk_dynamics(value, window, rf):
    returns = value.pct_change(fill_method=None)
    daily_rf = np.expm1(np.log1p(rf)/252)
    excess = returns - daily_rf
    vol = returns.rolling(window).std()
    downside = np.sqrt(excess.clip(upper=0).pow(2).rolling(window).mean())
    drawdown = value / value.cummax() - 1
    duration = drawdown.lt(0).groupby(drawdown.eq(0).cumsum()).cumsum()
    ewma_inputs = returns.dropna()
    ewma = pd.Series(ewma_variance(ewma_inputs), index=ewma_inputs.index)
    ewma.iloc[:21] = np.nan  # Show estimates only after the shared EWMA seed is observable.
    return pd.DataFrame({
        "Daily volatility": vol, "Annualized volatility": vol*np.sqrt(252),
        "Downside volatility": downside*np.sqrt(252), "Drawdown": drawdown, "Drawdown duration": duration,
        "Rolling Sharpe": excess.rolling(window).mean()/vol.where(vol.gt(0))*np.sqrt(252),
        "Rolling Sortino": excess.rolling(window).mean()/downside.where(downside.gt(0))*np.sqrt(252),
        "EWMA": np.sqrt(ewma*252),
    })


def breadth(prices, weights, rsi_period, limits, macd):
    invested = weights.reindex(columns=prices.columns).where(lambda x: x.gt(0))
    flags = {}
    for n in (20, 50, 200):
        ma = prices.rolling(n).mean()
        flags[f"Above SMA {n}"] = prices.gt(ma).where(ma.notna())
    change = prices.pct_change(21, fill_method=None)
    flags["Positive momentum"] = change.gt(0).where(change.notna())
    fast, slow, signal = macd
    divergence = (prices.ewm(span=fast, min_periods=fast, adjust=False).mean()
                  - prices.ewm(span=slow, min_periods=slow, adjust=False).mean())
    trigger = divergence.ewm(span=signal, min_periods=signal, adjust=False).mean()
    flags["Bullish MACD"] = divergence.gt(trigger).where(trigger.notna())
    strength = rsi(prices, rsi_period)
    flags["Overbought RSI"] = strength.gt(limits[1]).where(strength.notna())
    flags["Oversold RSI"] = strength.lt(limits[0]).where(strength.notna())
    equal, weighted, latest = {}, {}, {}
    for label, signal_values in flags.items():
        observed = signal_values.where(invested.notna()).astype(float)
        eligible = invested.where(observed.notna())
        equal[label] = observed.mean(axis=1)
        weighted[label] = (observed*eligible).sum(axis=1, min_count=1) / eligible.sum(axis=1, min_count=1)
        latest[label] = observed.iloc[-1]
    return pd.DataFrame(equal), pd.DataFrame(weighted), pd.DataFrame(latest)


def relative_strength(value, benchmark, window):
    common = pd.concat([value.rename("Portfolio"), benchmark.rename("Benchmark")], axis=1).dropna()
    if len(common) < 2:
        return pd.DataFrame()
    index = common.div(common.iloc[0]) * 100
    r = common.pct_change(fill_method=None)
    excess = r.Portfolio-r.Benchmark
    roll = r.Benchmark.rolling(window)
    tracking = excess.rolling(window).std()
    return index.assign(**{
        "Relative return": index.Portfolio/index.Benchmark-1, "Relative strength": index.Portfolio/index.Benchmark,
        "Rolling correlation": r.Portfolio.rolling(window).corr(r.Benchmark),
        "Rolling beta": r.Portfolio.rolling(window).cov(r.Benchmark)/roll.var().where(roll.var().gt(0)),
        "Tracking error": tracking*np.sqrt(252),
        "Information ratio": excess.rolling(window).mean()/tracking.where(tracking.gt(0))*np.sqrt(252),
    })


CHART_PERIODS = {"1D": ("D", "giorni", 252), "1W": ("W-FRI", "settimane", 52),
                 "1M": ("M", "mesi", 12), "1Q": ("Q-DEC", "trimestri", 4)}


def portfolio_ohlc(value, timeframe):
    """Aggregate synthetic daily closes and date each bar at its last observed close."""
    if timeframe == "1D":
        raise ValueError("Daily candles require synchronized intraday portfolio values.")
    observed = value.dropna().sort_index()
    periods = observed.index.to_period(CHART_PERIODS[timeframe][0])
    bars = observed.groupby(periods).ohlc()
    bars.index = pd.DatetimeIndex(observed.index.to_series().groupby(periods).last(), name=value.index.name)
    return bars
