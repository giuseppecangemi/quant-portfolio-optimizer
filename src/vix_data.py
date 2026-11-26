"""Official index history and a separate spot VIX snapshot."""
from io import StringIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st

from src.trading_market import MAX_AGE, aware_timestamp, fetch_intraday, positive

SOURCES = {
    "VIX": ("Cboe", "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"),
    "VSTOXX": ("STOXX", "https://www.stoxx.com/document/Indices/Current/HistoricalData/h_v2tx.txt"),
}


def parse_history(index, body):
    stoxx = index == "VSTOXX"
    frame = pd.read_csv(StringIO(body), sep=";" if stoxx else ",")
    frame.columns = frame.columns.str.strip().str.upper()
    columns = {"DATE", "SYMBOL", "INDEXVALUE"} if stoxx else {"DATE", "CLOSE"}
    if not columns.issubset(frame.columns):
        raise ValueError("Unexpected daily-history format.")
    if stoxx:
        frame = frame.loc[frame["SYMBOL"].astype(str).str.strip().eq("V2TX")]
    dates = pd.to_datetime(frame["DATE"], format="%d.%m.%Y" if stoxx else "%m/%d/%Y", errors="coerce")
    values = pd.to_numeric(frame["INDEXVALUE" if stoxx else "CLOSE"], errors="coerce")
    valid = dates.notna() & np.isfinite(values) & values.gt(0)
    series = pd.Series(values[valid].to_numpy(), index=pd.DatetimeIndex(dates[valid]), name=index)
    series = series.loc[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise ValueError("No valid daily closes.")
    return series


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def load_history(index):
    provider, url = SOURCES[index]
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=10) as response:
                return parse_history(index, response.read().decode("utf-8-sig"))
        except (OSError, ValueError) as exc:
            retryable = (exc.code in {408, 429, 500, 502, 503, 504}
                         if isinstance(exc, HTTPError) else isinstance(exc, (URLError, TimeoutError)))
            if attempt or not retryable:
                raise ValueError(f"{provider} daily history unavailable ({exc}).") from exc


@st.cache_data(ttl=55, max_entries=1, show_spinner=False)
def load_vix_intraday():
    return fetch_intraday("^VIX")


def vix_snapshot(response, now, confirmed_date=None):
    """Latest usable spot VIX observation with its original provider timestamp."""
    bars, meta, error = response
    zone = "America/New_York"
    session = now.tz_convert(zone).tz_localize(None).normalize()
    candidates = []
    market_time = aware_timestamp(meta.get("regularMarketTime"))
    market_price = positive(meta.get("regularMarketPrice"))
    if market_time is not None and market_price is not None:
        candidates.append((market_time, market_price))
    if isinstance(bars.index, pd.DatetimeIndex) and bars.index.tz is not None and "Close" in bars:
        bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index(ascending=False)
        for stamp, raw in bars["Close"].items():
            price = positive(raw)
            if pd.notna(stamp) and stamp <= now and price is not None:
                candidates.append((stamp, price))
                break
    candidates = [(stamp, price) for stamp, price in candidates
                  if stamp <= now and stamp.tz_convert(zone).date() == session.date()]
    if confirmed_date is not None and confirmed_date >= session:
        return None, "Official daily close published"
    if session.dayofweek >= 5 or not candidates:
        return None, error or "No current-session quotation"
    stamp, price = max(candidates, key=lambda quote: quote[0])
    age = now - stamp
    if age > MAX_AGE:
        return None, "Intraday quotation is stale"
    delay = positive(meta.get("exchangeDataDelayedBy"))
    delay_label = f"provider delay {delay:g} min" if delay else "feed delay unverified"
    return (stamp, price), f"{int(age.total_seconds() // 60)} min old · {delay_label}"
