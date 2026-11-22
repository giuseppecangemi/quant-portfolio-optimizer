"""Daily close series from the index publishers."""
from io import StringIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st

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
