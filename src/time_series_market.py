"""Cached completed Yahoo bars and reporting-currency prices for basket analysis."""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from src.fama_french import load_europe_ff3, load_us_ff3
from src.fama_french_5 import load_europe_ff5, load_us_ff5
from src.time_series_portfolio import prices_in_currency
from src.trading_market import CALENDARS, exchange_session, fx_symbol, positive

OHLC_FIELDS = ("Open", "High", "Low", "Close")


@st.cache_data(ttl=1800, max_entries=128, show_spinner=False)
def daily_bars(ticker):
    instrument = yf.Ticker(ticker)
    try:
        history = instrument.history(period="max", interval="1d", auto_adjust=False,
                                     actions=True, raise_errors=True, timeout=12)
        meta = instrument.history_metadata
    except Exception as exc:
        raise ValueError(f"{ticker}: dati Yahoo non disponibili ({type(exc).__name__}).") from exc
    zone, currency = meta.get("exchangeTimezoneName"), meta.get("currency")
    if history.empty or not currency or not zone or history.index.tz is None:
        raise ValueError(f"{ticker}: storico, valuta o fuso orario non disponibili.")
    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert(zone).tz_localize(None).normalize()
    history.index = history.index.tz_convert(zone).tz_localize(None).normalize()
    history = history.loc[~history.index.duplicated(keep="last")].sort_index()
    _, _, _, last_complete = exchange_session(meta, now)
    include = (history.index < today) | ((history.index == today) & (last_complete == today))
    pending = history.loc[(history.index == today) & ~include, "Stock Splits"]
    split = pending.replace(0, 1).prod()
    daily = history.loc[include].reindex(columns=[*OHLC_FIELDS, "Adj Close"])
    if split != 1 and positive(split):
        daily = daily * split
    if currency in {"GBp", "GBX"}:
        daily, currency = daily / 100, "GBP"
    daily = daily.where(np.isfinite(daily) & daily.gt(0))
    if daily.Close.dropna().empty:
        raise ValueError(f"{ticker}: nessuna chiusura completa valida.")
    daily.attrs["quote"] = {"symbol": meta.get("symbol") or ticker,
                            "name": meta.get("longName") or meta.get("shortName") or ticker,
                            "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
                            "timezone": zone, "calendar": CALENDARS.get(meta.get("exchangeName")), "original_currency": meta.get("currency"),
                            "type": meta.get("instrumentType"),
                            "last_native_close": daily.Close.last_valid_index().isoformat()}
    return daily, currency


def conversion_pair(currency, reporting, *, allow_inverse=False):
    inverse = allow_inverse and (currency, reporting) == ("USD", "EUR")
    symbol = fx_symbol(reporting, currency) if inverse else fx_symbol(currency, reporting)
    try:
        fx, quote_currency = daily_bars(symbol)
    except ValueError:
        if not allow_inverse:
            raise
        inverse = not inverse
        symbol = fx_symbol(reporting, currency) if inverse else fx_symbol(currency, reporting)
        fx, quote_currency = daily_bars(symbol)
    if quote_currency != (currency if inverse else reporting):
        raise ValueError(f"{symbol}: valuta del cambio incoerente.")
    cross = 1 / fx.Close if inverse else fx.Close
    return cross, symbol, inverse


def reporting_history(ticker, reporting, *, prior_fx=False):
    native, currency = daily_bars(ticker)
    if currency == reporting:
        return native, native, currency, None
    cross, symbol, inverse = conversion_pair(currency, reporting, allow_inverse=prior_fx)
    rate = prices_in_currency(pd.Series(1., index=native.index), cross, prior_day=prior_fx)
    converted = native.mul(rate, axis=0)
    last = rate.last_valid_index()
    lag = pd.Timedelta(days=int(prior_fx))
    observed = cross.dropna().loc[:last-lag].last_valid_index() if last is not None else None
    converted.attrs["fx"] = {"last_observation_date": observed.isoformat() if observed is not None else None,
                             "ticker": symbol, "direction": "divide" if inverse else "multiply",
                             "cutoff": "previous date" if prior_fx else "valuation date"}
    return converted, native, currency, rate


@st.cache_data(ttl=86400, max_entries=6, show_spinner=False)
def load_factors(model, market):
    if market == "Developed":
        root = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        parser, family = (load_europe_ff3, 3) if model == "FF3" else (load_europe_ff5, 5)
        return parser(url=f"{root}Developed_{family}_Factors_Daily_CSV.zip")
    return {("FF3", "USA"): load_us_ff3, ("FF5", "USA"): load_us_ff5,
            ("FF3", "Borsa Italiana"): load_europe_ff3,
            ("FF5", "Borsa Italiana"): load_europe_ff5}[model, market]()
