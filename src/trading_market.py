"""Yahoo snapshots for the Trading Signals monitor."""
import re
from dataclasses import dataclass

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yfinance as yf


CALENDARS = {
    "MIL": "XMIL", "NMS": "XNYS", "NGM": "XNYS", "NCM": "XNYS",
    "NYQ": "XNYS", "ASE": "XNYS", "PCX": "XNYS", "BTS": "XNYS", "BATS": "XNYS",
    "LSE": "XLON", "PAR": "XPAR", "AMS": "XAMS", "GER": "XETR",
    "EBS": "XSWX", "MCE": "XMAD", "TOR": "XTSE", "HKG": "XHKG",
    "JPX": "XTKS", "ASX": "XASX", "CCY": "24/5",
}
ACTION_COLUMNS = ("Stock Splits", "Dividends", "Capital Gains")
CLOSE_GRACE = pd.Timedelta(minutes=20)
MAX_AGE = pd.Timedelta(minutes=30)


@dataclass(frozen=True)
class Quote:
    ticker: str
    currency: str | None
    timezone: str | None
    market: str
    status: str
    price: float | None = None
    timestamp: pd.Timestamp | None = None
    session: pd.Timestamp | None = None
    previous_close: float | None = None
    detail: str = ""
    adjusted_price: float | None = None
    provider: str = "Yahoo Finance via yfinance"


def positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def aware_timestamp(value):
    try:
        stamp = pd.to_datetime(value, unit="s", utc=True) if isinstance(value, (int, float)) else pd.Timestamp(value)
        return stamp.tz_convert("UTC") if pd.notna(stamp) and stamp.tzinfo else None
    except (TypeError, ValueError, OverflowError):
        return None


def exchange_session(meta, now):
    """Regular session bounds and latest session whose daily bar could be final."""
    zone = meta.get("exchangeTimezoneName")
    if not zone:
        return "UNKNOWN", None, None, None
    try:
        day = now.tz_convert(zone).tz_localize(None).normalize()
        calendar = xcals.get_calendar(CALENDARS[meta["exchangeName"]])
        if not calendar.is_session(day):
            return "CLOSED", None, None, calendar.date_to_session(day, direction="previous")
        opening, closing = calendar.session_open(day), calendar.session_close(day)
        if now < opening:
            market = "PRE-MARKET"
        elif now >= closing:
            market = "CLOSED"
        elif calendar.is_open_on_minute(now.floor("min")):
            market = "OPEN"
        else:
            market = "SESSION BREAK"
        expected = day if now >= closing + CLOSE_GRACE else calendar.previous_session(day)
        return market, opening, closing, expected
    except (KeyError, ValueError, TypeError):
        regular = (meta.get("currentTradingPeriod") or {}).get("regular", {})
        opening = aware_timestamp(regular.get("start"))
        closing = aware_timestamp(regular.get("end"))
        if opening is not None and closing is not None and opening <= now < closing:
            try:
                local_open = opening.tz_convert(zone)
                if local_open.date() == now.tz_convert(zone).date() and local_open.weekday() < 5:
                    return "OPEN · PROVIDER SCHEDULE", opening, closing, None
            except (KeyError, ValueError):
                pass
        return "UNKNOWN", None, None, None


def confirmed_history(history, meta, now):
    """Keep prior daily bars and same-day bars only after regular-session close."""
    zone = meta["exchangeTimezoneName"]
    local_today = now.tz_convert(zone).tz_localize(None).normalize()
    daily = history.copy()
    daily.index = daily.index.tz_convert(zone).tz_localize(None).normalize()
    daily = daily.loc[~daily.index.duplicated(keep="last")].sort_index()
    _, _, _, expected = exchange_session(meta, now)
    complete = daily.index < local_today
    if expected == local_today:
        complete |= daily.index == local_today
    valid = daily["Close"].map(positive).notna()
    calendar_name = CALENDARS.get(meta.get("exchangeName"))
    if calendar_name and len(daily):
        calendar = xcals.get_calendar(calendar_name)
        valid &= daily.index.isin(calendar.sessions_in_range(daily.index[0], daily.index[-1]))
    elif calendar_name is None:
        valid &= daily.index.dayofweek < 5
    # Yahoo historical closes are split-normalized. Undo a still-unconfirmed
    # split so the completed history and stored quantities share the old basis.
    pending = daily.loc[(daily.index == local_today) & ~complete]
    factor = pending["Stock Splits"].replace(0, 1).prod()
    result = daily.loc[complete & valid].copy()
    if factor != 1 and positive(factor):
        result[["Close", "Adj Close", "Dividends"]] *= factor
    result.attrs["market"] = {k: meta.get(k) for k in (
        "currency", "exchangeTimezoneName", "exchangeName",
    )}
    result.attrs["pending_action"] = has_action(pending)
    return result


def has_action(bars):
    return any(bars[key].fillna(0).ne(0).any() for key in ACTION_COLUMNS if key in bars)


def fetch_intraday(ticker):
    """Cache the provider response separately. Failed symbols stay independent."""
    if not re.fullmatch(r"[A-Z0-9.^=/-]{1,30}", ticker):
        return pd.DataFrame(), {}, "Invalid symbol"
    try:
        instrument = yf.Ticker(ticker)
        bars = instrument.history(period="1d", interval="1m", prepost=False,
                                  auto_adjust=False, actions=True, raise_errors=True)
        meta = instrument.history_metadata
        # yfinance 1.7 exposes a lazy metadata Mapping with provider/session
        # internals. Copy only the chart fields needed by the serializable cache.
        return bars, {k: meta.get(k) for k in (
            "currency", "exchangeName", "exchangeTimezoneName", "regularMarketTime",
            "regularMarketPrice", "currentTradingPeriod", "exchangeDataDelayedBy",
        )}, ""
    except Exception as exc:
        return pd.DataFrame(), {}, f"Intraday feed unavailable ({type(exc).__name__})"


def select_quote(ticker, daily, response, now):
    bars, live_meta, error = response
    historic_meta = daily.attrs.get("market", {})
    meta = {**historic_meta, **{k: v for k, v in live_meta.items() if v is not None}}
    zone = meta.get("exchangeTimezoneName")
    currency = historic_meta.get("currency") or meta.get("currency")
    market, opening, closing, expected = exchange_session(meta, now)
    close = daily["Close"].dropna()
    previous = positive(close.iloc[-1]) if len(close) else None
    session = close.index[-1] if len(close) else None
    common = (ticker, currency, zone, market)
    same_session_close = closing is not None and expected == session and session == (
        closing.tz_convert(zone).tz_localize(None).normalize()
    )
    if same_session_close:
        return Quote(*common, "OFFICIAL_CLOSE", previous, closing, session,
                     positive(close.iloc[-2]) if len(close) > 1 else None,
                     "Confirmed daily bar", positive(daily["Adj Close"].iloc[-1]))

    if opening is not None and now >= opening and currency:
        if live_meta.get("currency") != currency:
            error = error or "Provider currency unavailable or inconsistent"
        elif historic_meta.get("exchangeTimezoneName") not in (None, zone):
            error = "Provider exchange timezone changed"
        else:
            valid = pd.DataFrame()
            if isinstance(bars.index, pd.DatetimeIndex) and bars.index.tz is not None and "Close" in bars:
                bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
                valid = bars.loc[(bars.index >= opening) & (bars.index < closing) & (bars.index <= now)]
            action = has_action(valid) or daily.attrs.get("pending_action", False)
            if not valid.empty:
                valid = valid.loc[valid["Close"].map(positive).notna()]
            stamp, raw, adjusted = None, None, None
            label = "1-minute bar"
            if not valid.empty:
                stamp = valid.index[-1].tz_convert("UTC")
                raw = float(valid["Close"].iloc[-1])
                adjusted = positive(valid["Adj Close"].iloc[-1]) if "Adj Close" in valid else None
            trade_time = aware_timestamp(live_meta.get("regularMarketTime"))
            trade_price = positive(live_meta.get("regularMarketPrice"))
            if (trade_price is not None and trade_time is not None and
                    opening <= trade_time <= min(now, closing) and (stamp is None or trade_time > stamp)):
                stamp, raw, adjusted, label = trade_time, trade_price, None, "Regular-market quote"
            if action:
                error = "Corporate action awaiting completed daily reconciliation"
            elif stamp is not None and now - stamp <= MAX_AGE and now < closing + CLOSE_GRACE:
                if adjusted is None and previous:
                    last_adjusted = positive(daily["Adj Close"].iloc[-1])
                    if last_adjusted is not None:
                        adjusted = raw * last_adjusted / previous
                day = stamp.tz_convert(zone).tz_localize(None).normalize()
                age = int((now - stamp).total_seconds() // 60)
                delay = positive(live_meta.get("exchangeDataDelayedBy"))
                freshness = f"provider delay {delay:g} min" if delay else "feed delay unverified"
                return Quote(*common, "INTRADAY", raw, stamp, day, previous,
                             f"{label} · {age} min old · {freshness}", adjusted)
            elif stamp is not None:
                error = "Latest intraday bar is stale or the daily close is pending"

    if previous is None:
        return Quote(*common, "UNAVAILABLE", detail=error or "No valid quotation or daily close")
    fresh = session == expected and market in {"CLOSED", "PRE-MARKET"} and (
        opening is None or now < opening
    )
    close_stamp = None
    try:
        close_stamp = xcals.get_calendar(CALENDARS[meta["exchangeName"]]).session_close(session)
    except (KeyError, ValueError):
        pass
    return Quote(*common, "OFFICIAL_CLOSE" if fresh else "STALE", previous, close_stamp, session,
                 positive(close.iloc[-2]) if len(close) > 1 else None,
                 ("Latest completed session" if fresh else f"{error or 'No current bar'} · prior daily close"),
                 positive(daily["Adj Close"].loc[session]))


def fx_symbol(source, target):
    # Yahoo uses a short alias for USD-base pairs in some listings. Request
    # the fully qualified pair so the multiplication direction remains clear.
    return f"{source}{target}=X"


def snapshot_values(quantities, cash, quotes, base_currency, fx_quotes):
    marks, rates, missing = {}, {}, []
    statuses = set()
    for ticker, qty in quantities.items():
        if not qty:
            continue
        quote = quotes[ticker]
        fx = fx_quotes.get(quote.currency)
        rate = None
        if quote.currency == base_currency:
            rate = 1.
        elif fx and fx.currency == base_currency and fx.status != "UNAVAILABLE":
            rate = positive(fx.price)
        rates[ticker] = rate
        statuses.add(quote.status)
        if fx:
            statuses.add(fx.status)
        if rate is None or quote.status == "UNAVAILABLE" or positive(quote.price) is None:
            missing.append(ticker)
        else:
            marks[ticker] = qty * quote.price * rate
    status = ("UNAVAILABLE" if missing else "STALE" if "STALE" in statuses else
              "PROVISIONAL" if "INTRADAY" in statuses else "CONFIRMED")
    return (None if missing else cash + sum(marks.values())), pd.Series(marks, dtype=float), rates, status


def temporary_series(series, session, value):
    """Return a copy with at most one mark per session and no retroactive edits."""
    if not len(series) or session is None or session < series.index[-1] or positive(value) is None:
        return series.copy()
    result = series.copy()
    result.loc[session] = value
    return result


def current_equity(equity, nav, value, intraday_peak=1.):
    """Link the live mark to flow-adjusted daily equity and its confirmed peak."""
    level = equity.iloc[-1] * value / nav
    return level, level / max(1., equity.max(), level, intraday_peak) - 1
