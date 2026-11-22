"""Coverage decisions on the observed market calendar, prior to FFR."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

STRICT = "Strict Historical Coverage"
ADAPTIVE = "Adaptive Calibration Window"
MANUAL = "Manual Universe Selection"
MIN_SESSIONS = 101  # 80 usable observations after trailing state warm-up


def download_history(tickers, benchmark, market):
    from src.data import _INDEX_ALIASES, _fetch_one

    zone = "America/New_York" if market == "USA" else "Europe/Rome"
    today = pd.Timestamp.now(tz=zone).date()

    def fetch(ticker):
        errors = []
        for symbol in _INDEX_ALIASES.get(ticker, (ticker,)):
            data, attempts = _fetch_one(symbol, "max", preserve_invalid=True)
            if data is not None:
                dates = pd.DatetimeIndex(data.index).tz_localize(None).normalize()
                data = data.set_axis(dates).sort_index()
                return data.loc[data.index.date < today], None
            errors.extend(attempts)
        return pd.Series(dtype=float), "; ".join(errors)

    with ThreadPoolExecutor(max_workers=4) as workers:
        fetched = dict(zip(tickers, workers.map(fetch, tickers)))
    calendar_data, calendar_error = fetch(benchmark)
    if calendar_error is not None or calendar_data.empty:
        raise ValueError(f"Reference calendar unavailable for {benchmark}: {calendar_error}")
    if not calendar_data.index.is_unique:
        raise ValueError(f"Reference calendar {benchmark} contains duplicate sessions.")
    return {t: v[0] for t, v in fetched.items()}, calendar_data.index, {
        t: v[1] for t, v in fetched.items() if v[1] is not None
    }


def assess_coverage(series, calendar, sessions, policy=STRICT, minimum=MIN_SESSIONS,
                    selected=None, download_errors=None):
    """Require a contiguous terminal window on the supplied reference calendar."""
    if (not isinstance(calendar, pd.DatetimeIndex) or not calendar.is_unique
            or not calendar.is_monotonic_increasing or calendar.empty):
        raise ValueError("Reference calendar must contain unique, chronological sessions.")
    if policy not in (STRICT, ADAPTIVE, MANUAL) or not MIN_SESSIONS <= minimum <= sessions:
        raise ValueError("Invalid calibration policy or window length.")
    download_errors = download_errors or {}
    selected = set(series if selected is None else selected)
    window = calendar[-(sessions + 1):]
    requested_prices = sessions + 1
    columns, rows, suffix = {}, [], {}
    for ticker, observations in series.items():
        dates = observations.index
        dates_ok = isinstance(dates, pd.DatetimeIndex)
        if not observations.empty and not dates_ok:
            raise ValueError(f"Invalid date index for {ticker}.")
        observations = observations.loc[dates <= window[-1]] if not observations.empty else observations
        raw = pd.to_numeric(observations, errors="coerce")
        good = raw[np.isfinite(raw) & raw.gt(0)]
        first, last = (good.index.min(), good.index.max()) if not good.empty else (pd.NaT, pd.NaT)
        duplicate = not raw.loc[raw.index >= window[0]].index.is_unique if not raw.empty else False
        aligned = raw[~raw.index.duplicated(keep=False)].reindex(window)
        valid = np.isfinite(aligned) & aligned.gt(0)
        invalid = int((aligned.notna() & ~valid).sum())
        interior = int((aligned.isna() & (window >= first) & (window <= last)).sum()) if pd.notna(first) else 0
        off_calendar = int(((raw.index >= window[0]) & ~raw.index.isin(window)).sum()) if not raw.empty else 0
        suffix[ticker] = len(window) - (np.flatnonzero(~valid)[-1] + 1 if not valid.all() else 0)
        reasons = []
        if ticker in download_errors:
            reasons.append("Download failed")
        if duplicate:
            reasons.append("Duplicate dates")
        if pd.isna(first) or first > window[0] or len(window) < requested_prices:
            reasons.append("Insufficient history")
        if pd.notna(last) and last < window[-1]:
            reasons.append("Latest sessions missing")
        if interior:
            reasons.append("Internal missing prices")
        if invalid:
            reasons.append("Nonpositive or nonfinite prices")
        if off_calendar:
            reasons.append("Off-calendar observations")
        usable = ticker not in download_errors and not duplicate and not off_calendar
        columns[ticker] = (aligned, usable)
        rows.append({"Asset": ticker, "First available": first, "Last available": last,
                     "Valid prices": int(valid.sum()), "Missing prices": requested_prices-int(valid.sum())-invalid,
                     "Invalid prices": invalid, "Coverage": float(valid.sum()/requested_prices),
                     "Off-calendar observations": off_calendar, "Selected": ticker in selected,
                     "Reason": "; ".join(reasons), "Download detail": download_errors.get(ticker, "")})
    count = requested_prices
    if policy == ADAPTIVE:
        candidates = [t for t in columns if t in selected and columns[t][1]]
        count = min((suffix[t] for t in candidates), default=0)
        if count < minimum + 1:
            count = 0
    effective = window[-count:] if count else window[:0]
    accepted = {}
    for row in rows:
        ticker = row["Asset"]
        values, usable = columns[ticker]
        eligible = ticker in selected and usable and count > 0 and suffix[ticker] >= count
        row["Eligible"] = eligible
        row["Effective start"] = effective[0] if eligible else pd.NaT
        row["Effective end"] = effective[-1] if eligible else pd.NaT
        row["Effective return sessions"] = len(effective) - 1 if eligible else 0
        if ticker not in selected:
            row["Reason"] = "Not selected" + ("; " + row["Reason"] if row["Reason"] else "")
        elif policy == ADAPTIVE and not count:
            row["Reason"] = (row["Reason"] + "; " if row["Reason"] else "") + "Common window below minimum"
        if eligible:
            accepted[ticker] = values.loc[effective]
    report = pd.DataFrame(rows)
    prices = pd.DataFrame(accepted, index=effective) if accepted else pd.DataFrame()
    return prices, report
