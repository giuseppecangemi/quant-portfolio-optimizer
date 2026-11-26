"""Investible benchmark quotes, currency conversion and matched OOS evaluation."""
from dataclasses import dataclass

import exchange_calendars
import numpy as np
import pandas as pd
import yfinance as yf

from src.backtest import calculate_backtest_metrics
from src.backtest_inputs import benchmark_curve

LEGACY_BENCHMARK = "Benchmark"


@dataclass(frozen=True)
class ETF:
    ticker: str
    isin: str
    venue: str
    calendar: str
    quote_currency: str
    fund_currency: str
    income: str
    inception: str
    listing: str


ETFS = {
    "iShares FTSE MIB ETF": ETF(
        "IMIB.MI", "IE00B1XNH568", "Borsa Italiana", "XMIL",
        "EUR", "EUR", "Distributing", "2007-07-06", "2008-03-18",
    ),
    "iShares MSCI ACWI ETF": ETF(
        "IUSQ.DE", "IE00B6R52259", "Xetra", "XETR",
        "EUR", "USD", "Accumulating", "2011-10-21", "2012-04-02",
    ),
}


def benchmark_label(name, market):
    if name == LEGACY_BENCHMARK:
        return "S&P 500 Index (Existing)" if market == "USA" else "FTSE MIB Index (Existing)"
    return name


def adjusted_close(frame, symbol, *, listing=None):
    dates = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    if "Adj Close" not in frame:
        raise ValueError(f"{symbol}: adjusted closes unavailable")
    series = pd.to_numeric(frame["Adj Close"], errors="coerce")
    series.index = dates
    series = series.sort_index()
    series = series.loc[~series.index.duplicated(keep="last")]
    raw = frame["Adj Close"].copy()
    raw.index = dates
    raw = raw.loc[~raw.index.duplicated(keep="last")].sort_index()
    classification = {
        day.isoformat(): "MISSING_ADJ_CLOSE" if pd.isna(raw.loc[day]) else "INVALID_ADJ_CLOSE"
        for day in series.index[~np.isfinite(series) | series.le(0)]
    }
    series = series.loc[np.isfinite(series) & series.gt(0)]
    if listing is not None:
        series = series.loc[listing:]
    if series.empty and not classification:
        raise ValueError(f"{symbol}: no valid adjusted closes")
    series.attrs["row_issues"] = classification
    return series.rename(symbol)


def download_adjusted(symbol, currency, start, end, *, listing=None):
    quote = yf.Ticker(symbol)
    history = quote.history(start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                            auto_adjust=False, actions=True, timeout=15)
    if not history.empty and "Adj Close" not in history:
        close = pd.Series(dtype=float, index=pd.DatetimeIndex([]), name=symbol)
        close.attrs["row_issues"] = {day.isoformat(): "MISSING_ADJ_CLOSE"
                                    for day in pd.DatetimeIndex(history.index).tz_localize(None).normalize()}
    else:
        close = adjusted_close(history, symbol, listing=listing)
    actual_currency = quote.history_metadata.get("currency")
    if actual_currency != currency:
        raise ValueError(f"{symbol}: expected {currency} quotes, received {actual_currency or 'unknown'}")
    close.attrs.update(provider="Yahoo Finance", requested=(str(start.date()), str(end.date())))
    return close


def load_etf_data(selected, start, end, target_currency, *, fetch=download_adjusted):
    prices, errors = {}, {}
    fx = None
    fx_error = None
    if target_currency == "USD" and any(name in ETFS for name in selected):
        try:
            fx = fetch("EURUSD=X", "USD", start, end)
        except Exception as exc:
            fx_error = str(exc)
    for name in dict.fromkeys(selected):
        if name not in ETFS:
            continue
        etf = ETFS[name]
        try:
            if fx_error:
                raise ValueError(f"EUR/USD unavailable: {fx_error}")
            prices[name] = fetch(etf.ticker, etf.quote_currency, start, end, listing=etf.listing)
        except Exception as exc:
            errors[name] = str(exc)
    return prices, fx, errors


def joint_sessions(etf, dates):
    start, end = dates[0], dates[-1]
    calendar = exchange_calendars.get_calendar(etf.calendar, start=start - pd.Timedelta(days=10),
                                               end=end + pd.Timedelta(days=10))
    return dates.intersection(calendar.sessions_in_range(start, end))


def etf_evaluation(etf, close, dates, capital, risk_free_rate, target_currency, fx=None):
    """Evaluate on joint sessions, rejecting missing marks on scheduled common dates."""
    start, end = dates[0], dates[-1]
    if start < pd.Timestamp(etf.listing):
        raise ValueError(f"{etf.ticker} listed {etf.listing}, after OOS start {start:%Y-%m-%d}")
    common = joint_sessions(etf, dates)
    if len(common) < 2:
        raise ValueError("Fewer than two joint trading sessions")
    marks = close.reindex(common)
    missing = ~np.isfinite(marks) | marks.le(0)
    if missing.any():
        raise ValueError(f"{etf.ticker}: missing adjusted close on scheduled session {missing[missing].index[0]:%Y-%m-%d}")
    if target_currency != etf.quote_currency:
        if (etf.quote_currency, target_currency) != ("EUR", "USD") or fx is None:
            raise ValueError("Required EUR/USD conversion data unavailable")
        cross = fx.reindex(common)
        missing = ~np.isfinite(cross) | cross.le(0)
        if missing.any():
            raise ValueError(f"EUR/USD missing on {missing[missing].index[0]:%Y-%m-%d}")
        marks = marks * cross
    curve = benchmark_curve(marks.to_frame(), common[0], common[-1], capital, required_dates=common)
    return {"curve": curve, "metrics": calculate_backtest_metrics(curve, risk_free_rate),
            "currency": target_currency, "ticker": etf.ticker, "quote_currency": etf.quote_currency,
            "history_start": close.index[0].date().isoformat(), "history_end": close.index[-1].date().isoformat()}


def matched_portfolio_metrics(curve, dates, risk_free_rate, *, gross=False):
    if not dates.isin(curve.index).all():
        raise ValueError("Portfolio does not cover all matched benchmark dates")
    column = "Gross Portfolio Value" if gross and "Gross Portfolio Value" in curve else "Portfolio Value"
    nav = curve.loc[dates, column]
    matched = pd.DataFrame({"Portfolio Value": nav, "Daily Return": nav.pct_change().fillna(0.)})
    if dates[0] == curve.index[0] and "initial_capital" in curve.attrs:
        matched.attrs["initial_capital"] = curve.attrs["initial_capital"]
    return calculate_backtest_metrics(matched, risk_free_rate)


def attach_etf_benchmarks(result, selected, market, risk_free_rate, quotes, fx, errors):
    """Attach optional benchmark outputs after portfolio execution."""
    dates = result["max_sharpe_bt"].index
    currency = result.get("portfolio_currency", "USD" if market == "USA" else "EUR")
    runs, reports, diagnostics = {}, [], []
    failures = dict(errors)
    if LEGACY_BENCHMARK in selected and "ftse_mib_bt" in result:
        reports.append({"Benchmark": LEGACY_BENCHMARK,
                        **index_coverage(result["ftse_mib_bt"], dates, market)})
    for name in selected:
        if name not in ETFS:
            continue
        run, report, detail = etf_coverage(
            ETFS[name], quotes.get(name), dates, result["capital"], risk_free_rate, currency,
            fx, error=errors.get(name),
        )
        reports.append({"Benchmark": name, **report})
        diagnostics.extend(detail)
        if run is None:
            failures[name] = report["Reason"]
        else:
            runs[name] = run
    coverage = {name: (series.index[0].date().isoformat(), series.index[-1].date().isoformat())
                for name, series in quotes.items() if not series.empty}
    result.update(selected_benchmarks=list(selected), etf_benchmarks=runs, benchmark_quote_coverage=coverage,
                  benchmark_failures=failures, benchmark_market=market,
                  coverage_reports=reports, benchmark_diagnostics=diagnostics)


def robustness_comparison(results, portfolios, benchmark_name, basis, matched):
    """One explicitly chosen benchmark; use identical daily observations for both legs."""
    available = results.loc[results["Model"].isin([*portfolios, benchmark_name])].copy()
    fields = ("Total Return", "Sharpe Ratio", "Max Drawdown")
    if basis == "Net":
        for field in fields:
            available[field] = available[f"Net {field}"]
    if benchmark_name in ETFS:
        for index, row in available.iterrows():
            model = row["Model"]
            if model == benchmark_name:
                continue
            metrics = matched.get(row["OOS Start"], {}).get(model, {}).get(basis)
            for field in fields:
                available.at[index, field] = metrics[field] if metrics else np.nan
            for field in ("Tracking Error", "Information Ratio"):
                available.at[index, field] = metrics.get(field, np.nan) if metrics else np.nan
    available = available.dropna(subset=list(fields))
    complete = available.groupby("OOS Start")["Model"].agg(
        lambda names: set([*portfolios, benchmark_name]).issubset(names)
    )
    return available.loc[available["OOS Start"].isin(complete[complete].index)].copy(), complete[complete].index


def retry_missing(etf, close, sessions, fetch):
    """One bounded pass over at most three gaps, with adjacent adjusted anchors."""
    missing = ~sessions.isin(close.index)
    recovered, attempts = set(), {}
    anchor_quotes = close.reindex(sessions).dropna()
    merged = close.copy()
    gaps = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    for first in gaps[:3]:
        end = next((i for i in range(first, len(sessions)) if not missing[i]), len(sessions))
        wanted = sessions[first:end]
        adjacent = anchor_quotes.index
        left, right = adjacent[adjacent < wanted[0]][-1:], adjacent[adjacent > wanted[-1]][:1]
        anchors = left.union(right)
        if anchors.empty:
            continue
        if len(wanted) > 20:
            if len(left):
                wanted, anchors = wanted[:20], left
            else:
                wanted, anchors = wanted[-20:], right
        requested = wanted.union(anchors)
        for day in wanted:
            attempts[day] = (str(requested[0].date()), str(requested[-1].date()))
        try:
            retry = fetch(etf.ticker, etf.quote_currency, requested[0], requested[-1], listing=etf.listing)
        except Exception:
            continue
        observed = anchors.intersection(retry.index)
        if not len(observed) or not np.allclose(retry.loc[observed], close.loc[observed], rtol=1e-5, atol=1e-8):
            continue
        valid = retry.reindex(wanted).dropna()
        valid = valid.loc[np.isfinite(valid) & valid.gt(0)]
        merged = pd.concat([merged, valid])
        recovered.update(valid.index)
    merged.attrs = close.attrs.copy()
    return merged.sort_index(), recovered, attempts


def impute_isolated_etf_gaps(etf, close, joint, start, end, recovered):
    """Use observed immediate exchange-session neighbours within the OOS."""
    if len(joint) < 2:
        return {}
    calendar = exchange_calendars.get_calendar(
        etf.calendar, start=start - pd.Timedelta(days=10), end=end + pd.Timedelta(days=10),
    )
    sessions = calendar.sessions_in_range(start, end)
    marks = close.reindex(sessions)
    valid = np.isfinite(marks) & marks.gt(0)
    missing = marks.isna() & valid.shift(1, fill_value=False) & valid.shift(-1, fill_value=False)
    issues = close.attrs.get("row_issues", {})
    source = close.attrs.get("provider", f"{etf.ticker} adjusted series")
    imputations = {}
    for day in marks.index[missing].intersection(joint):
        if issues.get(day.isoformat()) == "INVALID_ADJ_CLOSE" or day < pd.Timestamp(etf.listing):
            continue
        previous, following = calendar.previous_session(day), calendar.next_session(day)
        left, right = float(marks.loc[previous]), float(marks.loc[following])
        imputations[day] = {
            "Previous Date": str(previous.date()), "Previous Adj Close": left,
            "Previous Source": f"{source} retry" if previous in recovered else source,
            "Next Date": str(following.date()), "Next Adj Close": right,
            "Next Source": f"{source} retry" if following in recovered else source,
            "Imputed Adj Close": (left + right) / 2,
            "Quote Currency": etf.quote_currency, "Method": "ADJACENT_PRICE_MEAN",
        }
    return imputations


def benchmark_run_label(name, run):
    return f"{name} (imputed)" if run.get("status") == "COMPLETE_WITH_IMPUTATION" else name


def etf_coverage(etf, close, dates, capital, rate, currency, fx=None, *,
                 requested=None, error=None, fetch=download_adjusted):
    """Return an independent coverage record plus the strict run, when available."""
    start, end = dates[0], dates[-1]
    close = pd.Series(dtype=float, index=pd.DatetimeIndex([])) if close is None else close
    requested = close.attrs.get("requested", requested or (str(start.date()), str(end.date())))
    issues = close.attrs.get("row_issues", {})
    joint = joint_sessions(etf, dates)
    report = {"Ticker": etf.ticker, "Market": etf.venue, "Currency": currency,
              "OOS Start": str(start.date()), "OOS End": str(end.date()),
              "Requested Start": requested[0], "Requested End": requested[1],
              "Status": "UNAVAILABLE", "Benchmark Return": None,
              "From": None, "Through": None, "Reason": error}
    # Include only actual source observations for outside-OOS audit, never for return calculation.
    outside = close.index[(close.index < start) | (close.index > end)]
    recovered, retries = set(), {}
    if not error and close.attrs.get("provider") and len(joint) and start >= pd.Timestamp(etf.listing):
        bounded = close.loc[start:end]
        close, recovered, retries = retry_missing(etf, bounded, joint, fetch)
    imputations = impute_isolated_etf_gaps(etf, close, joint, start, end, recovered) if not error else {}
    observed_adjusted_close = close.loc[start:end].copy()
    analysis_adjusted_close = observed_adjusted_close.copy()
    for day, observation in imputations.items():
        analysis_adjusted_close.loc[day] = observation["Imputed Adj Close"]
    if imputations:
        analysis_adjusted_close = analysis_adjusted_close.sort_index()
        report["Imputed Dates"] = ", ".join(str(day.date()) for day in imputations)
    report["Imputed Count"] = len(imputations)
    if not error:
        try:
            run = etf_evaluation(etf, analysis_adjusted_close, dates, capital, rate, currency, fx)
        except ValueError as exc:
            report["Reason"] = str(exc)
            run = None
        else:
            report["Status"] = run["status"] = "COMPLETE_WITH_IMPUTATION" if imputations else "COMPLETE"
            run.update(observed_adjusted_close=observed_adjusted_close,
                       analysis_adjusted_close=analysis_adjusted_close,
                       history_start=close.index[0].date().isoformat(),
                       history_end=close.index[-1].date().isoformat())
    else:
        run = None

    marks = close.reindex(joint)
    marks = marks.where(np.isfinite(marks) & marks.gt(0))
    if currency != etf.quote_currency:
        cross = fx.reindex(joint) if fx is not None else pd.Series(np.nan, index=joint)
        marks *= cross.where(np.isfinite(cross) & cross.gt(0))
    observed = marks.dropna()
    if len(observed) >= 2:
        first, last = observed.index[[0, -1]]
        full = first == start and last == end
        if run is None:
            report["Status"] = "ENDPOINT_ONLY" if full else "PARTIAL"
        report.update({"From": str(first.date()), "Through": str(last.date()),
                       "Benchmark Return": float(observed.iloc[-1] / observed.iloc[0] - 1)})
        if run is not None and not full:
            report["Endpoint Scope"] = "PARTIAL"
        else:
            report["Endpoint Scope"] = "FULL" if full else "PARTIAL"
    diagnostics = []
    for day in dates.union(outside):
        if day < start or day > end:
            status = "OUTSIDE_OOS"
        elif day not in joint:
            status = "MARKET_HOLIDAY"
        elif day in imputations:
            status = "IMPUTED_ADJ_CLOSE"
        elif day not in close.index:
            status = issues.get(day.isoformat(), "MISSING_PROVIDER_ROW")
        elif pd.isna(close.loc[day]):
            status = "MISSING_ADJ_CLOSE"
        elif not np.isfinite(close.loc[day]) or close.loc[day] <= 0:
            status = "INVALID_ADJ_CLOSE"
        else:
            status = "VALID_OBSERVATION"
        retry_start, retry_end = retries.get(day, (None, None))
        diagnostics.append({**{key: report[key] for key in
                                ("Ticker", "Market", "OOS Start", "OOS End", "Requested Start", "Requested End")},
                            "Date": str(day.date()), "Type": status,
                            "Original Type": issues.get(day.isoformat(), "MISSING_PROVIDER_ROW") if day in recovered or status == "IMPUTED_ADJ_CLOSE" else status,
                            "Retry Start": retry_start, "Retry End": retry_end,
                            "Source": "Yahoo Finance retry" if day in recovered else
                            "Benchmark interpolation" if status == "IMPUTED_ADJ_CLOSE" else "Yahoo Finance",
                            **imputations.get(day, {})})
    return run, report, diagnostics


def endpoint_comparison(curve, report, *, gross=False):
    if report["Benchmark Return"] is None:
        return None
    dates = pd.to_datetime([report["From"], report["Through"]])
    if not dates.isin(curve.index).all():
        return None
    column = "Gross Portfolio Value" if gross and "Gross Portfolio Value" in curve else "Portfolio Value"
    first, last = curve.loc[dates, column]
    if dates[0] == curve.index[0]:
        first = curve.attrs.get("initial_capital", first)
    return float(last / first - 1)


def daily_relative_metrics(portfolio, benchmark, dates, *, gross=False):
    column = "Gross Portfolio Value" if gross and "Gross Portfolio Value" in portfolio else "Portfolio Value"
    portfolio_nav = portfolio.loc[dates, column]
    returns = portfolio_nav.pct_change()
    benchmark_returns = benchmark.loc[dates, "Portfolio Value"].pct_change()
    if dates[0] == portfolio.index[0] and "initial_capital" in portfolio.attrs:
        returns.iloc[0] = portfolio_nav.iloc[0] / portfolio.attrs["initial_capital"] - 1
        benchmark_returns.iloc[0] = 0.
    active = (returns - benchmark_returns).dropna()
    tracking = float(active.std() * np.sqrt(252))
    return {"Tracking Error": tracking,
            "Information Ratio": float(active.mean() * 252 / tracking) if tracking > 0 else np.nan}


def index_coverage(frame, dates, market, *, requested=None, ticker=None):
    """Coverage on portfolio dates; leave legacy index presentation unchanged."""
    start, end = dates[0], dates[-1]
    prices = frame.iloc[:, 0] if not frame.empty else pd.Series(dtype=float)
    prices = prices.reindex(dates).dropna()
    prices = prices.loc[np.isfinite(prices) & prices.gt(0)]
    report = {"Ticker": ticker or ("^GSPC" if market == "USA" else "FTSEMIB.MI"),
              "Market": market, "Currency": "USD" if market == "USA" else "EUR",
              "OOS Start": str(start.date()), "OOS End": str(end.date()),
              "Requested Start": requested[0] if requested else None,
              "Requested End": requested[1] if requested else None,
              "Status": "UNAVAILABLE", "Benchmark Return": None,
              "From": None, "Through": None, "Reason": None}
    if len(prices) < 2:
        report["Reason"] = "Fewer than two observed comparable index marks"
        return report
    full = prices.index[0] == start and prices.index[-1] == end
    report.update({"From": str(prices.index[0].date()), "Through": str(prices.index[-1].date()),
                   "Benchmark Return": float(prices.iloc[-1] / prices.iloc[0] - 1),
                   "Endpoint Scope": "FULL" if full else "PARTIAL",
                   "Status": "COMPLETE" if prices.index.equals(dates) else "ENDPOINT_ONLY" if full else "PARTIAL"})
    return report
