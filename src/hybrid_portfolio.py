"""Shared inputs for international portfolios and their existing estimators."""
import exchange_calendars as xcals
from io import StringIO

import numpy as np
import pandas as pd

from src.eur_risk_free import ESTR_ONLY
from src.time_series_market import conversion_pair, reporting_history
from src.time_series_portfolio import prices_in_currency

REGIONAL = "Regional Factors"
GLOBAL = "Global Developed Markets Factors"
REGIONS = ("Europe", "USA", "Other developed", "Unmapped")
BENCHMARKS = {"Europe": "^STOXX", "USA": "^GSPC", "Developed": "SWDA.MI"}


def effective_universe(quantitative, manual, market, currency="EUR", policy=REGIONAL, eur_rf_mode=ESTR_ONLY):
    region = "USA" if market == "USA" else "Europe" if market == "Borsa Italiana" else "Unmapped"
    entries = {ticker: dict(ticker=ticker, name=name, exchange=market, region=region,
                            sources=["Quantitative"], minimum=0., inclusion="Optimizer Discretion")
               for ticker, name in quantitative}
    for ticker, item in manual.items():
        sources = [*entries.get(ticker, {}).get("sources", []), "Manual"]
        entries[ticker] = {**item, "ticker": ticker, "sources": sources}
    return {"version": 1, "screening_market": market, "base_currency": currency,
            "factor_policy": policy, "eur_rf_mode": eur_rf_mode, "assets": list(entries.values())}


def load_hybrid_prices(tickers, currency):
    """Return separate execution and estimation frames with quote provenance."""
    from src.data import _INDEX_ALIASES

    def quote(ticker):
        for symbol in _INDEX_ALIASES.get(ticker, (ticker,)):
            try:
                return reporting_history(symbol, currency, prior_fx=True)
            except ValueError as exc:
                last_error = exc
        raise last_error

    loaded = {ticker: quote(ticker) for ticker in tickers}
    close, adjusted, native, provenance = {}, {}, {}, {}
    for ticker, (converted, local, denomination, _) in loaded.items():
        close[ticker], adjusted[ticker], native[ticker] = converted.Close, converted["Adj Close"], local["Adj Close"]
        provenance[ticker] = {**local.attrs.get("quote", {}), "quote_currency": denomination,
                              "base_currency": currency, "fx": converted.attrs.get("fx"),
                              "first_price": local.Close.first_valid_index().date().isoformat(),
                              "last_base_close": converted.Close.last_valid_index()}
    return (pd.DataFrame(close).sort_index(), pd.DataFrame(adjusted).sort_index(),
            pd.DataFrame(native).sort_index(), provenance)


def annotate_allocation(table, capital, config):
    minimum = {item["ticker"]: item["minimum"] for item in config["assets"]}
    result = table.assign(**{"Minimum weight": pd.Series(minimum).reindex(table.index).fillna(0.)})
    result["Effective weight"] = result.Invested / capital
    result["Below minimum"] = result["Effective weight"] + 1e-10 < result["Minimum weight"]
    return result


def historical_panel(native, config, *, execution_history=None):
    """Build the converted historical estimation panel without changing ranks."""
    market = config["screening_market"]
    source = "USD" if market == "USA" else "EUR"
    base = config["base_currency"]
    converted = native
    if source != base:
        cross, _, _ = conversion_pair(source, base, allow_inverse=True)
        rate = prices_in_currency(pd.Series(1., index=native.index), cross, prior_day=True)
        converted = native.mul(rate, axis=0)
    manual_only = [item["ticker"] for item in config["assets"]
                   if "Manual" in item["sources"] and item["ticker"] not in native]
    if manual_only:
        external = (load_hybrid_prices(manual_only, base)[1]
                    if execution_history is None else execution_history)
        converted = pd.concat([converted, external.reindex(native.index)], axis=1)
    return converted


def hybrid_sessions(index, config):
    """A hybrid execution calendar uses only known joint exchange sessions."""
    eligible = pd.Series(True, index=index)
    aliases = {"NASDAQ": "XNYS", "NYSE": "XNYS", "Borsa Italiana": "XMIL"}
    for item in config["assets"]:
        if "Manual" not in item["sources"]:
            continue
        name = item.get("calendar") or aliases.get(item.get("exchange"))
        if not name:
            # Without a verified calendar the execution engine requires quotes.
            continue
        cal = xcals.get_calendar(name)
        active = index >= pd.Timestamp(item["effective_from"])
        if active.any():
            sessions = cal.sessions_in_range(index[active][0], index[-1])
            eligible &= ~active | index.isin(sessions)
    return index[eligible.to_numpy()]


def native_ranking_prices(prices):
    if source := prices.attrs.get("ranking_source"):
        return pd.read_json(StringIO(source), orient="split")
    return None


def manual_records(config):
    return {row["ticker"]: row for row in config["assets"] if "Manual" in row["sources"]}


def eligible_manual(history, selected, manual, *, ranked=False):
    decision = history.index[-1]
    result = list(selected)
    status = {}
    for ticker, item in manual.items():
        mandatory = item["minimum"] > 0
        if decision < pd.Timestamp(item["effective_from"]):
            if mandatory:
                raise ValueError(f"{ticker}: mandatory inclusion not effective on {decision:%Y-%m-%d}.")
            if not ranked and ticker in result:
                result.remove(ticker)
            status[ticker] = "Before eligibility date"
            continue
        observed = history[ticker].dropna() if ticker in history else pd.Series(dtype=float)
        ready = (len(observed) >= int(np.ceil(.8*len(history)))
                 and observed.index[0] <= history.index[0] + pd.Timedelta(days=7)
                 and observed.index[-1] >= decision - pd.Timedelta(days=4))
        if mandatory and not ready:
            raise ValueError(f"{ticker}: missing pre-decision history for mandatory inclusion on {decision:%Y-%m-%d}.")
        if ready and ticker not in result:
            result.append(ticker)
        elif not ready and not ranked and ticker in result:
            result.remove(ticker)
        status[ticker] = (f"Eligible · {len(observed)}/{len(history)} observations" if ready
                          else "Insufficient pre-decision history")
    return result, status
