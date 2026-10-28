"""Automatic SEC XBRL companyfacts acquisition for US historical screening.

SEC filings are filtered by their filing date; no contemporary Yahoo fundamental
ratios are used. Annual net income is used as a *reported annual* earnings proxy,
not mislabeled as TTM. Missing fields are explicit NaNs.
"""
from __future__ import annotations
import os
import time
from functools import lru_cache
import requests
import numpy as np
import pandas as pd

SEC_TICKERS = 'https://www.sec.gov/files/company_tickers.json'
SEC_FACTS = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'

@lru_cache(maxsize=1)
def _session():
    identity = os.getenv('SEC_USER_AGENT', '').strip()
    if not identity or '@' not in identity:
        raise ValueError('Set SEC_USER_AGENT to an identifiable application and contact email (e.g. QuantResearch research@example.com) before SEC downloads.')
    session = requests.Session()
    session.headers.update({'User-Agent': identity, 'Accept-Encoding': 'gzip, deflate', 'Accept': 'application/json'})
    return session

def _json(url):
    for attempt in range(3):
        try:
            r = _session().get(url, timeout=25)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f'SEC endpoint unavailable: {url}')

@lru_cache(maxsize=1)
def ticker_cik_map():
    records = _json(SEC_TICKERS)
    return {str(x['ticker']).upper(): int(x['cik_str']) for x in records.values()}

@lru_cache(maxsize=512)
def companyfacts(cik):
    return _json(SEC_FACTS.format(cik=int(cik)))

FIELDS = {
    'income': ('NetIncomeLoss', 'ProfitLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic'),
    'equity': ('StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'),
    'debt': ('LongTermDebtCurrent', 'LongTermDebtNoncurrent', 'LongTermDebt', 'LongTermDebtAndCapitalLeaseObligations'),
    'shares': ('CommonStockSharesOutstanding', 'EntityCommonStockSharesOutstanding', 'WeightedAverageNumberOfDilutedSharesOutstanding'),
}

def _entries(facts, tags, units, duration=None):
    for tag in tags:
        for namespace in ('us-gaap', 'dei'):
            fact = facts.get(namespace, {}).get(tag, {})
            values = []
            for unit in units:
                values.extend(fact.get('units', {}).get(unit, []))
            rows = []
            for x in values:
                try:
                    filed = pd.Timestamp(x['filed']).normalize()
                    end = pd.Timestamp(x['end']).normalize()
                    val = float(x['val'])
                    if not np.isfinite(val): continue
                    start = pd.Timestamp(x['start']).normalize() if 'start' in x else None
                    if duration == 'annual' and (start is None or not 330 <= (end-start).days <= 380): continue
                    if duration == 'instant' and start is not None: continue
                    rows.append((filed, end, val, x.get('form', ''), x.get('accn','')))
                except (KeyError, ValueError, TypeError):
                    continue
            rows = [x for x in rows if x[3] in ('10-K','10-Q','10-K/A','10-Q/A')]
            if rows:
                return rows
    return []

def _asof(rows, cutoff):
    # Last reported fiscal period, then earliest accepted filing for that period.
    eligible = [x for x in rows if x[0] < cutoff and x[1] <= cutoff]
    if not eligible: return (np.nan, pd.NaT, pd.NaT)
    eligible.sort(key=lambda x: (x[1], x[0]))
    filed, end, val, *_ = eligible[-1]
    return val, filed, end

def _snapshots_for_ticker(ticker, cik, decision_dates):
    facts = companyfacts(cik).get('facts', {})
    inc = _entries(facts, FIELDS['income'], ('USD',), 'annual')
    eq = _entries(facts, FIELDS['equity'], ('USD',), 'instant')
    sh = _entries(facts, FIELDS['shares'], ('shares',), None)
    # SEC debt may be split current/noncurrent; use full long-term debt where available.
    debt = _entries(facts, ('LongTermDebt', 'LongTermDebtAndCapitalLeaseObligations', 'LongTermDebtNoncurrent'), ('USD',), 'instant')
    rows = []
    for date in decision_dates:
        cutoff = pd.Timestamp(date).normalize()
        earnings, ei, _ = _asof(inc, cutoff)
        equity, ef, _ = _asof(eq, cutoff)
        shares, sf, _ = _asof(sh, cutoff)
        liabilities, df, _ = _asof(debt, cutoff)
        published = max((x for x in (ei,ef,sf,df) if pd.notna(x)), default=pd.NaT)
        rows.append({'ticker': ticker, 'decision_date':cutoff, 'published_at':published,
                     'annual_earnings':earnings, 'equity':equity, 'shares':shares,
                     'debt':liabilities, 'income_filed_at':ei, 'equity_filed_at':ef,
                     'shares_filed_at':sf, 'debt_filed_at':df,
                     'fundamentals_source':'SEC EDGAR companyfacts', 'cik':cik})
    return rows

def fetch_sec_fundamentals(tickers, decision_dates, progress=None):
    """Return one snapshot per ticker/decision date; no CSV or manual inputs.

    Data downloaded live and cached in-process; app-level st.cache_data may cache
    this function's output. Caller must supply the actual decision dates.
    """
    mapping = ticker_cik_map()
    dates = sorted({pd.Timestamp(d).normalize() for d in decision_dates})
    rows=[]; failures=[]
    tickers=list(dict.fromkeys(str(t).upper() for t in tickers))
    for i, ticker in enumerate(tickers):
        cik = mapping.get(ticker) or mapping.get(ticker.replace('.', '-'))
        if cik is None:
            failures.append(f'{ticker}: no SEC CIK mapping')
            continue
        try:
            rows.extend(_snapshots_for_ticker(ticker, cik, dates))
        except (requests.RequestException, ValueError, KeyError) as exc:
            failures.append(f'{ticker}: {type(exc).__name__}: {str(exc)[:100]}')
        if progress is not None: progress(i+1, len(tickers))
        time.sleep(0.12)  # deliberately below SEC fair-access rate
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('No historical SEC companyfacts retrieved. ' + '; '.join(failures[:5]))
    result.attrs['failures'] = failures
    return result
