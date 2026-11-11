"""EUR Fama-French factors using ECB euro short-term rate (€STR).

€STR ECB series EST.B.EU000A2X2A25.WT: annual percent, ACT/360.
This data begins in 2019. Earlier periods are intentionally unsupported.
"""
from functools import lru_cache
from io import StringIO
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd

ECB_URL = 'https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT?format=csvdata'

@lru_cache(maxsize=1)
def load_estr_rates():
    request = Request(ECB_URL, headers={'Accept': 'text/csv', 'User-Agent': 'Mozilla/5.0'})
    with urlopen(request, timeout=25) as response:
        csv = response.read().decode('utf-8-sig')
    data = pd.read_csv(StringIO(csv), usecols=['TIME_PERIOD', 'OBS_VALUE'])
    data['TIME_PERIOD'] = pd.to_datetime(data['TIME_PERIOD'])
    series = data.dropna().drop_duplicates('TIME_PERIOD').set_index('TIME_PERIOD')['OBS_VALUE'].astype(float).sort_index()
    if series.empty:
        raise ValueError('ECB €STR returned no usable observations')
    return series


def convert_factors_to_eur(factors_usd, eurusd, *, rates=None, min_observations=80):
    """Convert USD FF factor returns to EUR, daily simple returns in decimals.

    ECB €STR observations are published on the following business day.
    We lag the observed rate one ECB observation to avoid using an unpublished rate.
    The daily RF approximation uses the elapsed calendar days / 360.
    """
    if not {'MKT-RF', 'RF'}.issubset(factors_usd.columns):
        raise ValueError('USD factors must contain MKT-RF and RF')
    fx = eurusd.iloc[:, 0] if isinstance(eurusd, pd.DataFrame) else eurusd
    fx = fx.astype(float).sort_index()
    fx.index = pd.to_datetime(fx.index).normalize()
    if not fx.index.is_unique or not np.isfinite(fx).all() or fx.le(0).any():
        raise ValueError('EURUSD prices must be finite, positive and have unique dates')
    factor_dates = pd.DatetimeIndex(factors_usd.index).sort_values()
    factor_dates = factor_dates[factor_dates >= pd.Timestamp('2019-10-02')]
    # Compute intervals before selecting the available French rows.
    fx = fx.loc[fx.index.dayofweek < 5]
    sessions = factor_dates.union(fx.index).sort_values()
    fx_on_sessions = fx.reindex(sessions)
    q = (fx_on_sessions.shift() / fx_on_sessions).reindex(factor_dates)
    rate = (load_estr_rates() if rates is None else rates.sort_index()).copy()
    # The preceding observation becomes known on the next ECB business date.
    rate.index = pd.to_datetime(rate.index).normalize()
    published = rate.shift(1).dropna()
    known = published.reindex(factor_dates, method='ffill', tolerance=pd.Timedelta(days=7))
    interval_start = sessions.to_series().shift()
    days = sessions.to_series().sub(interval_start).dt.days.reindex(factor_dates)
    rf = (known / 100.0) * days / 360.0
    cols = [c for c in ['SMB', 'HML', 'RMW', 'CMA'] if c in factors_usd.columns]
    frame = factors_usd.reindex(factor_dates)[['MKT-RF', 'RF', *cols]].copy()
    frame['q'] = q
    frame['rf_eur'] = rf
    frame = frame.dropna()
    if len(frame) < min_observations:
        raise ValueError('Insufficient overlapping €STR, EURUSD and FF factor dates')
    out = pd.DataFrame(index=frame.index)
    out['MKT-RF'] = (1 + frame['MKT-RF'] + frame['RF']) * frame['q'] - 1 - frame['rf_eur']
    for c in cols:
        out[c] = frame[c] * frame['q']
    out['RF'] = frame['rf_eur']
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError('Nonfinite converted EUR factors')
    out.attrs['currency'] = 'EUR'
    out.attrs['interval_starts'] = dict(interval_start.loc[out.index].items())
    return out


def convert_window_factors(factors_usd, fx, prices, *, rates=None):
    """Restrict factor and FX data to the estimation window, without lookahead."""
    if prices.index[0] < pd.Timestamp('2019-10-02'):
        raise ValueError('EUR factor estimation requires €STR history from October 2019')
    end = prices.index[-1]
    start = prices.index[0] - pd.Timedelta(days=10)
    factors = factors_usd.loc[(factors_usd.index >= start) & (factors_usd.index < end)]
    if factors.empty:
        raise ValueError('No historical factors in the estimation window')
    # FX must contain the preceding observation to compute the first daily change.
    fx = fx.iloc[:, 0] if isinstance(fx, pd.DataFrame) else fx
    fx = fx.loc[fx.index < end]
    return convert_factors_to_eur(factors, fx, rates=rates)
