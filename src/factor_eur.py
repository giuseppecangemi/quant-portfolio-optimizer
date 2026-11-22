"""EUR Fama-French factors using ECB euro short-term rate (€STR).

€STR ECB series EST.B.EU000A2X2A25.WT: annual percent, ACT/360.
Legacy €STR conversion and opt-in historical EONIA extension.
"""
from functools import lru_cache
from io import StringIO
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
from src.eur_risk_free import (
    EONIA_ESTR, ESTR_ONLY, FIRST_ESTR_PUBLICATION, FIRST_ESTR_REFERENCE,
    FIRST_SUPPORTED_EONIA, eonia_interval_returns, validate_mode,
)

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


def convert_factors_to_eur(factors_usd, eurusd, *, rates=None, min_observations=80, eur_rf_mode=ESTR_ONLY, eonia_rates=None):
    """Convert USD FF factor returns to EUR, daily simple returns in decimals.

    ECB €STR observations are published on the following business day.
    We lag the observed rate one ECB observation to avoid using an unpublished rate.
    The daily RF approximation uses the elapsed calendar days / 360.
    """
    validate_mode(eur_rf_mode)
    if not {'MKT-RF', 'RF'}.issubset(factors_usd.columns):
        raise ValueError('USD factors must contain MKT-RF and RF')
    fx = eurusd.iloc[:, 0] if isinstance(eurusd, pd.DataFrame) else eurusd
    fx = fx.astype(float).sort_index()
    fx.index = pd.to_datetime(fx.index).normalize()
    if not fx.index.is_unique or not np.isfinite(fx).all() or fx.le(0).any():
        raise ValueError('EURUSD prices must be finite, positive and have unique dates')
    factor_dates = pd.DatetimeIndex(factors_usd.index).sort_values()
    factor_dates = factor_dates[factor_dates >= (FIRST_ESTR_PUBLICATION if eur_rf_mode == ESTR_ONLY
                                          else FIRST_SUPPORTED_EONIA)]
    # Compute intervals before selecting the available French rows.
    fx = fx.loc[fx.index.dayofweek < 5]
    sessions = factor_dates.union(fx.index).sort_values()
    fx_on_sessions = fx.reindex(sessions)
    q = (fx_on_sessions.shift() / fx_on_sessions).reindex(factor_dates)
    needs_estr = eur_rf_mode == ESTR_ONLY or (factor_dates >= FIRST_ESTR_PUBLICATION).any()
    rate = ((load_estr_rates() if rates is None else rates.sort_index()).copy() if needs_estr
            else pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    # The preceding observation becomes known on the next ECB business date.
    rate.index = pd.to_datetime(rate.index).normalize()
    published = rate.shift(1).dropna()
    known = published.reindex(factor_dates, method='ffill', tolerance=pd.Timedelta(days=7))
    interval_start = sessions.to_series().shift()
    days = sessions.to_series().sub(interval_start).dt.days.reindex(factor_dates)
    rf = (known / 100.0) * days / 360.0
    if eur_rf_mode == EONIA_ESTR:
        historical = factor_dates[factor_dates <= FIRST_ESTR_REFERENCE]
        rf.loc[historical] = eonia_interval_returns(
            interval_start.reindex(historical), rates=eonia_rates,
        )
        # The first €STR fixing was available during the 2 October session.
        # Use it only for its actual one-day interval when that interval is present.
        if (FIRST_ESTR_PUBLICATION in rf.index
                and interval_start.loc[FIRST_ESTR_PUBLICATION] == FIRST_ESTR_REFERENCE
                and FIRST_ESTR_REFERENCE in rate.index):
            rf.loc[FIRST_ESTR_PUBLICATION] = rate.loc[FIRST_ESTR_REFERENCE] / 100 / 360
    cols = [c for c in ['SMB', 'HML', 'RMW', 'CMA'] if c in factors_usd.columns]
    frame = factors_usd.reindex(factor_dates)[['MKT-RF', 'RF', *cols]].copy()
    frame['q'] = q
    frame['rf_eur'] = rf
    frame = frame.dropna()
    if len(frame) < min_observations:
        raise ValueError(f'Insufficient overlapping {eur_rf_mode}, EURUSD and FF factor dates ({len(frame)} usable)')
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


def convert_window_factors(factors_usd, fx, prices, *, rates=None, eur_rf_mode=ESTR_ONLY):
    """Restrict factor and FX data to the estimation window, without lookahead."""
    if eur_rf_mode == ESTR_ONLY and prices.index[0] < FIRST_ESTR_PUBLICATION:
        raise ValueError('EUR factor estimation requires €STR history from October 2019')
    if eur_rf_mode == EONIA_ESTR and prices.index[0] < pd.Timestamp('2007-01-01'):
        raise ValueError('Extended EUR risk-free coverage starts in January 2007')
    end = prices.index[-1]
    start = prices.index[0] - pd.Timedelta(days=10)
    factors = factors_usd.loc[(factors_usd.index >= start) & (factors_usd.index < end)]
    if factors.empty:
        raise ValueError('No historical factors in the estimation window')
    # FX must contain the preceding observation to compute the first daily change.
    fx = fx.iloc[:, 0] if isinstance(fx, pd.DataFrame) else fx
    fx = fx.loc[fx.index < end]
    # Preserve the legacy sample exactly when no return interval uses EONIA.
    mode = ESTR_ONLY if prices.index[0] >= FIRST_ESTR_PUBLICATION and eur_rf_mode == EONIA_ESTR else eur_rf_mode
    return convert_factors_to_eur(factors, fx, rates=rates, eur_rf_mode=mode)
