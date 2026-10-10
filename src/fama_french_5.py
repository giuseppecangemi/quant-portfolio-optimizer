"""Fama-French European five-factor model, daily USD factors and EUR assets.

Independent of FF3. No future information is used in rolling estimation.
"""
from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

EUROPE_FF5_URL = (
    'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
    'Europe_5_Factors_Daily_CSV.zip'
)
FACTOR_COLUMNS = ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA']


def load_europe_ff5() -> pd.DataFrame:
    """Fetch daily Europe FF5 factors, converted from percent to decimals."""
    req = Request(EUROPE_FF5_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(req, timeout=30) as response:
        content = response.read()
    with ZipFile(BytesIO(content)) as archive:
        filename = next((n for n in archive.namelist() if n.lower().endswith('.csv')), None)
        if filename is None:
            raise ValueError('Europe FF5 archive does not contain a CSV file.')
        lines = archive.read(filename).decode('utf-8-sig', errors='replace').splitlines()
    records = []
    for line in lines:
        cells = [c.strip() for c in line.split(',')]
        if len(cells) < 7 or len(cells[0]) != 8 or not cells[0].isdigit():
            continue
        try:
            date = pd.to_datetime(cells[0], format='%Y%m%d', errors='raise')
            values = [float(cells[i]) / 100 for i in range(1, 7)]
        except (ValueError, IndexError):
            continue
        if all(np.isfinite(values)) and all(abs(v) < 0.5 for v in values):
            records.append((date, *values))
    if not records:
        raise ValueError('No valid daily observations found in Europe FF5 data.')
    return (pd.DataFrame(records, columns=['Date', *FACTOR_COLUMNS, 'RF'])
            .drop_duplicates('Date').set_index('Date').sort_index())


def fit_ff5(asset_usd_returns: pd.DataFrame, factors: pd.DataFrame,
            min_observations: int = 80) -> pd.DataFrame:
    """OLS with intercept; expected USD return excludes estimated alpha."""
    required = [*FACTOR_COLUMNS, 'RF']
    if not set(required).issubset(factors.columns):
        raise ValueError(f'Missing FF5 columns: {sorted(set(required) - set(factors.columns))}')
    rows = []
    for ticker in asset_usd_returns.columns:
        joined = pd.concat([asset_usd_returns[ticker].rename('Asset'),
                            factors[required]], axis=1, join='inner').dropna()
        if len(joined) < min_observations:
            raise ValueError(f'Insufficient common FF5 observations for {ticker}: {len(joined)}')
        y = (joined['Asset'] - joined['RF']).to_numpy(float)
        x = joined[FACTOR_COLUMNS].to_numpy(float)
        X = np.column_stack([np.ones(len(x)), x])
        coeff, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank != 6:
            raise ValueError(f'Singular FF5 regression for {ticker}')
        residual = y - X @ coeff
        tss = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum(residual ** 2) / tss if tss > 0 else np.nan
        usd_daily = float(joined['RF'].mean() + np.dot(coeff[1:],
                           joined[FACTOR_COLUMNS].mean().to_numpy(float)))
        rows.append({'Ticker': ticker, 'Alpha (annualized)': coeff[0] * 252,
                     **{f'Beta {name.replace("MKT-RF", "MKT")}': float(beta)
                        for name, beta in zip(FACTOR_COLUMNS, coeff[1:])},
                     'R Squared': r2, 'Observations': len(joined),
                     'FF5 Expected Return USD': usd_daily * 252,
                     'Historical Return USD': joined['Asset'].mean() * 252})
    return pd.DataFrame(rows).set_index('Ticker')


def estimate_ff5_eur_returns(historical_prices_eur, factors, eurusd_prices,
                             min_observations=80, return_stats=False):
    """Point-in-time FF5 EUR expected returns; mirrors existing FF3 convention."""
    eur = historical_prices_eur.sort_index().dropna()
    fx = eurusd_prices.iloc[:, 0] if isinstance(eurusd_prices, pd.DataFrame) else eurusd_prices
    fx = fx.sort_index().reindex(eur.index).ffill()
    if fx.isna().any() or (fx <= 0).any():
        raise ValueError('Missing historical EUR/USD exchange rates for FF5 estimation.')
    usd_returns = eur.mul(fx, axis=0).pct_change().dropna()
    available_factors = factors.loc[factors.index < eur.index[-1]]
    stats = fit_ff5(usd_returns, available_factors, min_observations=min_observations)
    common_dates = usd_returns.index.intersection(available_factors.index)
    fx_returns = fx.pct_change().reindex(common_dates).dropna()
    if len(fx_returns) < min_observations:
        raise ValueError('Insufficient FX history for point-in-time FF5.')
    mu_usd_daily = stats['FF5 Expected Return USD'] / 252
    mu_eur = ((1 + mu_usd_daily) / (1 + float(fx_returns.mean())) - 1) * 252
    expected = mu_eur.reindex(eur.columns)
    return (expected, stats) if return_stats else expected
