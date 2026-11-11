"""Daily EUR FF3 portfolio performance attribution, with explicit missing-data coverage.

Factor exposures are frozen at each rebalance and applied from the next trading
return. Attribution is linked additively to the initial portfolio capital.
"""
import numpy as np
import pandas as pd
from src.factor_eur import convert_factors_to_eur

COMPONENTS = ('Market', 'SMB', 'HML', 'Risk-free', 'Unexplained')
MISSING_COMPONENT = 'Unattributed days'


def compute_ff3_attribution(equity_eur, exposures, factors_usd, eurusd, *, rates=None, initial_capital=None):
    if not isinstance(equity_eur, (pd.Series, pd.DataFrame)) or exposures is None:
        raise ValueError('Missing FF3 portfolio or exposure history.')
    if initial_capital is None:
        initial_capital = equity_eur.attrs.get('initial_capital')
    equity = equity_eur['Portfolio Value'] if isinstance(equity_eur, pd.DataFrame) else equity_eur
    equity = equity.copy().astype(float)
    equity.index = pd.to_datetime(equity.index).normalize()
    equity = equity[~equity.index.duplicated(keep='last')].sort_index()
    if len(equity) < 2 or not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError('Not enough valid FF3 equity observations.')
    exp = exposures.copy()
    exp.index = pd.to_datetime(exp.index).normalize()
    exp = exp[~exp.index.duplicated(keep='last')].sort_index()
    factors = factors_usd.copy()
    factors.index = pd.to_datetime(factors.index).normalize()
    factors = factors[~factors.index.duplicated(keep='last')].sort_index()
    # Convert only factor observations that genuinely exist. No factor extrapolation.
    factors = factors.loc[equity.index[0]:equity.index[-1]]
    factors_eur = convert_factors_to_eur(factors, eurusd, rates=rates, min_observations=0)
    fac = factors_eur.reindex(equity.index)
    interval_start = pd.Series(factors_eur.attrs['interval_starts'], dtype='datetime64[ns]')
    same_interval = equity.index.to_series().shift().eq(interval_start.reindex(equity.index))
    # Rebalance-day exposures apply to the NEXT observed return.
    loadings = exp.reindex(equity.index, method='ffill').shift(1)
    eur_ret = equity.pct_change()
    previous_equity = equity.shift(1)
    initial_equity = float(equity.iloc[0] if initial_capital is None else initial_capital)
    if not np.isfinite(initial_equity) or initial_equity <= 0:
        raise ValueError('Initial capital must be finite and positive.')
    rows, missing = [], []
    entry_cost = float(equity.iloc[0] / initial_equity - 1)
    if entry_cost:
        rows.append((equity.index[0], {**dict.fromkeys(COMPONENTS, 0.0),
                                      'Unexplained': entry_cost, MISSING_COMPONENT: 0.0}))
    for dt in equity.index[1:]:
        daily_pnl = float((equity.loc[dt] - previous_equity.loc[dt]) / initial_equity)
        if (not same_interval.loc[dt] or not np.isfinite(eur_ret.loc[dt])
            or loadings.loc[dt, ['Market Beta', 'SMB Exposure', 'HML Exposure']].isna().any()
            or fac.loc[dt, ['MKT-RF', 'SMB', 'HML', 'RF']].isna().any()):
            missing.append(dt)
            rows.append((dt, {**{k: 0.0 for k in COMPONENTS}, MISSING_COMPONENT: daily_pnl}))
            continue
        beta = loadings.loc[dt]
        mk = float(beta['Market Beta'] * fac.loc[dt, 'MKT-RF'])
        smb = float(beta['SMB Exposure'] * fac.loc[dt, 'SMB'])
        hml = float(beta['HML Exposure'] * fac.loc[dt, 'HML'])
        rf = float(fac.loc[dt, 'RF'])
        residual = float(eur_ret.loc[dt]) - (mk + smb + hml + rf)
        scale = float(previous_equity.loc[dt] / initial_equity)
        rows.append((dt, {'Market': mk*scale, 'SMB': smb*scale,
                          'HML': hml*scale, 'Risk-free': rf*scale,
                          'Unexplained': residual*scale, MISSING_COMPONENT: 0.0}))
    daily = pd.DataFrame({dt: vals for dt, vals in rows}).T
    daily.index.name = 'Date'
    daily = daily.loc[:, [*COMPONENTS, MISSING_COMPONENT]]
    cumulative = daily.cumsum()
    total_return = float(equity.iloc[-1] / initial_equity - 1.0)
    difference = float(cumulative.iloc[-1].sum()) - total_return
    if not np.isclose(difference, 0, atol=1e-9, rtol=0):
        raise ValueError(f'Attribution reconciliation failed: difference={difference:.10f}')
    covered = equity.index[1:].difference(pd.DatetimeIndex(missing))
    return {'daily': daily, 'cumulative': cumulative,
            'summary': cumulative.iloc[-1].rename('Contribution'),
            'covered_days': len(covered), 'total_days': len(equity)-1,
            'missing_days': len(missing), 'missing_dates': pd.DatetimeIndex(missing),
            'last_covered_date': covered.max() if len(covered) else None,
            'last_factor_date': factors_eur.index.max(),
            'total_return': total_return, 'reconciliation_difference': difference}
