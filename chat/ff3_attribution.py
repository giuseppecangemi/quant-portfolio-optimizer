"""Ex-post FF3 attribution in EUR; no additional regressions or downloads."""
import numpy as np
import pandas as pd

COMPONENTS = ('Market', 'SMB', 'HML', 'Risk-free', 'Unexplained', 'FX translation')
MISSING_COMPONENT = 'Unattributed days'

def compute_ff3_attribution(equity_eur, exposures, factors_usd, eurusd):
    """Daily additive EUR attribution using *previously fixed* portfolio loadings.

    Factor premia are in USD. For daily USD/EUR FX return f, the exact
    EUR conversion is r_EUR = (r_USD - f)/(1+f). Thus every USD
    component is divided by (1+f), with an additional -f/(1+f) FX term.
    The unexplained component is the realized USD residual (not OLS alpha).
    Contributions are linked arithmetically to initial EUR capital.
    """
    if not isinstance(equity_eur, (pd.Series, pd.DataFrame)) or exposures is None:
        raise ValueError('Missing FF3 portfolio or exposure history.')
    equity = equity_eur['Portfolio Value'] if isinstance(equity_eur, pd.DataFrame) else equity_eur
    equity = equity.copy().astype(float)
    equity.index = pd.to_datetime(equity.index).normalize()
    equity = equity[~equity.index.duplicated(keep='last')].sort_index()
    if len(equity) < 2 or (equity <= 0).any():
        raise ValueError('Not enough valid FF3 equity observations.')
    exp = exposures.copy()
    exp.index = pd.to_datetime(exp.index).normalize()
    exp = exp[~exp.index.duplicated(keep='last')].sort_index()
    fx = eurusd.iloc[:, 0] if isinstance(eurusd, pd.DataFrame) else eurusd
    fx = fx.copy().astype(float)
    fx.index = pd.to_datetime(fx.index).normalize()
    fx = fx[~fx.index.duplicated(keep='last')].sort_index()
    factors = factors_usd.copy()
    factors.index = pd.to_datetime(factors.index).normalize()
    factors = factors[~factors.index.duplicated(keep='last')].sort_index()
    # Only historical FX fixes for missing quotation dates, never future FX.
    fx = fx.reindex(equity.index).ffill()
    fx_ret = fx.pct_change()
    eur_ret = equity.pct_change()
    # A rebalance performed on day t applies to the NEXT daily return.
    loadings = exp.reindex(equity.index, method='ffill').shift(1)
    fac = factors.reindex(equity.index)
    rows = []
    missing = []
    previous_equity = equity.shift(1)
    initial_equity = float(equity.iloc[0])
    for dt in equity.index[1:]:
        if (not np.isfinite(eur_ret.loc[dt]) or not np.isfinite(fx_ret.loc[dt])
            or loadings.loc[dt, ['Market Beta', 'SMB Exposure', 'HML Exposure']].isna().any()
            or fac.loc[dt, ['MKT-RF', 'SMB', 'HML', 'RF']].isna().any()):
            missing.append(dt)
            rows.append((dt, {**{k: 0.0 for k in COMPONENTS},
                              MISSING_COMPONENT: float((equity.loc[dt] - previous_equity.loc[dt]) / initial_equity)}))
            continue
        f = float(fx_ret.loc[dt]); r = float(eur_ret.loc[dt]); d = 1 + f
        if d <= 0:
            missing.append(dt)
            rows.append((dt, {**{k: 0.0 for k in COMPONENTS},
                              MISSING_COMPONENT: float((equity.loc[dt] - previous_equity.loc[dt]) / initial_equity)}))
            continue
        beta = loadings.loc[dt]
        usd = (1+r)*d-1
        mk = float(beta['Market Beta']*fac.loc[dt, 'MKT-RF'])
        smb = float(beta['SMB Exposure']*fac.loc[dt, 'SMB'])
        hml = float(beta['HML Exposure']*fac.loc[dt, 'HML'])
        rf = float(fac.loc[dt, 'RF'])
        residual = usd - (mk+smb+hml+rf)
        daily = {'Market':mk/d, 'SMB':smb/d, 'HML':hml/d,
                 'Risk-free':rf/d, 'Unexplained':residual/d,
                 'FX translation':-f/d}
        # Monetary EUR P&L relative to initial capital: exact additive linking.
        scale = float(previous_equity.loc[dt] / initial_equity)
        rows.append((dt, {**{k:v*scale for k,v in daily.items()}, MISSING_COMPONENT: 0.0}))
    if not rows:
        raise ValueError('No matching FF3 factors, FX and equity dates for attribution.')
    daily = pd.DataFrame({dt: vals for dt,vals in rows}).T
    daily.index.name = 'Date'
    daily = daily.loc[:, [*COMPONENTS, MISSING_COMPONENT]]
    cumulative = daily.cumsum()
    # Full-period P&L is accounted for even on dates without factor data.
    total_return = float(equity.iloc[-1] / initial_equity - 1.0)
    reconstructed = float(cumulative.iloc[-1].sum())
    difference = reconstructed - total_return
    if not np.isclose(difference, 0.0, atol=1e-9, rtol=0):
        raise ValueError(f'Attribution reconciliation failed: difference={difference:.10f}')
    return {'daily': daily, 'cumulative': cumulative,
            'summary': cumulative.iloc[-1].rename('Contribution'),
            'covered_days': len(equity)-1-len(missing), 'total_days': len(equity)-1,
            'missing_days': len(missing), 'total_return': total_return,
            'reconciliation_difference': difference}
