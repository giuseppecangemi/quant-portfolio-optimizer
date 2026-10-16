"""Confronto controllato, su base giornaliera, 
tra rendimenti attesi stimati mediante metodi storici, modelli fattoriali e Machine Learning.
Tutti gli stimatori utilizzano esclusivamente le informazioni disponibili fino alla data di decisione inclusa. 
La combinazione ibrida dei modelli è fissata **ex ante al 50%** e NON viene calibrata sui dati del backtest.
I file dei fattori Fama-French FF3/FF5 sono denominati in USD e vengono convertiti in EUR utilizzando gli stimatori già implementati.
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from src.optimization import optimize_maximum_sharpe
from src.factors import capm_expected_returns
from src.fama_french import estimate_ff3_eur_returns
from src.fama_french_5 import estimate_ff5_eur_returns
from src.ml_forecasting import fit_forecasts, MODELS

BASES = ('Historical Mean', 'CAPM', 'FF3', 'FF5')


def estimators(prices, market_prices, ff3, ff5, eurusd, risk_free_rate, lookback=756):
    """Return DAILY expected simple returns. No factor data after decision date."""
    p = prices.tail(lookback + 1)
    r = p.pct_change(fill_method=None).dropna()
    date = p.index[-1]
    out = {'Historical Mean': r.mean()}
    market = market_prices.loc[:date].reindex(p.index).ffill().pct_change(fill_method=None)
    out['CAPM'] = capm_expected_returns(r, market, risk_free_rate=risk_free_rate) / 252
    # Existing FF3/FF5 implementations are the source of truth for FX conversion.
    out['FF3'] = estimate_ff3_eur_returns(p, ff3.loc[:date], eurusd.loc[:date]) / 252
    out['FF5'] = estimate_ff5_eur_returns(p, ff5.loc[:date], eurusd.loc[:date]) / 252
    return {k: v.reindex(prices.columns).astype(float) for k, v in out.items()}


def portfolio_table(daily_estimates, cov_annual, risk_free_rate, max_weight):
    rows, allocations = [], []
    for name, mu_day in daily_estimates.items():
        mu_annual = mu_day * 252
        if not np.isfinite(mu_annual).all():
            raise ValueError(f'Invalid expected returns for {name}')
        w = optimize_maximum_sharpe(mu_annual, cov_annual, risk_free_rate, max_weight)
        expected = float(w @ mu_annual)
        volatility = float(np.sqrt(w @ cov_annual @ w))
        sharpe = (expected-risk_free_rate)/volatility if volatility > 0 else np.nan
        rows.append({'Strategy': name, 'Expected Return': expected,
                     'Volatility': volatility, 'Sharpe Ratio': sharpe})
        allocations.extend({'Strategy':name, 'Ticker':ticker, 'Weight':float(weight)} for ticker,weight in w.items())
    return pd.DataFrame(rows), pd.DataFrame(allocations)


def compare_at_date(prices, market_prices, ff3, ff5, eurusd, risk_free_rate=0.,
                    max_weight=.4, training_years=3, model_names=MODELS,
                    hybrid_alpha=.5, as_of=None):
    if not 0 <= hybrid_alpha <= 1:
        raise ValueError('Hybrid alpha must be in [0,1].')
    p = prices.loc[:as_of] if as_of is not None else prices
    p = p.sort_index().dropna(how='any')
    bases = estimators(p, market_prices, ff3, ff5, eurusd, risk_free_rate, int(training_years*252))
    fit = fit_forecasts(p, training_years=training_years, horizon=1,
                        model_names=model_names, risk_free_rate=risk_free_rate,
                        max_weight=max_weight, validation_fraction=.2)
    r = p.pct_change(fill_method=None).dropna().tail(int(training_years*252))
    cov = pd.DataFrame(LedoitWolf().fit(r).covariance_*252,index=p.columns,columns=p.columns)
    all_mu = dict(bases)
    for name in model_names:
        # Horizon=1: the ML output is a forecast of next-day simple return.
        ml = fit.forecasts.loc[fit.forecasts.Model==name].set_index('Ticker')['Forecast horizon return'].reindex(p.columns)
        all_mu[f'Pure ML · {name}'] = ml
        for base, historical in bases.items():
            all_mu[f'{base} + {name}'] = (1-hybrid_alpha)*historical + hybrid_alpha*ml
    stats, weights = portfolio_table(all_mu,cov,risk_free_rate,max_weight)
    daily = pd.DataFrame(all_mu).rename_axis('Ticker').reset_index()
    return {'daily':daily, 'stats':stats, 'weights':weights, 'quality':fit.metrics,
            'covariance':cov, 'as_of':p.index[-1]}


def walk_forward_comparison(prices, market_prices, ff3, ff5, eurusd,
                            training_years=3, model_names=MODELS, test_sessions=252,
                            rebalance=63, risk_free_rate=0., max_weight=.4,
                            hybrid_alpha=.5, fee_bps=3., benchmark=None):
    """One shared walk-forward clock; weights drift between rebalance dates.

    Point-in-time asset selection is not simulated; initial universe is fixed.
    """
    p = prices.sort_index().dropna(how='any')
    start = len(p)-int(test_sessions)-1
    if start < training_years*252+252:
        raise ValueError('Need additional pre-test history for 252-session features and training.')
    capital, holdings, rows = {}, {}, []
    for t in range(start,len(p)-1):
        if (t-start)%rebalance == 0:
            comparison = compare_at_date(p.iloc[:t+1],market_prices,ff3,ff5,eurusd,
                    risk_free_rate,max_weight,training_years,model_names,hybrid_alpha)
            for strategy,g in comparison['weights'].groupby('Strategy'):
                target = g.set_index('Ticker').Weight.reindex(p.columns).fillna(0.)
                if strategy not in capital:
                    capital[strategy], holdings[strategy] = 1., pd.Series(0.,index=p.columns)
                turnover = float((target-holdings[strategy]).abs().sum())
                capital[strategy] *= 1-fee_bps/10000*turnover
                holdings[strategy] = target
        daily = p.iloc[t+1].div(p.iloc[t]).sub(1)
        for strategy in capital:
            growth = 1+float((holdings[strategy]*daily).sum())
            capital[strategy] *= growth
            holdings[strategy] = holdings[strategy].mul(1+daily).div(growth)
            rows.append({'Date':p.index[t+1], 'Strategy':strategy,'Equity':capital[strategy]})
    if benchmark is not None:
        bm = benchmark.loc[:p.index[-1]].reindex(p.index).ffill()
        base = bm.iloc[start]
        for date,value in bm.iloc[start+1:].items():
            if np.isfinite(value) and np.isfinite(base) and base > 0:
                rows.append({'Date':date,'Strategy':'FTSE MIB (price index)','Equity':float(value/base)})
    return pd.DataFrame(rows)


def backtest_metrics(equity, risk_free_rate=0.):
    results=[]
    for strategy,g in equity.groupby('Strategy'):
        v=g.sort_values('Date').set_index('Date').Equity
        r=v.pct_change().dropna()
        if r.empty: continue
        peak=v.cummax()
        years=len(r)/252
        results.append({'Strategy':strategy,'Total Return':v.iloc[-1]-1,
            'CAGR':v.iloc[-1]**(1/years)-1,
            'Volatility':r.std()*np.sqrt(252),
            'Sharpe':(r.mean()*252-risk_free_rate)/(r.std()*np.sqrt(252)) if r.std()>0 else np.nan,
            'Max Drawdown':(v/peak-1).min()})
    return pd.DataFrame(results).sort_values('Total Return',ascending=False)
