"""Previsioni dei rendimenti tramite Machine Learning, basate su un modello aggregato (pooled) 
e su informazioni disponibili esclusivamente fino alla data di previsione (point-in-time), 
per l'ottimizzatore Kangemi esistente."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import LedoitWolf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.optimization import optimize_maximum_sharpe

FEATURES = ['mom_21','mom_63','mom_126','mom_252','vol_21','vol_63','dd_63','market_mom_63','rel_mom_63']
MODELS = ('Elastic Net','Random Forest','XGBoost')

@dataclass
class ForecastResult:
    forecasts: pd.DataFrame
    weights: pd.DataFrame
    metrics: pd.DataFrame
    covariance: pd.DataFrame
    train_rows: int
    validation_rows: int
    as_of: pd.Timestamp


def make_features(prices: pd.DataFrame, horizon: int, benchmark: pd.Series | None = None):
    """Target is H-session forward simple return, never a daily-return proxy."""
    prices = prices.sort_index().astype(float).dropna(how='any')
    r = prices.pct_change(fill_method=None)
    if benchmark is None:
        market = r.mean(axis=1).fillna(0.0)
        market_price = (1 + market).cumprod()
    else:
        market_price = benchmark.reindex(prices.index).ffill().bfill()
    frames = []
    for ticker in prices.columns:
        p = prices[ticker]
        f = pd.DataFrame(index=prices.index)
        for n in (21,63,126,252):
            f[f'mom_{n}'] = p.pct_change(n, fill_method=None)
        for n in (21,63):
            f[f'vol_{n}'] = r[ticker].rolling(n).std() * np.sqrt(252)
        f['dd_63'] = p / p.rolling(63).max() - 1
        f['market_mom_63'] = market_price.pct_change(63, fill_method=None)
        f['rel_mom_63'] = f['mom_63'] - f['market_mom_63']
        f['target'] = p.shift(-horizon) / p - 1
        f['ticker'] = ticker
        f['date'] = prices.index
        frames.append(f.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True).replace([np.inf,-np.inf], np.nan)


def _model(name):
    if name == 'Elastic Net':
        return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),ElasticNet(alpha=0.003,l1_ratio=0.25,max_iter=10000,random_state=42))
    if name == 'Random Forest':
        return make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=120,max_depth=5,min_samples_leaf=12,n_jobs=-1,random_state=42))
    if name == 'XGBoost':
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError('Install xgboost from requirements.txt') from exc
        return make_pipeline(SimpleImputer(strategy='median'),XGBRegressor(n_estimators=180,max_depth=3,learning_rate=0.035,subsample=0.85,colsample_bytree=0.85,reg_lambda=12,objective='reg:squarederror',n_jobs=2,random_state=42))
    raise ValueError(name)


def fit_forecasts(prices: pd.DataFrame, training_years=3, horizon=63, model_names=MODELS,
                  risk_free_rate=0.035, max_weight=0.4, as_of=None, validation_fraction=0.2):
    """Pooled training with chronological validation and purging of overlapping targets.

    as_of denotes the last observable price date. Only labels whose END is <= as_of
    enter the fit; validation begins after a purged training boundary.
    """
    prices = prices.sort_index().dropna(how='any')
    if as_of is not None:
        prices = prices.loc[:pd.Timestamp(as_of)]
    if len(prices) < 252 + horizon + 100:
        raise ValueError('Not enough price history for features and forecast horizon.')
    as_of = prices.index[-1]
    train_start_idx = max(0, len(prices)-1-int(training_years*252))
    if train_start_idx >= len(prices)-252-horizon-50:
        raise ValueError('Training window too short for 252-day features and labeled targets; choose more history.')
    frame = make_features(prices, horizon)
    dates = prices.index
    # Labeled sample dates must be at most H sessions before as_of.
    last_labeled = dates[-horizon-1]
    sample = frame[(frame.date >= dates[train_start_idx]) & (frame.date <= last_labeled)].dropna(subset=FEATURES+['target']).copy()
    unique_dates = sorted(sample.date.unique())
    if len(unique_dates) < 90:
        raise ValueError('Insufficient labeled dates; use a longer training period.')
    split = max(1,int(len(unique_dates)*(1-validation_fraction)))
    validation_start = unique_dates[split]
    # Train labels cannot end after the first validation feature date.
    train_boundary = dates[dates.get_loc(validation_start)-horizon] if dates.get_loc(validation_start)>=horizon else dates[0]
    train = sample[sample.date <= train_boundary]
    val = sample[sample.date >= validation_start]
    if train.empty or val.empty:
        raise ValueError('No valid temporal train/validation split.')
    final = frame[frame.date == as_of].dropna(subset=FEATURES)
    if len(final) != len(prices.columns):
        raise ValueError('Some assets lack complete current features.')
    cov = pd.DataFrame(LedoitWolf().fit(prices.pct_change(fill_method=None).dropna().tail(int(training_years*252))).covariance_*252,index=prices.columns,columns=prices.columns)
    results, weights, metrics = [], [], []
    for name in model_names:
        model = _model(name)
        model.fit(train[FEATURES],train.target)
        pred_val = model.predict(val[FEATURES])
        metrics.append({'Model':name,'MAE':mean_absolute_error(val.target,pred_val),'RMSE':np.sqrt(mean_squared_error(val.target,pred_val)), 'Rank IC':_rank_ic(val,pred_val)})
        # Refit on all labels available by as_of after validation measurement.
        model.fit(sample[FEATURES],sample.target)
        predicted_horizon = pd.Series(model.predict(final[FEATURES]),index=final.ticker).reindex(prices.columns)
        # Geometric annualization, with floor to avoid invalid fractional powers.
        annual = (1+predicted_horizon.clip(lower=-0.95))**(252/horizon)-1
        # Max Sharpe optimizer accepts annual expected returns and annual covariance.
        w = optimize_maximum_sharpe(annual,cov,risk_free_rate=risk_free_rate,max_weight=max_weight)
        results.append(pd.DataFrame({'Model':name,'Ticker':annual.index,'Forecast horizon return':predicted_horizon.values,'Expected return annualized':annual.values}))
        weights.append(pd.DataFrame({'Model':name,'Ticker':w.index,'Weight':w.values}))
    return ForecastResult(pd.concat(results,ignore_index=True),pd.concat(weights,ignore_index=True),pd.DataFrame(metrics),cov,len(train),len(val),as_of)


def _rank_ic(validation, predictions):
    v = validation[['date','ticker','target']].copy()
    v['pred'] = predictions
    ics = [g.target.corr(g.pred,method='spearman') for _,g in v.groupby('date') if len(g)>=3]
    valid = [x for x in ics if pd.notna(x)]
    return float(np.mean(valid)) if valid else np.nan


def backtest_ml(prices, training_years=3, horizon=63, rebalance=63, model_names=MODELS,
                risk_free_rate=0.035, max_weight=0.4, test_sessions=252, fee_bps=10):
    """Quarterly retrain and rebalance; fixed initial universe; daily buy-and-hold between trades.

    Fees are proportional to absolute traded notional; output equity begins at 1.
    """
    prices = prices.sort_index().dropna(how='any')
    if len(prices) < training_years*252+test_sessions+1:
        raise ValueError('Need training history PLUS separate test history; request more data.')
    start = len(prices)-test_sessions-1
    if start < 252+horizon+90:
        raise ValueError('Insufficient history for purged ML training and features.')
    equity = {m:1.0 for m in model_names}
    holdings = {m:pd.Series(0.0,index=prices.columns) for m in model_names}
    records=[]
    for t in range(start,len(prices)-1):
        if (t-start)%rebalance == 0:
            fit = fit_forecasts(prices.iloc[:t+1],training_years,horizon,model_names,risk_free_rate,max_weight)
            for name in model_names:
                new = fit.weights[fit.weights.Model==name].set_index('Ticker').Weight.reindex(prices.columns).fillna(0.0)
                traded = float((new-holdings[name]).abs().sum())
                equity[name] *= (1-fee_bps/10000*traded)
                holdings[name] = new
        daily = prices.iloc[t+1]/prices.iloc[t]-1
        for name in model_names:
            growth = 1+float((holdings[name]*daily).sum())
            equity[name] *= growth
            holdings[name] = holdings[name]*(1+daily)/growth
            records.append({'Date':prices.index[t+1],'Model':name,'Equity':equity[name]})
    return pd.DataFrame(records)
