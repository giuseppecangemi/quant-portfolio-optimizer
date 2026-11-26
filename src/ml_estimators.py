"""Causal ML portfolio inputs shared by research and existing backtests."""
import numpy as np
import pandas as pd

from src.forward_mc import covariance, validate_covariance
from src.ml_forecasting import FEATURES, MODELS, _model, make_features
from src.optimization import (
    efficient_frontier, optimize_hrp, optimize_maximum_sharpe,
    optimize_minimum_volatility, optimize_risk_parity,
)

ANNUAL = 252
OPTIMIZERS = ("Maximum Sharpe", "Minimum Volatility", "Risk Parity", "HRP")
CONFIGS = ("Historical", "ML Returns", "ML Covariance", "Full ML")


def checked_prices(prices):
    prices = prices.sort_index()
    if (not prices.index.is_unique or not prices.columns.is_unique or prices.empty
            or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()):
        raise ValueError("ML requires complete positive prices on a unique common session index.")
    return prices


def horizon_to_daily(values, horizon):
    values = np.asarray(values)
    if horizon < 1 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Forecast horizon returns must be finite and greater than −100%.")
    return np.expm1(np.log1p(values) / horizon)


def return_panel(prices, horizon):
    frame = make_features(prices, horizon)
    ends = pd.Series(prices.index, index=prices.index).shift(-horizon)
    frame["label_end"] = frame.date.map(ends)
    return frame.dropna(subset=FEATURES)


def return_fit(panel, origin, model_name):
    train = panel.loc[(panel.date < origin) & (panel.label_end <= origin)]
    point = panel.loc[panel.date == origin]
    if train.date.nunique() < 80 or point.empty:
        raise ValueError("Need at least 80 labeled training sessions after feature warm-up and horizon purge.")
    estimator = _model(model_name)
    estimator.fit(train[FEATURES], train.target)
    prediction = pd.Series(estimator.predict(point[FEATURES]),index=point.ticker)
    baseline = train.groupby("ticker").target.mean().reindex(prediction.index)
    regressor = estimator.steps[-1][1]
    importance = getattr(regressor,"feature_importances_",getattr(regressor,"coef_",None))
    return prediction, baseline, pd.Series(importance,index=FEATURES), train.label_end.max()


def variance_panel(returns):
    """Pooled asset-level HAR-style predictors for next-day squared return."""
    square = returns**2
    lagged = {f"Variance {n}": square.rolling(n).mean().stack() for n in (1,5,21,63)}
    return pd.DataFrame({**lagged,"Target":square.shift(-1).stack()}).sort_index()


def estimate_covariance(returns, method, panel=None):
    if method != "ML":
        matrix = covariance(returns.to_numpy(),method,.94)
        return pd.DataFrame(validate_covariance(matrix)*ANNUAL,index=returns.columns,columns=returns.columns)
    if len(returns) < 143:
        raise ValueError("ML covariance requires at least 143 daily returns.")
    frame = variance_panel(returns) if panel is None else panel
    origin = returns.index[-1]
    fields = ["Variance 1","Variance 5","Variance 21","Variance 63"]
    train = frame.loc[(frame.index.get_level_values(0)<origin)].dropna(subset=[*fields,"Target"])
    point = frame.xs(origin)[fields].reindex(returns.columns)
    if len(train) < 80 or point.isna().any().any():
        raise ValueError("Insufficient complete variance features.")
    # Nonnegative supervised predictions. Correlation uses Ledoit–Wolf shrinkage.
    model = _model("Random Forest")
    model.fit(train[fields]*1e4,train.Target*1e4)
    daily_var = np.maximum(model.predict(point*1e4)/1e4,1e-12)
    shrunk = covariance(returns.to_numpy(),"Ledoit–Wolf",.94)
    sd = np.sqrt(np.maximum(np.diag(shrunk),1e-12))
    corr = shrunk/np.outer(sd,sd)
    np.fill_diagonal(corr,1.)
    matrix = corr*np.sqrt(np.outer(daily_var,daily_var))
    return pd.DataFrame(validate_covariance(matrix)*ANNUAL,index=returns.columns,columns=returns.columns)


def optimize_inputs(mu, cov, method, rf, cap):
    from src.portfolio_context import check_minimum
    if method == "Maximum Sharpe":
        w = optimize_maximum_sharpe(mu,cov,risk_free_rate=rf,max_weight=cap)
    elif method == "Minimum Volatility":
        w = optimize_minimum_volatility(mu,cov,max_weight=cap)
    elif method == "Risk Parity":
        w = optimize_risk_parity(cov,max_weight=cap)
    elif method == "HRP":
        w = optimize_hrp(cov,max_weight=cap)
    else:
        raise ValueError(f"Unknown optimizer: {method}")
    if not np.isfinite(w).all() or not np.isclose(w.sum(),1,atol=1e-6) or w.min()<-1e-7 or w.max()>cap+1e-6:
        raise ValueError("Optimized portfolio failed weight constraints.")
    check_minimum(w)
    return w


def fit_portfolio(prices, config, rf, cap, cache=None):
    prices = checked_prices(prices)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    cfg,horizon,name,method,base_cov = (config[k] for k in ("configuration","horizon","model","optimizer","historical_cov"))
    key = ("ml_estimators",prices.index[0],prices.index[-1],tuple(prices.columns))
    cache = {} if cache is None else cache
    mu = returns.mean()*ANNUAL
    cov_name = "ML" if cfg in ("ML Covariance","Full ML") else base_cov
    cov_key = (*key,cov_name)
    if cov_key not in cache:
        cache[cov_key] = estimate_covariance(returns,cov_name)
    cov = cache[cov_key]
    if cfg in ("ML Returns","Full ML"):
        mean_key = (*key,"mean",name,horizon)
        if mean_key not in cache:
            estimate = return_fit(return_panel(prices,horizon),prices.index[-1],name)[0].reindex(prices.columns)
            cache[mean_key] = pd.Series(horizon_to_daily(estimate,horizon)*ANNUAL,index=prices.columns)
        mu = cache[mean_key]
    try:
        weights = optimize_inputs(mu, cov, method, rf, cap)
    except ValueError as exc:
        matrix = cov.loc[mu.index, mu.index].to_numpy()
        eigenvalues = np.linalg.eigvalsh(matrix)
        vol = np.sqrt(np.diag(matrix))
        alternatives = {}
        if method == "Maximum Sharpe" and cfg == "Full ML":
            historical_mean = returns.mean() * ANNUAL
            historical_cov = estimate_covariance(returns, base_cov)
            for label, mean, risk in (
                ("Historical", historical_mean, historical_cov),
                ("ML Returns", mu, historical_cov),
                ("ML Covariance", historical_mean, cov),
            ):
                try:
                    candidate = optimize_inputs(mean, risk, method, rf, cap)
                    alternatives[label] = float((candidate @ mean - rf) / np.sqrt(candidate @ risk @ candidate))
                except ValueError as failure:
                    alternatives[label] = str(failure)
        raise ValueError(
            f"Estimate through {prices.index[-1].date()}; {exc} "
            f"Annualized inputs (252 sessions), risk-free={rf}, asset count={len(mu)}, "
            f"long-only, sum=1, cap={cap}. Expected return range={[mu.min(), mu.max()]}, "
            f"volatility range={[vol.min(), vol.max()]}, eigenvalues={eigenvalues.tolist()}, "
            f"condition={np.linalg.cond(matrix)}, additional solver regularization=0. "
            f"Expected returns by asset={mu.to_dict()}, covariance in the same order={matrix.tolist()}. "
            f"Same-date comparisons (Sharpe or failure)={alternatives}."
        ) from exc
    return weights, mu, cov


def validate_returns(prices,horizon,models,stride):
    panel=return_panel(prices,horizon)
    dates=pd.Index(panel.loc[panel.label_end.notna(),"date"].unique()).sort_values()
    if len(dates)<180:
        raise ValueError("Need at least 180 labeled sessions for chronological validation and test.")
    validation_start,test_start=dates[int(.60*len(dates))],dates[int(.80*len(dates))]
    rows=[]
    for origin in dates[dates>=validation_start][::stride]:
        actual=panel.loc[panel.date.eq(origin)].set_index("ticker")
        if origin<test_start and actual.label_end.iloc[0]>=test_start:
            continue
        for name in models:
            est,baseline,_,trained_through=return_fit(panel,origin,name)
            rows.append(pd.DataFrame({"Date":origin,"Ticker":est.index,"Model":name,
                "Phase":"Validation" if origin<test_start else "Test","Predicted":est.to_numpy(),
                "Actual":actual.target.reindex(est.index).to_numpy(),"Historical":baseline.to_numpy(),
                "Training label cutoff":trained_through}))
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def forecast_metrics(frame):
    rows,ic=[],[]
    for (name,phase),g in frame.groupby(["Model","Phase"]):
        error=g.Predicted-g.Actual
        benchmark=(g.Historical-g.Actual)**2
        rank=pd.Series({date:sample.Predicted.corr(sample.Actual,method="spearman")
                        if len(sample)>=3 and sample.Predicted.nunique()>1 and sample.Actual.nunique()>1 else np.nan
                        for date,sample in g.groupby("Date")},dtype=float).sort_index()
        ic.append(rank.rename("Rank IC").rename_axis("Date").reset_index().assign(Model=name,Phase=phase))
        rows.append(dict(Model=name,Phase=phase,MAE=error.abs().mean(),RMSE=np.sqrt((error**2).mean()),
                         Bias=error.mean(),Pearson=g.Predicted.corr(g.Actual),Rank_IC=rank.mean(),
                         Directional_Accuracy=(np.sign(g.Predicted)==np.sign(g.Actual)).mean(),
                         OOS_R2=1-(error**2).sum()/benchmark.sum() if benchmark.sum() else np.nan,
                         Prediction_Dispersion=g.Predicted.std(),Origins=g.Date.nunique()))
    return pd.DataFrame(rows),pd.concat(ic,ignore_index=True)


def validate_covariances(returns,methods,stride,proxy_days=21):
    rows=[]
    weights=np.full(returns.shape[1],1/returns.shape[1])
    for i in range(max(150,int(len(returns)*.60)),len(returns)-proxy_days+1,stride):
        past,future=returns.iloc[:i],returns.iloc[i:i+proxy_days]
        # Second moment proxy for zero-mean one-day innovation risk.
        realized=future.to_numpy().T @ future.to_numpy()/proxy_days
        for method in methods:
            forecast=estimate_covariance(past,method).to_numpy()/ANNUAL
            error=np.linalg.norm(forecast-realized)
            reg=forecast+np.eye(len(weights))*1e-12
            _,logdet=np.linalg.slogdet(reg)
            ll=-.5*(len(weights)*np.log(2*np.pi)+logdet+np.trace(np.linalg.solve(reg,realized)))
            rows.append(dict(Date=past.index[-1],Method=method,Frobenius=error,
                Relative=error/max(np.linalg.norm(realized),1e-12),
                PortfolioVarianceError=float(weights@(forecast-realized)@weights),
                PredictedVol=np.sqrt(weights@forecast@weights*ANNUAL),
                RealizedVol=np.sqrt(weights@realized@weights*ANNUAL),
                MinimumEigenvalue=np.linalg.eigvalsh(forecast).min(),
                ConditionNumber=np.linalg.cond(reg),LogLikelihood=ll))
    return pd.DataFrame(rows)


def summarize_backtest(curve, weights, rf):
    from src.backtest import calculate_backtest_metrics

    metrics = calculate_backtest_metrics(curve, rf)
    drawdown = abs(metrics["Max Drawdown"])
    log = curve.attrs.get("transaction_log")
    turnover = log["Executed Turnover"].sum() if log is not None else np.nan
    metrics.update({
        "Calmar Ratio": metrics["CAGR"]/drawdown if drawdown else np.nan,
        "Executed Turnover": turnover,
        "Transaction Costs": log["Total Cost"].sum() if log is not None else 0.,
    })
    return metrics
