"""Common-shock stress scenarios using the forward simulation NAV engine."""
import numpy as np
import pandas as pd

from src.forward_mc import covariance, nav_paths, summarize, validate_covariance
from src.transaction_costs import fund_orders


def allocation_matrix(portfolios, names, date):
    """Return only allocations with verifiable evidence available at cutoff."""
    rows = {}
    date = pd.Timestamp(date).date()
    for name in names:
        item = portfolios[name]
        stamp = pd.Timestamp(item.optimized_at) if item.optimized_at else pd.NaT
        dates = [x.date() for x in (item.as_of, stamp) if pd.notna(x)]
        if not dates or max(dates) > date:
            raise ValueError(f"{name}: allocation unavailable on {date}. Choose a date at or after its saved allocation.")
        w = item.weights.astype(float)
        if not w.index.is_unique or not np.isfinite(w).all() or (w < -1e-10).any() or not np.isclose(w.sum(), 1):
            raise ValueError(f"{name}: invalid target weights.")
        rows[name] = w[w > 1e-10]
    if not rows:
        raise ValueError("Select at least one portfolio.")
    result = pd.DataFrame(rows).fillna(0.)
    return result.div(result.sum())


def replay_assets(prices, start, end, assets):
    prices = prices.sort_index().loc[:, assets]
    sample = prices.pct_change(fill_method=None).loc[start:end]
    if sample.empty or not np.isfinite(sample.to_numpy()).all() or (sample <= -1).any().any():
        raise ValueError("This historical window has unavailable adjusted prices for at least one held asset.")
    return sample


def joint_shock(factors, horizon, *, start=None, end=None, quantile=None):
    """Use additive cumulative factor shocks and actual joint windows."""
    if horizon < 1:
        raise ValueError("Invalid stress duration.")
    if quantile is None:
        chosen = factors.loc[start:end]
        if len(chosen) != horizon:
            raise ValueError(f"Select a historical window with exactly {horizon} factor sessions (found {len(chosen)}).")
        return chosen.sum(), str(chosen.index[0].date()), str(chosen.index[-1].date()), 1
    sums = factors.rolling(horizon).sum().dropna()
    if len(sums) * quantile < 1:
        raise ValueError("Too few historical windows for that percentile.")
    market = sums.iloc[:,0]
    threshold = market.quantile(quantile)
    # Use the observed window immediately at or below the interpolated quantile.
    end_date = market[market <= threshold].idxmax()
    pos = factors.index.get_loc(end_date)
    return sums.loc[end_date], str(factors.index[pos-horizon+1].date()), str(end_date.date()), len(sums)


def shock_path(shock, steps, start, duration, path_type):
    if path_type not in ("One-day Shock", "Distributed Shock", "Persistent Regime Shift"):
        raise ValueError("Unknown shock path.")
    length = 1 if path_type == "One-day Shock" else steps-start if path_type == "Persistent Regime Shift" else duration
    if start < 0 or length < 1 or start+length > steps:
        raise ValueError("Shock timing and duration must fit within the horizon.")
    path = np.zeros((steps,len(shock)))
    path[start:start+length] = np.asarray(shock)/length
    return path


def linear_attribution(weights, beta, vector, daily_mean, days, capital):
    labels = vector.index
    beta = pd.DataFrame(beta,index=weights.index,columns=labels)
    factor = weights.T @ beta
    contributions = factor.mul(vector)
    asset_return = beta @ vector
    asset = weights.mul(asset_return,axis=0)
    impact = contributions.sum(axis=1)
    np.testing.assert_allclose(impact, asset.sum(axis=0), atol=1e-12)
    baseline = weights.T @ daily_mean * days
    summary = pd.DataFrame({"Baseline Expected Return":baseline,"Factor Shock Impact":impact,
                            "Stressed Expected Return":baseline+impact,"Impact EUR":capital*impact})
    return summary,factor,contributions,asset,asset_return


def stressed_covariance(values, market, baseline, quantile=.20):
    mask = market <= np.quantile(market,quantile)
    if mask.sum() < max(30,values.shape[1]+2):
        raise ValueError("Insufficient stressed observations for correlation calibration.")
    selected = values[mask]
    if (selected.std(axis=0) <= 0).any():
        raise ValueError("Stress-window correlation is undefined for a constant series.")
    correlation = np.atleast_2d(np.corrcoef(selected,rowvar=False))
    vol = np.sqrt(np.diag(baseline))
    return validate_covariance(correlation * np.outer(vol,vol)), int(mask.sum())


def replay_attribution(returns, weights, capital, rebalance=0, costs=None, components=None):
    """Absolute P&L contributions on changing positions, plus execution costs."""
    dates = returns.index
    results, factor_results = {}, {}
    for strategy,w in weights.items():
        held = w*0
        nav = capital
        pnl = w*0
        spent = 0.
        factor_pnl = np.zeros(len(components[0])) if components else None
        for step,(_, shocks) in enumerate(returns.iterrows()):
            if step == 0 or (rebalance and step % rebalance == 0):
                budget = nav
                if costs is not None:
                    budget, charge, _ = fund_orders(nav,w,held,pd.Series(1.,index=w.index),costs)
                    spent += charge
                held = budget*w
            if components:
                factor_pnl += np.array([held @ series.iloc[step] for series in components[1]])
            delta = held*shocks
            pnl += delta
            held += delta
            nav = held.sum()
        results[strategy] = pd.concat([pnl,pd.Series({"Transaction Costs":-spent})])/capital
        if components:
            factor_results[strategy] = pd.Series([*factor_pnl, -spent], index=[*components[0], "Transaction Costs"])/capital
    paths = nav_paths(returns.to_numpy()[None,:,:], weights.to_numpy(),capital,rebalance,costs)[0]
    table = pd.DataFrame(results)
    np.testing.assert_allclose(table.sum(),paths[-1]/capital-1,atol=1e-10)
    factors = pd.DataFrame(factor_results)
    if components:
        np.testing.assert_allclose(factors.sum(), table.sum(), atol=1e-10)
    return pd.DataFrame(paths,index=pd.Index(["Initial",*dates.strftime("%Y-%m-%d")],name="Date"),columns=weights.columns),table,factors


def paired_simulation(weights, mean, beta, factors, residuals, delta, *, capital, paths, seed,
                      method, decay, sessions, rf, rebalance=0, costs=None, factor_multiplier=1.,
                      residual_multiplier=1., historical_correlations=False, active=None):
    steps, n_factors = delta.shape
    n_assets = len(weights)
    if paths*(steps+1)*len(weights.columns) > 16_000_000 or min(paths, steps) < 1:
        raise ValueError("Reduce the horizon, number of portfolios or simulations to fit the memory budget.")
    if costs and paths*len(weights.columns)*((steps-1)//(rebalance or steps)+1)>6000:
        raise ValueError("Reduce the number of paths or use a longer interval for transaction-cost simulation.")
    factor_cov = validate_covariance(covariance(factors,method,decay))
    residual_cov = validate_covariance(covariance(residuals,method,decay))
    stress_cov, observations = (stressed_covariance(factors,factors[:,0],factor_cov)
                               if historical_correlations else (factor_cov,None))
    def root(matrix):
        eig, vec = np.linalg.eigh(matrix)
        return (vec*np.sqrt(np.maximum(eig,0))) @ vec.T

    base_root, stress_root, residual_root = map(root,(factor_cov,stress_cov,residual_cov))
    rng = np.random.default_rng(seed)
    base = np.empty((paths,steps+1,len(weights.columns)))
    stress = np.empty_like(base)
    active = np.asarray(active, dtype=bool) if active is not None else np.any(delta != 0,axis=1)
    if active.shape != (steps,) or min(factor_multiplier, residual_multiplier) <= 0:
        raise ValueError("Invalid stress regime.")
    batch = max(1,min(128,1_000_000//(steps*n_assets)))
    for start in range(0,paths,batch):
        stop = min(start+batch,paths)
        f = rng.standard_normal((stop-start,steps,n_factors))
        e = rng.standard_normal((stop-start,steps,n_assets)) @ residual_root
        ff = f @ base_root
        stressed = ff.copy()
        stressed[:,active] = (f[:,active] @ stress_root)*factor_multiplier
        residual = e.copy()
        residual[:,active] *= residual_multiplier
        baseline = mean+ff @ beta.T+e
        adverse = mean+(stressed+delta) @ beta.T+residual
        base[start:stop] = nav_paths(baseline,weights.to_numpy(),capital,rebalance,costs)
        stress[start:stop] = nav_paths(adverse,weights.to_numpy(),capital,rebalance,costs)
    b = summarize(base,weights.columns,capital,sessions,rf)
    s = summarize(stress,weights.columns,capital,sessions,rf)
    return b,s,dict(factor_cov=factor_cov,residual_cov=residual_cov,stress_correlation_cov=stress_cov,
                    stressed_observations=observations)
