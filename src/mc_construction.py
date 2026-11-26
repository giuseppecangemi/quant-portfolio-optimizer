"""Equity FFR and scenario optimization, based on Vorobets (2026-10-22)."""
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.special import entr, logsumexp


STRATEGIES = ("Minimum CVaR", "Minimum CML", "CVaR Constrained")
STATES = ("Universe realized volatility", "Universe momentum")


def equity_logs(prices):
    values = prices.to_numpy(dtype=float)
    if (not isinstance(prices.index, pd.DatetimeIndex)
            or not prices.index.is_monotonic_increasing or not prices.index.is_unique
            or not prices.columns.is_unique or min(prices.shape) < 1
            or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError("Provide positive aligned daily prices with unique dates and assets.")
    return np.log(prices).diff().iloc[1:]


def state_variables(logs, names):
    daily = logs.mean(axis=1)
    available = {
        STATES[0]: daily.rolling(21).std(),
        STATES[1]: daily.rolling(21).sum(),
    }
    return pd.DataFrame({name: available[name] for name in names}).dropna()


def entropy_posterior(prior, state, sample):
    """Minimum KL under equality on mean and an upper second moment."""
    scale = state.std()
    if scale <= 0:
        raise ValueError("Constant state variable.")
    z = (state - state.mean()) / scale
    matrix = np.column_stack((z, z*z))
    target = matrix[sample].mean(axis=0)
    logp = np.log(prior)

    def dual(theta):
        logits = logp + matrix @ theta
        normalizer = logsumexp(logits)
        q = np.exp(logits - normalizer)
        return normalizer - target @ theta, q @ matrix - target

    fit = minimize(dual, [0., 0.], jac=True, method="L-BFGS-B",
                   bounds=[(None, None), (None, 0.)],
                   options={"gtol": 1e-9, "ftol": 1e-14})
    theta = fit.x
    if not np.isfinite(theta).all() or theta[1] > 0:
        raise ValueError(f"Entropy Pooling returned invalid dual coefficients: {fit.message}")
    logits = logp + matrix @ theta
    q = np.exp(logits - logsumexp(logits))
    error = q @ matrix - target
    # Exponential tilting enforces stationarity of the primal Lagrangian.
    # Independently check primal feasibility and complementarity of the
    # second-moment inequality, even when the line search reports a failure.
    tolerance = 1e-6
    complementarity = abs(theta[1] * error[1])
    if (not np.isfinite(q).all() or abs(q.sum()-1.) > 1e-10
            or abs(error[0]) > tolerance or error[1] > tolerance
            or complementarity > tolerance):
        raise ValueError(
            f"Entropy Pooling failed certification: {fit.message}; "
            f"moment residuals={error}; complementarity={complementarity:.3g}"
        )
    return q


def calibrate_ffr(logs, states, half_life, cuts=(.25, .75)):
    if (not logs.index.is_unique or not states.index.is_unique
            or not states.index.is_monotonic_increasing or not states.index.isin(logs.index).all()
            or not np.isfinite(states.to_numpy()).all()):
        raise ValueError("State observations must be finite and aligned with calibrated log returns.")
    logs = logs.loc[states.index]
    if (not np.isfinite(logs.to_numpy()).all() or len(logs) < 80 or not 1 <= states.shape[1] <= 2 or not np.isfinite(half_life)
            or half_life < 1 or len(cuts) != 2 or not 0 < cuts[0] < cuts[1] < 1):
        raise ValueError("FFR requires ≥80 aligned observations and valid half-life and quantiles.")
    prior_log = -np.log(2) * np.arange(len(logs)-1, -1, -1) / half_life
    prior = np.exp(prior_log - logsumexp(prior_log))
    if (prior <= 0).any():
        raise ValueError("Half-life is too short for this history.")
    bins, marginals = [], []
    for state in states.to_numpy().T:
        boundaries = np.quantile(state, cuts)
        membership = np.searchsorted(boundaries, state, side="left")
        if np.bincount(membership, minlength=3).min() < 5:
            raise ValueError("A marginal state has fewer than five observations.")
        bins.append(membership)
        marginals.append([entropy_posterior(prior, state, membership == j) for j in range(3)])
    combinations = list(product(range(3), repeat=len(bins)))
    posterior = []
    for indices in combinations:
        vectors = np.array([marginals[m][j] for m, j in enumerate(indices)])
        if len(vectors) == 1:
            posterior.append(vectors[0])
            continue
        distance = np.linalg.norm(np.sqrt(vectors[0]) - np.sqrt(vectors[1])) / np.sqrt(2)
        diversity = np.array([distance, distance])
        weights = np.exp(entr(vectors).sum(axis=1)) * diversity
        posterior.append(np.average(vectors, axis=0, weights=weights) if distance > 1e-12 else vectors[0])
    q = np.asarray(posterior)
    membership = np.ravel_multi_index(bins, (3,) * len(bins))
    transition = q @ np.eye(len(combinations))[membership]
    if (not np.isfinite(q).all() or (q < 0).any()
            or not np.allclose(q.sum(axis=1), 1.) or not np.allclose(transition.sum(axis=1), 1.)):
        raise ValueError("Invalid posterior or transition probabilities.")
    return dict(logs=logs, prior=prior, q=q, membership=membership, transition=transition)


def draw_paths(calibration, simulations, horizon, seed):
    logs = calibration["logs"]
    if min(simulations, horizon) < 1 or simulations * horizon * logs.shape[1] > 8_000_000:
        raise ValueError("The requested scenario tensor exceeds the supported size.")
    rng = np.random.default_rng(seed)
    current = np.full(simulations, calibration["membership"][-1])
    picks = np.empty((simulations, horizon), dtype=int)
    state_paths = np.empty_like(picks)
    for h in range(horizon):
        for state in np.unique(current):
            mask = current == state
            picks[mask, h] = rng.choice(len(logs), mask.sum(), p=calibration["q"][state])
        current = calibration["membership"][picks[:, h]]
        state_paths[:, h] = current
    paths = np.expm1(np.cumsum(logs.to_numpy()[picks].transpose(0, 2, 1), axis=2))
    if not np.isfinite(paths).all() or (paths <= -1).any():
        raise ValueError("Invalid reconstructed equity paths.")
    return paths, state_paths


def tail(loss, alpha):
    loss = np.asarray(loss, dtype=float)
    if loss.ndim != 1 or not len(loss) or not np.isfinite(loss).all() or not 0 < alpha < 1:
        raise ValueError("Tail statistics require finite losses and a confidence in (0, 1).")
    mass = (1-alpha) * len(loss)
    ordinal = min(int(np.ceil(mass))-1, len(loss)-1)
    threshold = np.sort(loss)[::-1][ordinal]
    above = loss > threshold
    at = loss == threshold
    weights = (above + at * max(0., mass-above.sum()) / at.sum()) / mass
    return threshold, weights @ loss


def cvar_contributions(paths, weights, alpha):
    terminal_losses = -paths[:, :, -1]
    loss = terminal_losses @ weights
    threshold, _ = tail(loss, alpha)
    above = loss > threshold
    at = loss == threshold
    mass = (1-alpha)*len(loss)
    tail_weights = (above + at*max(0., mass-above.sum())/at.sum())/mass
    marginal = tail_weights @ terminal_losses
    return marginal, weights * marginal


def scenario_allocation(paths, strategy, alpha, cap, budget=None, *, assets=None):
    """Linear programs on cumulative simple P&L relative to initial capital."""
    if (paths.ndim != 3 or min(paths.shape) < 1 or not np.isfinite(paths).all()
            or strategy not in STRATEGIES or not 0 < alpha < 1 or not 0 < cap <= 1):
        raise ValueError("Invalid scenarios or allocation parameters.")
    s, n, h = paths.shape
    if n * cap < 1 - 1e-12:
        raise ValueError("Infeasible long-only cap.")
    cml = strategy == "Minimum CML"
    terminal = paths[:, :, -1]
    loss = -paths.transpose(0, 2, 1).reshape(s*h, n) if cml else -terminal
    rows = len(loss)
    ids = np.repeat(np.arange(s), h) if cml else np.arange(s)
    excess = csr_matrix((-np.ones(rows), (np.arange(rows), ids)), shape=(rows, s))
    lhs = hstack((csr_matrix(loss), csr_matrix(-np.ones((rows, 1))), excess), format="csr")
    rhs = np.zeros(rows)
    objective = np.r_[np.zeros(n), 1, np.full(s, 1/((1-alpha)*s))]
    if strategy == "CVaR Constrained":
        if budget is None or not np.isfinite(budget):
            raise ValueError("Set a finite horizon CVaR budget.")
        lhs = vstack((lhs, csr_matrix(objective)), format="csr")
        rhs = np.r_[rhs, budget]
        objective = np.r_[-terminal.mean(axis=0), np.zeros(s+1)]
    bounds = [(0, cap)] * n
    if assets is not None:
        from src.optimization import weight_bounds

        bounds = list(zip(*weight_bounds(assets, cap)))
    fit = linprog(objective, A_ub=lhs, b_ub=rhs, A_eq=[np.r_[np.ones(n), np.zeros(s+1)]],
                  b_eq=[1], bounds=bounds + [(0 if cml else None, None)] + [(0, None)]*s,
                  method="highs")
    if not fit.success:
        raise ValueError(f"Scenario optimization failed ({fit.status}): {fit.message}")
    weights = fit.x[:n]
    if not np.isfinite(weights).all() or abs(weights.sum()-1) > 1e-7 or (
            weights.min() < -1e-7 or weights.max() > cap+1e-7):
        raise ValueError("Solver returned infeasible portfolio weights.")
    var, cvar = tail(-(terminal @ weights), alpha)
    maximum_loss = np.maximum(0, -(paths.transpose(0, 2, 1) @ weights).min(axis=1))
    cml_value = tail(maximum_loss, alpha)[1]
    expected = float(terminal.mean(axis=0) @ weights)
    objective_check = {"Minimum CVaR": cvar, "Minimum CML": cml_value, "CVaR Constrained": -expected}
    if abs(fit.fun - objective_check[strategy]) > 1e-6 or (
            strategy == "CVaR Constrained" and cvar > budget+1e-7):
        raise ValueError("Solver objective or CVaR budget failed independent verification.")
    return weights, dict(expected_return=expected, volatility=float((terminal @ weights).std()),
                         var=var, cvar=cvar, cml=cml_value,
                         probability_loss=float(((terminal @ weights) < 0).mean()), solver=fit.message)


def fit_construction(prices, config, rf, cap, cache=None):
    from src.mc_coverage import MIN_SESSIONS, STRICT, assess_coverage

    if not prices.columns.is_unique:
        raise ValueError("Calibration contains duplicate asset identifiers.")
    policy = config.get("coverage_policy", STRICT)
    candidates = {ticker: prices[ticker] for ticker in prices}
    prices, report = assess_coverage(
        candidates, prices.index, len(prices)-1, policy,
        config.get("coverage_minimum", MIN_SESSIONS), selected=config.get("coverage_selection"),
    )
    if prices.empty or prices.shape[1] * cap < 1 - 1e-12:
        reasons = report.loc[~report["Eligible"], ["Asset", "Reason"]]
        raise ValueError(f"Coverage leaves too few assets for the portfolio cap. {reasons.to_dict('records')}")
    key = ("construction", tuple(prices.columns), prices.index[0], prices.index[-1],
           tuple(config["states"]), config["half_life"], tuple(config["cuts"]),
           config["simulations"], config["horizon"], config["seed"])
    cache = {} if cache is None else cache
    if key not in cache:
        logs = equity_logs(prices)
        calibrated = calibrate_ffr(logs, state_variables(logs, config["states"]), config["half_life"], config["cuts"])
        old_keys = [item for item in cache if item[0] == "construction"]
        if len(old_keys) >= 4:
            del cache[old_keys[0]]
        cache[key] = draw_paths(calibrated, config["simulations"], config["horizon"], config["seed"])[0]
    paths = cache[key]
    w, _ = scenario_allocation(paths, config["strategy"], config["alpha"], cap, config["budget"], assets=prices.columns)
    # Existing allocation cards expect annualized simple daily moments.
    cumulative = 1 + paths
    daily = np.diff(cumulative, axis=2, prepend=1.) / np.concatenate(
        (np.ones_like(cumulative[:, :, :1]), cumulative[:, :, :-1]), axis=2)
    daily = daily.transpose(0, 2, 1).reshape(-1, len(w))
    return (pd.Series(w, index=prices.columns), pd.Series(daily.mean(axis=0)*252, index=prices.columns),
            pd.DataFrame(np.cov(daily, rowvar=False, ddof=0).reshape(len(w), len(w))*252,
                         index=prices.columns, columns=prices.columns))
