from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from src.portfolio_context import minimum_for
from src.risk import portfolio_return, portfolio_volatility


def _validated_covariance(covariance, labels):
    if (len(labels) == 0 or not labels.is_unique
            or not covariance.index.is_unique or not covariance.columns.is_unique
            or set(labels) != set(covariance.index) or set(labels) != set(covariance.columns)):
        raise ValueError("Covariance must have one row and column per asset")
    covariance = covariance.loc[labels, labels].astype(float)
    values = covariance.to_numpy()
    scale = np.max(np.abs(values))
    if not np.isfinite(scale):
        raise ValueError("Covariance must be finite")
    if scale > 0:
        scaled = values / scale
        if not np.allclose(scaled, scaled.T, rtol=0, atol=1e-10):
            raise ValueError("Covariance must be symmetric")
        if np.linalg.eigvalsh(scaled)[0] < -1e-10:
            raise ValueError("Covariance must be positive semidefinite")
    return covariance, scale


def weight_bounds(labels, max_weight, minimum_weights=None):
    minimum_weights = minimum_for(labels, minimum_weights)
    minimum = pd.Series(dtype=float) if minimum_weights is None else pd.Series(minimum_weights, dtype=float)
    if not minimum.index.is_unique or not minimum.index.isin(labels).all():
        raise ValueError("Minimum weights must refer to unique tickers in the universe.")
    lower = minimum.reindex(labels, fill_value=0.).to_numpy()
    upper = np.full(len(labels), min(max_weight, 1.))
    if (not np.isfinite(max_weight) or not np.isfinite(lower).all() or (lower < 0).any()
            or (lower > upper).any() or lower.sum() > 1 + 1e-12 or upper.sum() < 1 - 1e-12):
        raise ValueError("Infeasible individual weight bounds.")
    return lower, upper


def _linear_weights(coefficients, lower, upper):
    order = np.argsort(-np.asarray(coefficients), kind="stable")
    capacity = (upper - lower)[order]
    remaining = 1 - lower.sum() - np.r_[0., capacity.cumsum()[:-1]]
    weights = lower.copy()
    weights[order] += np.clip(remaining, 0., capacity)
    return weights


def _check_weights(weights, lower, upper):
    if (not np.isfinite(weights).all() or abs(weights.sum()-1) > 1e-8
            or (weights < lower - 1e-8).any() or (weights > upper + 1e-8).any()):
        raise ValueError("Optimizer did not satisfy the individual weight constraints.")


def _bounded_vertices(lower, upper):
    """Every box-simplex vertex has at most one free coordinate."""
    cap = upper - lower
    budget = 1 - lower.sum()
    if budget <= 1e-12:
        yield lower.copy()
        return
    for free in range(len(cap)):
        if cap[free] <= 1e-12:
            continue
        others = [i for i in range(len(cap)) if i != free and cap[i] > 1e-12]
        suffix = np.r_[np.cumsum(cap[others][::-1])[::-1], 0.]
        increment = np.zeros(len(cap))

        def visit(offset, remaining):
            if remaining < -1e-12 or remaining > suffix[offset] + cap[free] + 1e-12:
                return
            if offset == len(others):
                increment[free] = np.clip(remaining, 0., cap[free])
                yield lower + increment
                return
            i = others[offset]
            increment[i] = 0.
            yield from visit(offset+1, remaining)
            increment[i] = cap[i]
            yield from visit(offset+1, remaining-cap[i])

        yield from visit(0, budget)


def optimize_minimum_volatility(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    max_weight: float = 1.0,
    *, minimum_weights=None
) -> pd.Series:
    """
    Trova il portafoglio con la minima volatilità.

    max_weight rappresenta il peso massimo consentito
    per ogni singolo asset.
    """

    covariance, scale = _validated_covariance(covariance, expected_returns.index)
    lower, upper = weight_bounds(expected_returns.index, max_weight, minimum_weights)
    initial_weights = (lower + (1-lower.sum())*(upper-lower)/(upper-lower).sum()
                       if (upper-lower).sum() > 0 else lower.copy())
    bounds = list(zip(lower, upper))
    if np.isclose(lower.sum(), 1., rtol=0, atol=1e-12):
        return pd.Series(lower, index=expected_returns.index)

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    if scale == 0:
        return pd.Series(initial_weights, index=expected_returns.index)
    matrix = covariance.to_numpy() / scale
    result = minimize(
        lambda weights: float(weights @ matrix @ weights),
        initial_weights,
        jac=lambda weights: 2 * (matrix @ weights),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12},
    )

    if not result.success:
        raise ValueError(
            f"Ottimizzazione fallita: {result.message}"
        )

    _check_weights(result.x, lower, upper)
    return pd.Series(
        result.x,
        index=expected_returns.index
    )


def _maximum_sharpe_at_vertex(excess, matrix, labels, max_weight):
    n_assets = len(labels)
    # For non-positive excess returns a vertex maximizes Sharpe: for
    # positive risk, the norm of a convex combination is at most the
    # combination of norms, while the nonnegative loss is linear.
    cap = min(max_weight, 1.0)
    full = min(n_assets, int(np.floor(1 / cap + 1e-12)))
    remainder = max(0.0, 1 - full * cap)
    count = comb(n_assets, full) * (n_assets - full if remainder > 1e-12 else 1)
    if count > 100_000:
        raise ValueError("Maximum Sharpe with nonpositive excess returns has too many feasible vertices for exact enumeration")
    best = None
    best_score = -np.inf
    for capped in combinations(range(n_assets), full):
        free = (set(range(n_assets)) - set(capped)) if remainder > 1e-12 else (None,)
        for last in free:
            weights = np.zeros(n_assets)
            weights[list(capped)] = cap
            if last is not None:
                weights[last] = remainder
            score = float(weights @ excess / np.sqrt(weights @ matrix @ weights))
            if score > best_score:
                best, best_score = weights, score
    return pd.Series(best, index=labels)


def optimize_maximum_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    *, minimum_weights=None
) -> pd.Series:
    """
    Trova il portafoglio con il massimo Sharpe Ratio.

    max_weight rappresenta il peso massimo consentito
    per ogni singolo asset.
    """

    covariance, scale = _validated_covariance(covariance, expected_returns.index)
    excess = expected_returns.to_numpy(float) - risk_free_rate
    if not np.isfinite(excess).all():
        raise ValueError("Expected excess returns must be finite")
    n_assets = len(expected_returns)
    lower, upper = weight_bounds(expected_returns.index, max_weight, minimum_weights)
    bounds = list(zip(lower, upper))

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    if scale == 0:
        raise ValueError("Sharpe is undefined for a zero-risk covariance")
    matrix = covariance.to_numpy() / scale
    if np.linalg.eigvalsh(matrix)[0] <= 1e-10:
        min_risk = optimize_minimum_volatility(expected_returns, covariance, max_weight, minimum_weights=minimum_weights).to_numpy()
        if min_risk @ matrix @ min_risk <= 1e-12:
            raise ValueError("Sharpe is undefined on a feasible zero-risk portfolio")

    initial_weights = _linear_weights(excess, lower, upper)
    max_excess = initial_weights @ excess
    if max_excess == 0 or np.isclose(lower.sum(), 1., rtol=0, atol=1e-12):
        return pd.Series(initial_weights, index=expected_returns.index)
    if max_excess < 0:
        if not lower.any():
            return _maximum_sharpe_at_vertex(excess, matrix, expected_returns.index, max_weight)
        best, best_score = None, -np.inf
        for count, weights in enumerate(_bounded_vertices(lower, upper), start=1):
            if count > 100_000:
                raise ValueError("Too many feasible vertices to certify nonpositive-excess Maximum Sharpe.")
            score = weights @ excess / np.sqrt(weights @ matrix @ weights)
            if score > best_score:
                best, best_score = weights, score
        if best is None:
            raise ValueError("No feasible vertex for Maximum Sharpe.")
        return pd.Series(best, index=expected_returns.index)

    normalized_excess = excess / np.max(np.abs(excess))
    constraints = [constraints, {
        "type": "ineq",
        "fun": lambda weights: float(weights @ normalized_excess),
        "jac": lambda weights: normalized_excess,
    }]

    def negative_sharpe(weights):
        return -float(weights @ normalized_excess / np.sqrt(weights @ matrix @ weights))

    def gradient(weights):
        marginal = matrix @ weights
        variance = weights @ marginal
        return ((weights @ normalized_excess) * marginal / variance - normalized_excess) / np.sqrt(variance)

    result = minimize(
        negative_sharpe,
        initial_weights,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12},
    )

    def certification_gap(weights):
        if (not np.isfinite(weights).all() or abs(weights.sum() - 1) > 1e-8
                or (weights < lower - 1e-8).any() or (weights > upper + 1e-8).any()
                or weights @ normalized_excess <= 0 or weights @ matrix @ weights <= 0):
            return np.inf
        # A linear oracle checks the first-order condition on the entire capped simplex.
        grad = gradient(weights)
        gap = grad @ (weights - _linear_weights(-grad, lower, upper))
        return gap / max(1., abs(negative_sharpe(weights)))

    def certified(weights):
        return certification_gap(weights) <= 1e-7

    if certified(result.x):
        return pd.Series(result.x, index=expected_returns.index)

    # Normalize excess return to one. The resulting convex problem minimizes
    # variance with linear constraints and recovers the original weights by normalization.
    target = normalized_excess @ initial_weights
    cap_matrix = upper[:, None] - np.eye(n_assets)
    floor_matrix = np.eye(n_assets) - lower[:, None]
    box_matrix = np.vstack([cap_matrix, floor_matrix])
    transformed = minimize(
        lambda x: float(x @ matrix @ x),
        initial_weights,
        jac=lambda x: 2 * (matrix @ x),
        method="SLSQP",
        bounds=[(0., None)] * n_assets,
        constraints=[
            {"type": "eq", "fun": lambda x: normalized_excess @ x - target,
             "jac": lambda x: normalized_excess},
            {"type": "ineq", "fun": lambda x: box_matrix @ x,
             "jac": lambda x: box_matrix},
        ],
        options={"ftol": 1e-13, "maxiter": 500},
    )
    convex_gap = np.inf
    if transformed.x.sum() > 0:
        weights = transformed.x / transformed.x.sum()
        convex_gap = certification_gap(weights)
        if convex_gap <= 1e-7:
            return pd.Series(weights, index=expected_returns.index)

    # Polish near-optimal iterates with exact feasible pairwise exchanges.
    # This avoids asking SLSQP to resolve tiny objective changes near a cap.
    candidates = [(certification_gap(result.x), result.x)]
    if transformed.x.sum() > 0:
        candidates.append((convex_gap, weights))
    gap, candidate = min(candidates, key=lambda entry: entry[0])
    if gap <= 1e-3:
        weights = candidate.copy()
        for _ in range(2000):
            if certified(weights):
                return pd.Series(weights, index=expected_returns.index)
            grad = gradient(weights)
            source = np.argmax(np.where(weights > lower + 1e-12, grad, -np.inf))
            destination = np.argmin(np.where(weights < upper - 1e-12, grad, np.inf))
            if grad[source] <= grad[destination]:
                break
            bound = min(weights[source]-lower[source], upper[destination]-weights[destination])
            marginal = matrix @ weights
            a = weights @ normalized_excess
            b = normalized_excess[destination] - normalized_excess[source]
            c = weights @ marginal
            d = marginal[destination] - marginal[source]
            e = matrix[source, source] + matrix[destination, destination] - 2 * matrix[source, destination]
            curvature = a * e - b * d
            step = min(bound, (b * c - a * d) / curvature) if curvature > 0 else bound
            if step <= 0:
                break
            weights[source] -= step
            weights[destination] += step

    raise ValueError(
        f"Maximum Sharpe failed certification. Ratio SLSQP status={result.status}, "
        f"iterations={result.nit}: {result.message}. Convex SLSQP "
        f"status={transformed.status}, iterations={transformed.nit}: {transformed.message}. "
        f"Certification gaps ratio={certification_gap(result.x):.3g}, convex={convex_gap:.3g}; "
        f"required <=1e-7. Initial weights={initial_weights.tolist()}."
    )


def optimize_risk_parity(
    covariance: pd.DataFrame,
    max_weight: float = 1.0,
    *, minimum_weights=None
) -> pd.Series:
    """
    Trova un portafoglio Equal Risk Contribution (Risk Parity).

    L'obiettivo è fare in modo che ogni asset contribuisca
    in misura il più possibile uguale al rischio complessivo
    del portafoglio.

    A differenza del Maximum Sharpe, Risk Parity non utilizza
    i rendimenti attesi. Utilizza esclusivamente la matrice
    di covarianza.

    max_weight rappresenta il peso massimo consentito
    per ogni singolo asset.
    """

    n_assets = len(covariance)
    covariance, scale = _validated_covariance(covariance, covariance.index)
    lower, upper = weight_bounds(covariance.index, max_weight, minimum_weights)
    if scale == 0:
        raise ValueError("Risk contributions are undefined for zero portfolio variance")
    covariance_values = covariance.to_numpy() / scale
    initial_weights = lower + (1-lower.sum())*(upper-lower)/(upper-lower).sum() if (upper-lower).sum() else lower
    bounds = list(zip(lower, upper))
    if minimum_weights is not None and (np.isclose(lower.sum(), 1., rtol=0, atol=1e-12)
                                        or np.isclose(upper.sum(), 1., rtol=0, atol=1e-12)):
        _check_weights(initial_weights, lower, upper)
        if initial_weights @ covariance_values @ initial_weights <= 0:
            raise ValueError("Risk contributions are undefined for zero portfolio variance")
        return pd.Series(initial_weights, index=covariance.index)


    constraints = {
        "type": "eq",
        "fun": lambda weights:
            np.sum(weights) - 1
    }

    target_risk_contribution = (
        np.ones(n_assets) / n_assets
    )

    def risk_parity_objective(weights):
        """
        Minimizza la distanza tra i contributi percentuali
        al rischio e il contributo target 1/N.
        """

        portfolio_variance = float(
            weights.T
            @ covariance_values
            @ weights
        )

        if portfolio_variance <= 0:
            return 1e10

        marginal_contribution = (
            covariance_values @ weights
        )

        asset_risk_contribution = (
            weights * marginal_contribution
        )

        percentage_risk_contribution = (
            asset_risk_contribution
            / portfolio_variance
        )

        return float(
            np.sum(
                (
                    percentage_risk_contribution
                    - target_risk_contribution
                ) ** 2
            )
        )

    result = minimize(
        risk_parity_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1000,
        }
    )

    if not result.success:
        raise ValueError(
            "Ottimizzazione Risk Parity fallita: "
            f"{result.message}"
        )

    weights = result.x
    if weights @ covariance_values @ weights <= 0:
        raise ValueError("Risk contributions are undefined for zero portfolio variance")

    if lower.any():
        _check_weights(weights, lower, upper)
        return pd.Series(weights, index=covariance.index)

    # Pulizia dei piccoli residui numerici.
    weights[
        np.abs(weights) < 1e-10
    ] = 0.0

    weights = (
        weights / weights.sum()
    )

    return pd.Series(
        weights,
        index=covariance.index
    )


def risk_contributions(
    weights: pd.Series,
    covariance: pd.DataFrame
) -> pd.Series:
    """
    Calcola il contributo percentuale di ogni asset
    alla varianza complessiva del portafoglio.

    La somma dei contributi è pari a 1.
    """

    aligned_weights = (
        weights
        .reindex(covariance.index)
        .fillna(0.0)
    )

    weights_values = (
        aligned_weights.values
    )

    covariance_values = (
        covariance.values
    )

    portfolio_variance = float(
        weights_values.T
        @ covariance_values
        @ weights_values
    )

    if portfolio_variance <= 0:
        return pd.Series(
            np.zeros(len(aligned_weights)),
            index=aligned_weights.index
        )

    marginal_contribution = (
        covariance_values
        @ weights_values
    )

    asset_risk_contribution = (
        weights_values
        * marginal_contribution
    )

    percentage_risk_contribution = (
        asset_risk_contribution
        / portfolio_variance
    )

    return pd.Series(
        percentage_risk_contribution,
        index=aligned_weights.index
    )



def _apply_max_weight(weights: pd.Series, max_weight: float) -> pd.Series:
    """
    Applica un limite massimo ai pesi e redistribuisce l'eccesso
    proporzionalmente tra gli asset che hanno ancora capacità.
    """

    weights = weights.astype(float).copy()
    n_assets = len(weights)

    if max_weight < 1 / n_assets:
        raise ValueError(
            "Il peso massimo per asset è troppo basso "
            "rispetto al numero di asset."
        )

    weights = weights / weights.sum()

    for _ in range(n_assets + 2):
        over = weights > max_weight + 1e-12

        if not over.any():
            break

        excess = float((weights[over] - max_weight).sum())
        weights.loc[over] = max_weight

        under = weights < max_weight - 1e-12

        if not under.any():
            break

        capacity_weights = weights.loc[under]

        if capacity_weights.sum() > 0:
            addition = (
                excess
                * capacity_weights
                / capacity_weights.sum()
            )
        else:
            capacity = max_weight - weights.loc[under]
            addition = excess * capacity / capacity.sum()

        addition = np.minimum(
            addition.values,
            (max_weight - weights.loc[under]).values,
        )

        weights.loc[under] += addition

        remaining = 1.0 - weights.sum()

        if remaining > 1e-12:
            capacity = max_weight - weights
            capacity = capacity.clip(lower=0.0)

            if capacity.sum() > 0:
                weights += remaining * capacity / capacity.sum()

    weights[weights.abs() < 1e-12] = 0.0
    return weights / weights.sum()


def _cluster_variance(covariance: pd.DataFrame, cluster_items: list[str]) -> float:
    """Calcola la varianza del cluster usando pesi inverse-variance."""

    cluster_cov = covariance.loc[cluster_items, cluster_items]
    diagonal = np.diag(cluster_cov.values)
    diagonal = np.where(diagonal <= 0, 1e-12, diagonal)

    inverse_variance = 1.0 / diagonal
    inverse_variance /= inverse_variance.sum()

    return float(
        inverse_variance.T
        @ cluster_cov.values
        @ inverse_variance
    )


def hrp_cluster_order(covariance: pd.DataFrame) -> list[str]:
    """
    Restituisce l'ordine degli asset ottenuto dal clustering
    gerarchico usato da HRP.
    """

    if covariance.empty:
        raise ValueError("La matrice di covarianza è vuota.")

    if len(covariance) == 1:
        return covariance.index.tolist()

    std = np.sqrt(np.diag(covariance.values))
    denominator = np.outer(std, std)

    correlation = np.divide(
        covariance.values,
        denominator,
        out=np.zeros_like(covariance.values, dtype=float),
        where=denominator > 0,
    )

    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)

    distance = np.sqrt(
        np.maximum(0.0, (1.0 - correlation) / 2.0)
    )
    np.fill_diagonal(distance, 0.0)

    condensed_distance = squareform(
        distance,
        checks=False,
    )

    linkage_matrix = linkage(
        condensed_distance,
        method="single",
    )

    order = leaves_list(linkage_matrix)

    return covariance.index[order].tolist()


def optimize_hrp(
    covariance: pd.DataFrame,
    max_weight: float = 1.0
) -> pd.Series:
    """
    Costruisce un portafoglio Hierarchical Risk Parity (HRP).

    HRP usa la struttura di correlazione per raggruppare gli asset,
    ordina gli strumenti secondo il clustering gerarchico e assegna
    ricorsivamente il capitale in funzione del rischio dei cluster.
    Non utilizza i rendimenti attesi.
    """

    n_assets = len(covariance)

    if n_assets == 0:
        raise ValueError("La matrice di covarianza è vuota.")

    if max_weight < 1 / n_assets:
        raise ValueError(
            "Il peso massimo per asset è troppo basso "
            "rispetto al numero di asset."
        )

    if n_assets == 1:
        return pd.Series(1.0, index=covariance.index)

    ordered_assets = hrp_cluster_order(covariance)
    weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]

    while clusters:
        next_clusters = []

        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            split = len(cluster) // 2
            left_cluster = cluster[:split]
            right_cluster = cluster[split:]

            left_variance = _cluster_variance(
                covariance,
                left_cluster,
            )
            right_variance = _cluster_variance(
                covariance,
                right_cluster,
            )

            total_variance = left_variance + right_variance

            if total_variance <= 0:
                alpha = 0.5
            else:
                alpha = 1.0 - (
                    left_variance / total_variance
                )

            weights.loc[left_cluster] *= alpha
            weights.loc[right_cluster] *= (1.0 - alpha)

            next_clusters.extend(
                [left_cluster, right_cluster]
            )

        clusters = next_clusters

    weights = weights.reindex(covariance.index)
    weights = _apply_max_weight(weights, max_weight)

    return weights

def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    points: int = 100,
    max_weight: float = 1.0,
    *, minimum_weights=None
) -> pd.DataFrame:
    """
    Calcola la frontiera di Markowitz.

    max_weight rappresenta il peso massimo consentito
    per ogni singolo asset.
    """

    n_assets = len(expected_returns)

    if max_weight < 1 / n_assets:
        raise ValueError(
            "Il peso massimo per asset è troppo basso "
            "rispetto al numero di asset."
        )

    min_volatility_weights = (
        optimize_minimum_volatility(
            expected_returns,
            covariance,
            max_weight=max_weight, minimum_weights=minimum_weights
        )
    )

    min_volatility_return = (
        portfolio_return(
            min_volatility_weights.values,
            expected_returns
        )
    )

    min_return = expected_returns.min()
    max_return = expected_returns.max()

    target_returns = np.linspace(
        min_return,
        max_return,
        points
    )

    initial_weights = (
        np.ones(n_assets) / n_assets
    )

    bounds = [
        (0.0, max_weight)
        for _ in range(n_assets)
    ]

    if minimum_for(expected_returns.index, minimum_weights):
        lower, upper = weight_bounds(expected_returns.index, max_weight, minimum_weights)
        bounds = list(zip(lower, upper))
        initial_weights = min_volatility_weights.to_numpy()
        max_feasible = expected_returns @ _linear_weights(expected_returns, lower, upper)
        if np.isclose(max_feasible, min_volatility_return, rtol=1e-10, atol=1e-12):
            return pd.DataFrame({"return": [min_volatility_return],
                                 "volatility": [portfolio_volatility(initial_weights, covariance)],
                                 "efficient": [True]})
        target_returns = np.linspace(min_volatility_return, max_feasible, points)

    frontier = []

    for target_return in target_returns:

        constraints = [
            {
                "type": "eq",
                "fun": lambda weights:
                    np.sum(weights) - 1
            },
            {
                "type": "eq",
                "fun": (
                    lambda weights,
                    target=target_return:
                        portfolio_return(
                            weights,
                            expected_returns
                        ) - target
                )
            }
        ]

        result = minimize(
            portfolio_volatility,
            initial_weights,
            args=(covariance,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        if result.success:

            volatility = (
                portfolio_volatility(
                    result.x,
                    covariance
                )
            )

            frontier.append( 
                {
                    "return":
                        target_return,

                    "volatility":
                        volatility,

                    "efficient":
                        target_return
                        >= min_volatility_return
                }
            )

    return pd.DataFrame(frontier)