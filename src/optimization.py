import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.risk import portfolio_return, portfolio_volatility


def optimize_minimum_volatility(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    max_weight: float = 1.0
) -> pd.Series:
    """
    Trova il portafoglio con la minima volatilità.

    max_weight rappresenta il peso massimo consentito
    per ogni singolo asset.
    """

    n_assets = len(expected_returns)

    initial_weights = np.ones(n_assets) / n_assets

    if max_weight < 1 / n_assets:
        raise ValueError(
            "Il peso massimo per asset è troppo basso "
            "rispetto al numero di asset."
        )

    bounds = [(0.0, max_weight)] * n_assets

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    result = minimize(
        portfolio_volatility,
        initial_weights,
        args=(covariance,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(
            f"Ottimizzazione fallita: {result.message}"
        )

    return pd.Series(
        result.x,
        index=expected_returns.index
    )


def optimize_maximum_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0
) -> pd.Series:
    """
    Trova il portafoglio con il massimo Sharpe Ratio.

    max_weight rappresenta il peso massimo consentito
    per ogni singolo asset.
    """

    n_assets = len(expected_returns)

    initial_weights = np.ones(n_assets) / n_assets

    if max_weight < 1 / n_assets:
        raise ValueError(
            "Il peso massimo per asset è troppo basso "
            "rispetto al numero di asset."
        )

    bounds = [(0.0, max_weight)] * n_assets

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    def negative_sharpe(weights):
        portfolio_ret = portfolio_return(
            weights,
            expected_returns
        )

        portfolio_vol = portfolio_volatility(
            weights,
            covariance
        )

        if portfolio_vol == 0:
            return 0

        sharpe = (
            portfolio_ret - risk_free_rate
        ) / portfolio_vol

        return -sharpe

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(
            f"Ottimizzazione fallita: {result.message}"
        )

    return pd.Series(
        result.x,
        index=expected_returns.index
    )


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    points: int = 100,
    max_weight: float = 1.0
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

    min_volatility_weights = optimize_minimum_volatility(
        expected_returns,
        covariance,
        max_weight=max_weight
    )

    min_volatility_return = portfolio_return(
        min_volatility_weights.values,
        expected_returns
    )

    min_return = expected_returns.min()
    max_return = expected_returns.max()

    target_returns = np.linspace(
        min_return,
        max_return,
        points
    )

    initial_weights = np.ones(n_assets) / n_assets
    bounds = [(0.0, max_weight)] * n_assets

    frontier = []

    for target_return in target_returns:

        constraints = [
            {
                "type": "eq",
                "fun": lambda weights: np.sum(weights) - 1
            },
            {
                "type": "eq",
                "fun": lambda weights, target=target_return:
                    portfolio_return(
                        weights,
                        expected_returns
                    ) - target
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

            volatility = portfolio_volatility(
                result.x,
                covariance
            )

            frontier.append({
                "return": target_return,
                "volatility": volatility,
                "efficient": target_return >= min_volatility_return
            })

    return pd.DataFrame(frontier)
