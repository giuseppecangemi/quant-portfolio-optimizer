import numpy as np
import pandas as pd


def annualized_volatility(
    returns: pd.DataFrame,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calcola la volatilità annualizzata di ogni asset.
    """

    return returns.std() * (periods_per_year ** 0.5)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la matrice di correlazione tra gli asset.
    """

    return returns.corr()


def covariance_matrix(
    returns: pd.DataFrame,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Calcola la matrice di covarianza annualizzata.
    """

    return returns.cov() * periods_per_year


def portfolio_return(
    weights: np.ndarray,
    expected_returns: pd.Series
) -> float:
    """
    Calcola il rendimento atteso del portafoglio.
    """

    return float(np.dot(weights, expected_returns.values))


def portfolio_volatility(
    weights: np.ndarray,
    covariance: pd.DataFrame
) -> float:
    """
    Calcola la volatilità annualizzata del portafoglio.
    """

    return float(
        np.sqrt(
            weights.T
            @ covariance.values
            @ weights
        )
    )