import pandas as pd


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i rendimenti giornalieri semplici.

    Esempio:
    se un'azione passa da 100 a 102,
    il rendimento è 2%.
    """

    returns = prices.pct_change()

    return returns.dropna()


def annualized_returns(
    returns: pd.DataFrame,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calcola il rendimento medio annuo stimato
    a partire dai rendimenti giornalieri.
    """

    return returns.mean() * periods_per_year