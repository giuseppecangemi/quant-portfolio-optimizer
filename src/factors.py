import numpy as np
import pandas as pd


def calculate_capm(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Stima CAPM per ogni asset usando rendimenti giornalieri allineati."""
    if asset_returns.empty:
        raise ValueError("Il DataFrame dei rendimenti degli asset è vuoto.")

    market_returns = market_returns.squeeze().rename("Market")
    daily_rf = risk_free_rate / periods_per_year
    rows = []

    for ticker in asset_returns.columns:
        aligned = pd.concat(
            [asset_returns[ticker].rename("Asset"), market_returns],
            axis=1,
            join="inner",
        ).dropna()

        if len(aligned) < 2:
            raise ValueError(f"Dati insufficienti per stimare il CAPM di {ticker}.")

        asset_excess = aligned["Asset"] - daily_rf
        market_excess = aligned["Market"] - daily_rf
        market_variance = float(market_excess.var())

        if market_variance <= 0 or np.isnan(market_variance):
            raise ValueError("La varianza del market proxy non è valida.")

        beta = float(asset_excess.cov(market_excess) / market_variance)
        alpha_daily = float(asset_excess.mean() - beta * market_excess.mean())
        alpha_annualized = alpha_daily * periods_per_year

        correlation = float(aligned["Asset"].corr(aligned["Market"]))
        r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0

        historical_return = float(aligned["Asset"].mean() * periods_per_year)
        market_expected_return = float(aligned["Market"].mean() * periods_per_year)
        capm_expected_return = float(
            risk_free_rate
            + beta * (market_expected_return - risk_free_rate)
        )

        rows.append(
            {
                "Ticker": ticker,
                "Beta": beta,
                "Alpha": alpha_annualized,
                "R Squared": r_squared,
                "Market Correlation": correlation,
                "Historical Return": historical_return,
                "CAPM Expected Return": capm_expected_return,
            }
        )

    return pd.DataFrame(rows).set_index("Ticker")


def capm_expected_returns(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Restituisce il vettore mu_CAPM da passare all'ottimizzatore."""
    results = calculate_capm(
        asset_returns,
        market_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    return results["CAPM Expected Return"].copy()
