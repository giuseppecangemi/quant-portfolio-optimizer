import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    min_volatility_weights: pd.Series,
    max_sharpe_weights: pd.Series,
    risk_free_rate: float = 0.0
) -> go.Figure:
    """
    Crea il grafico completo della frontiera di Markowitz.

    Mostra:
    - parte inefficiente della frontiera
    - parte efficiente della frontiera
    - Minimum Volatility
    - Maximum Sharpe
    - singoli asset
    """

    fig = go.Figure()

    # --------------------------------------------------
    # Inefficient Frontier
    # --------------------------------------------------

    inefficient = frontier[
        frontier["efficient"] == False
    ]

    fig.add_trace(
        go.Scatter(
            x=inefficient["volatility"],
            y=inefficient["return"],
            mode="lines",
            name="Inefficient Frontier",
            line=dict(
                dash="dash"
            ),
            hovertemplate=(
                "Return: %{y:.2%}<br>"
                "Volatility: %{x:.2%}"
                "<extra></extra>"
            )
        )
    )

    # --------------------------------------------------
    # Efficient Frontier
    # --------------------------------------------------

    efficient = frontier[
        frontier["efficient"] == True
    ]

    fig.add_trace(
        go.Scatter(
            x=efficient["volatility"],
            y=efficient["return"],
            mode="lines",
            name="Efficient Frontier",
            hovertemplate=(
                "Return: %{y:.2%}<br>"
                "Volatility: %{x:.2%}"
                "<extra></extra>"
            )
        )
    )

    # --------------------------------------------------
    # Portfolio metrics
    # --------------------------------------------------

    min_return = float(
        np.dot(
            min_volatility_weights.values,
            expected_returns.values
        )
    )

    min_risk = float(
        np.sqrt(
            min_volatility_weights.values.T
            @ covariance.values
            @ min_volatility_weights.values
        )
    )

    min_sharpe = (
        (min_return - risk_free_rate) / min_risk
        if min_risk > 0
        else 0
    )

    max_return = float(
        np.dot(
            max_sharpe_weights.values,
            expected_returns.values
        )
    )

    max_risk = float(
        np.sqrt(
            max_sharpe_weights.values.T
            @ covariance.values
            @ max_sharpe_weights.values
        )
    )

    max_sharpe = (
        (max_return - risk_free_rate) / max_risk
        if max_risk > 0
        else 0
    )

    # --------------------------------------------------
    # Minimum Volatility
    # --------------------------------------------------

    min_hover = (
        f"<b>Minimum Volatility</b><br>"
        f"Return: {min_return:.2%}<br>"
        f"Volatility: {min_risk:.2%}<br>"
        f"Sharpe: {min_sharpe:.2f}<br><br>"
    )

    for asset, weight in min_volatility_weights.items():
        min_hover += f"{asset}: {weight:.2%}<br>"

    fig.add_trace(
        go.Scatter(
            x=[min_risk],
            y=[min_return],
            mode="markers",
            name="Minimum Volatility",
            marker=dict(size=13),
            hovertemplate=min_hover + "<extra></extra>"
        )
    )

    # --------------------------------------------------
    # Maximum Sharpe
    # --------------------------------------------------

    max_hover = (
        f"<b>Maximum Sharpe</b><br>"
        f"Return: {max_return:.2%}<br>"
        f"Volatility: {max_risk:.2%}<br>"
        f"Sharpe: {max_sharpe:.2f}<br><br>"
    )

    for asset, weight in max_sharpe_weights.items():
        max_hover += f"{asset}: {weight:.2%}<br>"

    fig.add_trace(
        go.Scatter(
            x=[max_risk],
            y=[max_return],
            mode="markers",
            name="Maximum Sharpe",
            marker=dict(size=13),
            hovertemplate=max_hover + "<extra></extra>"
        )
    )

    # --------------------------------------------------
    # Individual Assets
    # --------------------------------------------------

    asset_volatility = np.sqrt(
        np.diag(covariance.values)
    )

    asset_hover = []

    for asset, ret, vol in zip(
        expected_returns.index,
        expected_returns.values,
        asset_volatility
    ):
        asset_hover.append(
            f"<b>{asset}</b><br>"
            f"Return: {ret:.2%}<br>"
            f"Volatility: {vol:.2%}"
        )

    fig.add_trace(
        go.Scatter(
            x=asset_volatility,
            y=expected_returns.values,
            mode="markers+text",
            text=expected_returns.index,
            textposition="top center",
            name="Assets",
            marker=dict(size=10),
            hovertext=asset_hover,
            hoverinfo="text"
        )
    )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    fig.update_layout(
        title="Markowitz Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        template="plotly_white",
        hovermode="closest"
    )

    return fig