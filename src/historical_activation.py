"""Historical price-only activation and causal risk reconstruction."""
import pandas as pd

from src.optimization import (
    optimize_hrp, optimize_maximum_sharpe, optimize_minimum_volatility,
    optimize_risk_parity,
)
from src.returns import annualized_returns
from src.screening import screen_prices_point_in_time
from src.trading_signals import forecast, paper_orders, tail


class PortfolioConfiguration(dict):
    """JSON-compatible settings captured when the source optimizer runs."""

    def replay_blocker(self):
        if self["return_model"] != "Historical":
            return "Historical replay for this model requires verified data vintages. Current Paper activation is available."
        if not self["universe"] or any(not ticker.endswith(".MI") for ticker in self["universe"]):
            return "Historical replay currently supports Milan-listed EUR assets."
        if not self["period"].removesuffix("y").isdigit() or not self["period"].endswith("y"):
            return "Historical replay requires an imported training window in years."
        if self["selection_mode"] == "Quant Selection":
            period = self["screening_period"] or ""
            if (not self["screening_market_matches"] or not self["top_n"]
                    or set(self["active_factors"]) != {"Momentum", "Risk"}
                    or not period.endswith("y") or not period[:-1].isdigit()):
                return "Historical replay requires an imported Momentum + Risk screen with a window in years. Other screening configurations require historical inputs."
        return None


def reconstruct(history, calendar, decision, years, top_n, optimizer, capital,
                commission, whole, max_weight, risk_free, screening_years=None, active_factors=("Momentum", "Risk")):
    execution = calendar.date_to_session(pd.Timestamp(decision), direction="next")
    decision_session = calendar.previous_session(execution)
    sessions = calendar.sessions_in_range(
        decision_session - pd.DateOffset(years=max(years, screening_years or 0)), decision_session
    )
    adjusted = pd.DataFrame({t: h["Adj Close"] for t, h in history.items()}).reindex(sessions)
    adjusted = adjusted.dropna(axis=1)
    # Normalize away adjustments that only rescale the entire training history.
    adjusted = adjusted.div(adjusted.iloc[-1])
    if top_n is None:
        selected = list(history)
        if set(selected) != set(adjusted.columns):
            raise ValueError("Incomplete training history for the imported manual basket.")
    else:
        screening = adjusted.loc[decision_session-pd.DateOffset(years=screening_years):]
        _, selected = screen_prices_point_in_time(screening, top_n=top_n, active_factors=active_factors)
    if not selected:
        raise ValueError("No candidates have complete training history on the selected calendar.")
    returns = adjusted.loc[decision_session-pd.DateOffset(years=years):, selected].pct_change(fill_method=None).dropna()
    if len(returns) < 2:
        raise ValueError("Insufficient returns in the imported training window.")
    expected, covariance = annualized_returns(returns), returns.cov() * 252
    if optimizer == "Maximum Sharpe":
        weights = optimize_maximum_sharpe(expected, covariance, risk_free, max_weight)
    elif optimizer == "Minimum Volatility":
        weights = optimize_minimum_volatility(expected, covariance, max_weight)
    else:
        method = {"Risk Parity": optimize_risk_parity, "HRP": optimize_hrp}[optimizer]
        weights = method(covariance, max_weight)
    active = weights[weights > 0]
    prices = {}
    for ticker in active.index:
        h = history[ticker]
        if execution not in h.index:
            raise ValueError(f"No execution-session quote for {ticker}. The allocation was not changed.")
        split_multiplier = h.loc[h.index > execution, "Stock Splits"].replace(0, 1).prod()
        prices[ticker] = float(h.loc[execution, "Close"] * split_multiplier)
    orders, cash = paper_orders(active, prices, capital, commission, whole)
    for order in orders:
        order["date"] = str(execution.date())
    volatility = float((weights @ covariance @ weights) ** .5)
    snapshot = dict(
        model=f"Historical · {optimizer}", decision_timestamp=calendar.session_close(decision_session).isoformat(),
        execution_timestamp=calendar.session_close(execution).isoformat(),
        calendar=calendar.name, weights=weights.to_dict(), expected_returns=expected.to_dict(),
        covariance=covariance.to_dict(), volatility=volatility,
        expected_return=float(weights @ expected),
        sharpe=float((weights @ expected - risk_free)/volatility) if volatility > 0 else None,
        training_years=years, top_n=top_n, universe=list(history),
        risk_free_rate=risk_free, max_weight=max_weight, commission=commission, whole_shares=whole,
    )
    return snapshot, orders, cash


def historical_forecasts(returns, calendar, until, existing):
    """Estimate each forecast using a prefix ending at the origin session."""
    for i in range(125, len(returns)-1):
        origin, valid = returns.index[i:i+2]
        if valid > until or calendar.next_session(origin) != valid:
            continue
        for dist in ("normal", "t"):
            key = f"{origin.date()}:{dist}"
            if key in existing:
                continue
            fit = forecast(returns.iloc[:i+1], dist)
            if fit is None:
                yield key, {"origin": str(origin.date()), "status": "No valid fit"}
                continue
            yield key, dict(
                origin=str(origin.date()), valid=str(valid.date()), model=dist,
                horizon=1, reconstructed=True, information_cutoff=calendar.session_close(origin).isoformat(),
                mean=fit["mean"], sigma=fit["sigma"], params=fit["params"],
                tails={str(c): tail(fit["mean"], fit["sigma"], c, fit["params"].get("nu"))
                       for c in (.95, .975, .99)},
            )
