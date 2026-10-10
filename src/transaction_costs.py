"""Transparent ex-post transaction cost overlay for historical backtests.

The signal/portfolio optimizer is unchanged. Execution is approximated at the
same adjusted close used by the existing backtest, not at executable quotes.
The model uses actual target-weight changes, drifted by observed asset prices.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostAssumptions:
    commission_rate: float = 0.0005
    min_commission: float = 3.0
    half_spread_bps: float = 5.0
    slippage_bps: float = 5.0
    # 0 by default: Italian FTT depends on issuer, capitalization, venue, date.
    # It must NOT be applied indiscriminately to every Italian or foreign ticker.
    buy_tax_rate: float = 0.0


PRESETS = {
    "Optimistic": CostAssumptions(min_commission=1.25, half_spread_bps=2, slippage_bps=2),
    "Base": CostAssumptions(),
    "Conservative": CostAssumptions(half_spread_bps=10, slippage_bps=15),
}


def apply_transaction_costs(gross_curve: pd.DataFrame, weights: pd.DataFrame,
                            prices: pd.DataFrame, capital: float,
                            assumptions: CostAssumptions, risk_free_rate: float = 0.02):
    """Return net metrics, execution log, net curve. No optimizer modification.

    At a rebalance, drift prior target weights using prices since the previous
    rebalance, compare with the new target, and charge on traded notional.
    Net capital evolves with gross daily returns, minus explicit order costs.
    Fractional shares and same-close fills are assumptions of the base engine.
    """
    from src.backtest import calculate_backtest_metrics
    curve = gross_curve.sort_index().copy()
    if curve.empty or weights.empty:
        raise ValueError("Missing portfolio curve or rebalance weights")
    curve = curve.loc[~curve.index.duplicated(keep="first")]
    prices = prices.sort_index()
    weights = weights.sort_index()
    returns = curve["Portfolio Value"].pct_change().fillna(0.0)
    net = float(capital)
    prev_w = pd.Series(dtype=float)
    prev_date = None
    records = []
    net_values = []
    net_returns = []
    all_tickers = prices.columns.union(weights.columns)
    for date, daily_r in returns.items():
        before = net
        net *= (1 + float(daily_r))
        if date in weights.index:
            target = weights.loc[date].reindex(all_tickers).fillna(0.0).astype(float).clip(lower=0)
            if isinstance(target, pd.DataFrame):
                target = target.iloc[-1]
            target = target / target.sum() if target.sum() > 0 else target
            if prev_date is None:
                old = pd.Series(0.0, index=all_tickers)
            else:
                p0 = prices.loc[prev_date].reindex(all_tickers)
                p1 = prices.loc[date].reindex(all_tickers)
                ratios = (p1 / p0).replace([np.inf, -np.inf], np.nan).fillna(1.0)
                drift = prev_w.reindex(all_tickers).fillna(0.0) * ratios
                old = drift / drift.sum() if drift.sum() > 0 else drift
            delta = target - old
            traded = (delta.abs() * net).loc[lambda x: x > 1e-9]
            buy_notional = float(delta.clip(lower=0).sum() * net)
            variable = float(traded.sum()) * (assumptions.half_spread_bps + assumptions.slippage_bps) / 10000
            commissions = sum(max(assumptions.min_commission, assumptions.commission_rate * float(v)) for v in traded)
            tax = buy_notional * assumptions.buy_tax_rate
            total = float(variable + commissions + tax)
            if total >= net:
                raise ValueError("Trading costs exhaust portfolio capital")
            net -= total
            records.append({"Rebalance Date": date, "Orders": len(traded),
                            "Traded Notional": float(traded.sum()), "Turnover": float(delta.abs().sum()),
                            "Commission": float(commissions), "Spread and Slippage": float(variable),
                            "Buy Tax": float(tax), "Total Cost": total})
            prev_w = target
            prev_date = date
        net_values.append(net)
        net_returns.append(net / before - 1 if before > 0 else 0.0)
    net_curve = pd.DataFrame({"Portfolio Value": net_values, "Daily Return": net_returns}, index=curve.index)
    # Include first-order costs in Sharpe and max drawdown by inserting starting
    # capital as a reference point; no artificial calendar date is introduced.
    metric_curve = net_curve.copy()
    metric_curve.loc[metric_curve.index[0], "Daily Return"] = net_values[0] / capital - 1
    metrics = calculate_backtest_metrics(metric_curve, risk_free_rate)
    metrics["Total Return"] = net_values[-1] / capital - 1
    dd = np.asarray([capital] + net_values, dtype=float)
    metrics["Max Drawdown"] = float((dd / np.maximum.accumulate(dd) - 1).min())
    return metrics, pd.DataFrame(records), net_curve


def estimate_order_costs(quantity_changes, execution_prices, assumptions):
    """Estimate costs from executed share changes at the rebalance close.

    Per-order minimum commissions, one-way half spread, and slippage are
    applied to each nonzero order. The optional buy tax is not applied unless
    explicitly configured. No taxes on realized capital gains are included.
    """
    import numpy as np
    import pandas as pd
    notional = (quantity_changes.abs() * execution_prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    notional = notional[notional > 1e-8]
    commission = sum(max(assumptions.min_commission, assumptions.commission_rate * float(v)) for v in notional)
    execution = float(notional.sum()) * (assumptions.half_spread_bps + assumptions.slippage_bps) / 10000.0
    buys = (quantity_changes.clip(lower=0) * execution_prices).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tax = float(buys.sum()) * assumptions.buy_tax_rate
    return commission + execution + tax, {
        "Orders": int(len(notional)), "Traded Notional": float(notional.sum()),
        "Commission": float(commission), "Spread and Slippage": float(execution),
        "Buy Tax": float(tax),
    }
