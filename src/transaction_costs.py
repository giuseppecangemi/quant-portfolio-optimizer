"""Modello trasparente di stima ex-post dei costi di transazione
applicato ai backtest storici.

I segnali e l'ottimizzatore del portafoglio rimangono invariati.
L'esecuzione delle operazioni viene approssimata utilizzando gli stessi
prezzi di chiusura rettificati impiegati nel backtest esistente,
anziché quotazioni effettivamente negoziabili.

Il modello utilizza le variazioni effettive dei pesi target,
tenendo conto degli scostamenti dei pesi determinati dai movimenti
osservati dei prezzi degli asset.
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
    # Valore predefinito pari a 0: l'applicabilità della tassa italiana
    # sulle transazioni finanziarie (FTT) dipende dall'emittente, dalla
    # capitalizzazione, dalla sede di negoziazione e dalla data.
    # NON deve essere applicata indiscriminatamente a tutti i titoli
    # italiani o esteri.
    buy_tax_rate: float = 0.0


PRESETS = {
    "Optimistic": CostAssumptions(min_commission=1.25, half_spread_bps=2, slippage_bps=2),
    "Base": CostAssumptions(),
    "Conservative": CostAssumptions(half_spread_bps=10, slippage_bps=15),
}


def apply_transaction_costs(gross_curve: pd.DataFrame, weights: pd.DataFrame,
                            prices: pd.DataFrame, capital: float,
                            assumptions: CostAssumptions, risk_free_rate: float = 0.02):
    """Restituisce le metriche nette, il registro delle operazioni eseguite
    e la curva del portafoglio al netto dei costi di transazione,
    senza modificare l'ottimizzatore.

    A ogni ribilanciamento, i pesi target precedenti vengono aggiornati
    in base alle variazioni dei prezzi osservate dall'ultimo ribilanciamento
    e confrontati con i nuovi pesi target.

    I costi di transazione vengono calcolati sul controvalore effettivamente
    negoziato.

    Il capitale netto evolve in funzione dei rendimenti giornalieri lordi,
    al netto dei costi espliciti delle operazioni.

    L'utilizzo di quote frazionarie e l'esecuzione delle operazioni
    ai prezzi di chiusura della stessa giornata sono ipotesi
    adottate dal motore di backtesting originale.
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
    records = []
    net_values = []
    net_returns = []
    all_tickers = prices.columns.union(weights.columns)
    previous = pd.Series(0.0, index=all_tickers)
    for date, daily_r in returns.items():
        before = net
        net *= 1 + float(daily_r)
        if date in weights.index:
            target = weights.loc[[date]].iloc[-1].reindex(all_tickers).fillna(0.0).astype(float)
            if (target < 0).any() or target.sum() <= 0:
                raise ValueError("Invalid long-only target weights")
            target /= target.sum()
            execution = prices.loc[date].reindex(all_tickers)
            active = target.gt(0) | previous.ne(0)
            if not np.isfinite(execution[active]).all() or execution[active].le(0).any():
                raise ValueError(f"Missing or invalid execution price on {date}")
            pretrade = net
            net, total, detail = fund_orders(pretrade, target, previous, execution, assumptions)
            previous = (net * target).div(execution.where(target > 0)).fillna(0.0)
            records.append({"Rebalance Date": date, **detail,
                            "Turnover": detail["Traded Notional"] / pretrade,
                            "Total Cost": total})
        net_values.append(net)
        net_returns.append(net / before - 1)
    net_curve = pd.DataFrame({"Portfolio Value": net_values, "Daily Return": net_returns}, index=curve.index)
    net_curve.attrs["initial_capital"] = float(capital)
    metrics = calculate_backtest_metrics(net_curve, risk_free_rate)
    return metrics, pd.DataFrame(records), net_curve


def estimate_order_costs(quantity_changes, execution_prices, assumptions):
    """Stima i costi di transazione in base alle variazioni effettive delle
    quantità di titoli negoziate ai prezzi di chiusura del ribilanciamento.

    Per ogni ordine con quantità diversa da zero vengono applicate
    le commissioni minime previste, il semi-spread denaro-lettera
    e i costi di slippage.

    L'eventuale imposta sugli acquisti viene applicata esclusivamente
    se configurata esplicitamente.

    Non sono incluse le imposte sulle plusvalenze realizzate.
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


def fund_orders(capital, weights, previous_quantities, prices, assumptions):
    """Solve the self-financing budget and verify it against the final orders."""
    from scipy.optimize import brentq

    def orders(budget):
        quantities = (budget * weights).div(prices.where(weights > 0)).fillna(0.0)
        return estimate_order_costs(quantities - previous_quantities, prices, assumptions)

    cost, detail = orders(capital)
    if cost == 0:
        return capital, cost, detail
    if orders(0.0)[0] >= capital:
        raise ValueError("Trading costs exhaust portfolio capital")
    budget = brentq(lambda amount: amount + orders(amount)[0] - capital, 0.0, capital)
    cost, detail = orders(budget)
    if budget <= 0 or not np.isclose(budget + cost, capital, atol=1e-8, rtol=1e-12):
        raise ValueError("Trading costs have no consistent self-financing budget")
    return budget, cost, detail
