"""Genera avvisi informativi sulla volatilità, senza eseguire ordini
di trading né modificare i risultati dei backtest."""
import math


def risk_alert(forecast_vol, target_vol=15.0, current_exposure=100.0,
               caution=20.0, warning=25.0, critical=30.0,
               review_gap=5.0, rebalance_gap=10.0):
    """Restituisce un avviso trasparente sul livello di rischio e una proposta
    di adeguamento dell'esposizione basata su un obiettivo di volatilità
    (volatility targeting), senza ricorrere alla leva finanziaria.

    Tutte le volatilità e le esposizioni sono espresse in percentuale,
    mentre gli scostamenti sono espressi in punti percentuali.
    """
    vals = [forecast_vol, target_vol, current_exposure, caution, warning,
            critical, review_gap, rebalance_gap]
    if not all(math.isfinite(float(x)) for x in vals):
        raise ValueError("Risk alert parameters must be finite")
    if not (0 < caution < warning < critical):
        raise ValueError("Risk thresholds must be strictly increasing and positive")
    if not (target_vol > 0 and forecast_vol > 0 and 0 <= current_exposure <= 100):
        raise ValueError("Invalid target, forecast or current exposure")
    if not (0 <= review_gap < rebalance_gap):
        raise ValueError("Rebalancing thresholds must be increasing")

    if forecast_vol >= critical:
        status, color = "CRITICAL", "#ef4444"
    elif forecast_vol >= warning:
        status, color = "HIGH", "#f97316"
    elif forecast_vol >= caution:
        status, color = "ELEVATED", "#eab308"
    else:
        status, color = "NORMAL", "#22c55e"
    suggested = min(100.0, 100.0 * target_vol / forecast_vol)
    gap = abs(current_exposure - suggested)
    if gap >= rebalance_gap:
        action = "REBALANCE REVIEW"
    elif gap >= review_gap:
        action = "MONITOR / REVIEW"
    else:
        action = "NO CHANGE INDICATED"
    return dict(status=status, color=color, forecast=float(forecast_vol),
                suggested_exposure=suggested, suggested_cash=100-suggested,
                current_exposure=float(current_exposure), gap_pp=gap,
                action=action, target=float(target_vol))
