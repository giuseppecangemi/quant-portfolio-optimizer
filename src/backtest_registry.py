"""Strategy configuration and consumer view of the existing backtest results."""
import json

from src.ml_forecasting import MODELS

LEGACY = {
    "Maximum Sharpe": ("max_sharpe_bt", "max_sharpe_weights_history"),
    "CAPM Maximum Sharpe": ("capm_bt", "capm_weights_history"),
    "FF3 Maximum Sharpe": ("ff3_bt", "ff3_weights_history"),
    "FF5 Maximum Sharpe": ("ff5_bt", "ff5_weights_history"),
    "FF3 Factor Cov Maximum Sharpe": ("ff3_factor_bt", "ff3_factor_weights_history"),
    "FF5 Factor Cov Maximum Sharpe": ("ff5_factor_bt", "ff5_factor_weights_history"),
    "Minimum Volatility": ("min_vol_bt", "min_vol_weights_history"),
    "Risk Parity": ("risk_parity_bt", "risk_parity_weights_history"),
    "HRP": ("hrp_bt", "hrp_weights_history"),
    "Equal Weight": ("equal_weight_bt", "equal_weight_weights_history"),
    "Benchmark": ("ftse_mib_bt", None),
}


def ml_backtest_configs(market, registered):
    configs = [
        dict(configuration=variant, model=model, horizon=1, optimizer="Maximum Sharpe",
             historical_cov="Sample", market=market)
        for variant in ("ML Returns", "ML Covariance", "Full ML")
        for model in ((None,) if variant == "ML Covariance" else MODELS)
    ]
    configs.extend(c for c in registered.values() if c.get("market") == market)
    result = {}
    for config in configs:
        variant = config["configuration"]
        if variant == "Historical":
            continue
        model = config["model"]
        horizon = config["horizon"]
        optimizer = config["optimizer"]
        risk = "ML risk" if variant != "ML Returns" else config["historical_cov"]
        label = " · ".join(filter(None, (variant, model, risk, optimizer, f"H{horizon}" if model else None)))
        identity = (variant, model, horizon if model else None, optimizer, risk, market)
        result[label] = {**config, "strategy_id": json.dumps(identity), "training_policy": "Backtest estimation window"}
    return result


def strategy_options(configs):
    """Portfolio labels for the main run and the overlay selector."""
    return list(dict.fromkeys([*(label for label, (_, weights) in LEGACY.items() if weights), *configs]))


def backtest_strategies(result):
    strategies = {}
    for label, (curve_key, history_key) in LEGACY.items():
        curve = result.get(curve_key)
        if curve is not None and not curve.empty:
            strategies[label] = dict(curve=curve, weights=result.get(history_key), strategy_id=curve_key,
                                     ex_ante=result.get("initial_ex_ante_kpis", {}).get(curve_key))
    for label, run in result.get("ml_runs", {}).items():
        if (run.get("status", "Completed") == "Completed" and not run["curve"].empty
                and not run["weights"].empty):
            config = run.get("configuration", {})
            identity = config.get("strategy_id") or json.dumps(config, sort_keys=True, default=str)
            strategies[label] = {**run, "strategy_id": f"registered:{label}:{identity}"}
    return strategies


def allocation_csv(table):
    return table.to_csv(index_label="Rebalance Date").encode("utf-8-sig")


def transaction_cost_row(curve, capital):
    gross = curve.get("Gross Portfolio Value")
    log = curve.attrs.get("transaction_log")
    if gross is None or log is None:
        return None
    return {
        "Total Costs (€)": float(log["Total Cost"].sum()),
        "Orders": int(log["Orders"].sum()),
        "Gross Return (%)": (float(gross.iloc[-1]) / capital - 1) * 100,
        "Net Return (%)": (float(curve["Portfolio Value"].iloc[-1]) / capital - 1) * 100,
    }
