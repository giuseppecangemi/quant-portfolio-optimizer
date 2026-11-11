"""Offline integration check for the two-month FF backtest cutoff."""
import ast
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import backtest, factor_currency, factor_covariance, factor_eur, factors, optimization, returns, risk

dates = pd.bdate_range("2024-01-01", periods=210)
rng = np.random.default_rng(101)
names = ["MKT-RF", "SMB", "HML", "RMW", "CMA"]
ff = pd.DataFrame(rng.normal(0.0005, 0.004, (len(dates), 5)), index=dates, columns=names)
ff["RF"] = 0.0001
daily = ff[names].to_numpy() @ np.array([[1.2, 0.8], [.2, -.3], [.4, .1], [.1, -.2], [.2, .2]])
prices = pd.DataFrame(100 * np.cumprod(1 + daily + .001, axis=0), index=dates, columns=["A", "B"])
seen = []


def checked(estimator, kind):
    def estimate(window, archived, *args, **kwargs):
        signal = window.index[-1]
        cutoff = signal - pd.DateOffset(months=2)
        pd.testing.assert_frame_equal(archived, ff.loc[:cutoff])
        seen.append((kind, kwargs["model"], signal))
        return estimator(window, archived, *args, **kwargs)

    return estimate


mu = checked(factor_currency.estimate_factor_returns, "mu")
cov = checked(factor_covariance.estimate_factor_covariance, "cov")
cutoff_dates = pd.to_datetime(["2026-09-29", "2026-09-30", "2026-10-01"])
assert factor_currency.factors_for_backtest_signal(
    pd.DataFrame(index=cutoff_dates), "2026-11-30"
).index.equals(cutoff_dates[:2])

path = ROOT / "app_core.py"
wanted = {"run_fixed_horizon_engine", "_cost_adjusted_metrics", "build_benchmark_backtest"}
tree = ast.parse(path.read_text())
scope = {
    "np": np, "pd": pd, "get_prices": lambda *a, **kw: prices[["A"]],
    "FTSE_MIB_TICKER": "fixture", "calculate_returns": returns.calculate_returns,
    "annualized_returns": returns.annualized_returns, "covariance_matrix": risk.covariance_matrix,
    "capm_expected_returns": factors.capm_expected_returns,
    "estimate_factor_returns": mu,
    "factors_for_backtest_signal": factor_currency.factors_for_backtest_signal,
    "calculate_backtest_metrics": backtest.calculate_backtest_metrics,
    **{name: getattr(optimization, name) for name in dir(optimization) if name.startswith("optimize_")},
}
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)

fx = pd.Series(1.1, index=dates)
rates = pd.Series(2.0, index=dates)
for market in ("USA", "Borsa Italiana"):
    seen.clear()
    with patch.object(factor_eur, "load_estr_rates", return_value=rates):
        with patch.object(backtest, "estimate_factor_returns", mu), patch.object(factor_covariance, "estimate_factor_covariance", cov):
            for model in ("ff3", "ff5"):
                curve, weights = backtest.backtest_portfolio(
                    prices, strategy=f"{model}_factor_cov_max_sharpe",
                    lookback_days=160, rebalance_days=20, market=market, eurusd_prices=fx, **{f"{model}_factors": ff},
                )
                assert len(weights) == 3 and len(curve) == 50
            fixed = scope["run_fixed_horizon_engine"](
                ["A", "B"], prices, 160, 48, 0.0, 1.0, 100.0,
                ff3_factors=ff, ff5_factors=ff, market=market, eurusd_prices=fx,
            )
            assert len(fixed["ff3_factor_bt"]) == 49
            assert len(fixed["ff5_factor_bt"]) == 49

        assert len(seen) == 16
        for model in ("ff3", "ff5"):
            assert [date for kind, key, date in seen if kind == "mu" and key == model] == [
                dates[159], dates[179], dates[199], dates[160]
            ]
            assert [date for kind, key, date in seen if kind == "cov" and key == model] == [
                dates[159], dates[179], dates[199], dates[160]
            ]
print("PASS: FF3 and FF5 cutoff for expected returns and Factor Cov across both backtests")

from src.ff3_attribution import compute_ff3_attribution, compute_ff5_attribution

function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "add_factor_attributions")
scope.update(backtest_portfolio=backtest.backtest_portfolio,
             compute_ff3_attribution=compute_ff3_attribution,
             compute_ff5_attribution=compute_ff5_attribution)
engine = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "run_backtest_engine")
exec(compile(ast.Module(body=[function, engine], type_ignores=[]), str(path), "exec"), scope)
with patch.object(factor_eur, "load_estr_rates", return_value=rates):
    standard = scope["run_backtest_engine"](
        ["A", "B"], prices, 160, 20, 0.0, 1.0, 100.0,
        ff3_factors=ff, ff5_factors=ff, market="Borsa Italiana", eurusd_prices=fx,
    )
    for result in (standard, fixed):
        scope["add_factor_attributions"](result, "Borsa Italiana", ff, ff, fx)
        for model in ("ff3", "ff5"):
            for strategy in (model, f"{model}_factor"):
                attribution = result[f"{strategy}_performance_attribution"]
                nav = result[f"{strategy}_bt"]["Portfolio Value"]
                assert np.isclose(attribution["summary"].sum(), nav.iloc[-1] / 100 - 1)
            cov_weights = result[f"{model}_factor_weights_history"]
            history = result[f"{model}_factor_cov_exposure_history"]
            pd.testing.assert_index_equal(history.index, cov_weights.index, check_names=False)
            signal = dates[159] if result is standard else dates[160]
            window = prices.loc[:signal].tail(160 if result is standard else 161)
            eligible = ff.loc[:signal - pd.DateOffset(months=2)]
            _, stats = factor_currency.estimate_factor_returns(
                window, eligible, fx, market="Borsa Italiana", model=model, return_stats=True
            )
            assert np.isclose(history["Market Beta"].iloc[0], cov_weights.iloc[0] @ stats["Beta MKT"])
print("PASS: all four EUR attributions reconcile with their own NAV in both backtests")
