"""Exercise the rolling runner against the actual Walk-Forward engine offline."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import factor_eur, robustness
from src.backtest import backtest_portfolio
from src.transaction_costs import PRESETS


def test_rolling_robustness():
    index = pd.bdate_range("2023-12-01", "2026-01-07")
    rng = np.random.default_rng(23)
    ff = pd.DataFrame(rng.normal(.0004, .005, (len(index), 5)), index=index,
                      columns=["MKT-RF", "SMB", "HML", "RMW", "CMA"])
    ff["RF"] = .0001
    prices = pd.DataFrame(100 * np.cumprod(1 + ff.iloc[:, :2].to_numpy()
                           + rng.normal(.0005, .003, (len(index), 2)), axis=0),
                          index=index, columns=["A", "B"])
    benchmark = prices[["A"]].rename(columns={"A": "INDEX"})
    fx = pd.DataFrame({"EURUSD=X": 1.1}, index=index)
    rates = pd.Series(2.0, index=index)
    sources = pd.concat([prices, benchmark, fx], axis=1)
    costs = PRESETS["Base"]

    def download(tickers, start, end):
        return sources.loc[(index >= start) & (index < end), tickers]

    with (patch.object(robustness, "_download_close", side_effect=download) as fetch,
          patch.object(robustness, "load_us_ff3", return_value=ff) as us3,
          patch.object(robustness, "load_us_ff5", return_value=ff) as us5,
          patch.object(robustness, "load_europe_ff3", return_value=ff) as eu3,
          patch.object(robustness, "load_europe_ff5", return_value=ff) as eu5,
          patch.object(factor_eur, "load_estr_rates", return_value=rates)):
        for market in ("USA", "Borsa Italiana"):
            progress = []
            result, skipped, info = robustness.run_rolling_robustness(
                ["A", "B"], "INDEX", start_year=2025, end_year=2025, estimation_years=1,
                top_n=None, max_weight=.65, market=market, transaction_costs=costs, progress=progress.append,
            )
            assert skipped.empty, skipped
            assert set(result["Model"]) == {*robustness.STRATEGIES, "Benchmark"}
            assert info["market"] == market and progress[-1] == 1.0
            window = prices.loc["2024":"2025"]
            direct, weights = backtest_portfolio(
                window, strategy="ff5_factor_cov_max_sharpe", lookback_days=len(prices.loc["2024"]),
                risk_free_rate=.02, max_weight=.65, market=market, eurusd_prices=fx,
                rebalance_days=63,
                ff5_factors=ff, transaction_costs=costs, capture_initial_ex_ante=True,
            )
            row = result.set_index("Model").loc["FF5 Factor Cov"]
            assert row["Execution End"] == "2025-12-31"
            assert row["Estimation End"] == "2024-12-31"
            assert row["Transaction Costs"] > 0
            assert np.isfinite(row["Initial Sharpe"])
            assert row["Net Total Return"] < row["Total Return"]
            assert np.isclose(row["Net Total Return"], direct["Portfolio Value"].iloc[-1] / 10000 - 1)
            assert np.isclose(row["Total Return"], direct["Gross Portfolio Value"].iloc[-1] / 10000 - 1)
            assert row["Rebalances"] == len(weights) - 1
        assert us3.call_count == us5.call_count == eu3.call_count == eu5.call_count == 1
        assert sum(call.args[0] == ["EURUSD=X"] for call in fetch.call_args_list) == 1


if __name__ == "__main__":
    test_rolling_robustness()
    print("PASS: ten models, benchmark, EUR/USD routing, initial KPIs and engine costs")


def test_oos_boundary_uses_last_scheduled_session():
    import exchange_calendars

    dates = exchange_calendars.get_calendar("XMIL", start="2023-12-01", end="2026-01-07").sessions
    rng = np.random.default_rng(52)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(.0003, .008, (len(dates), 2)), axis=0),
                          index=dates, columns=["A", "B"])
    last = pd.Timestamp("2025-12-30")
    capture = []
    def simulate(*args, **kwargs):
        result = backtest_portfolio(*args, **kwargs)
        capture.append(result)
        return result
    def evaluate(panel):
        def download(tickers, start, end):
            return panel[["A"]].rename(columns={"A": "INDEX"}) if tickers == ["INDEX"] else panel
        with (patch.object(robustness, "_download_close", side_effect=download),
              patch.object(robustness, "backtest_portfolio", side_effect=simulate)):
            return robustness.run_rolling_robustness(
                ["A", "B"], "INDEX", 2025, 2025, 1, top_n=None, market="Borsa Italiana",
                models=["Equal Weight"], max_weight=.65, transaction_costs=PRESETS["Base"],
            )
    full, full_skips, _ = evaluate(prices)
    terminal, terminal_skips, _ = evaluate(prices.loc[:last])
    pd.testing.assert_frame_equal(full, terminal)
    assert full_skips.empty and terminal_skips.empty
    (a, wa), (b, wb) = capture
    assert a.index[-1] == b.index[-1] == last
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    pd.testing.assert_frame_equal(wa, wb)
    pd.testing.assert_frame_equal(a.attrs["transaction_log"], b.attrs["transaction_log"])
    incomplete, reasons, _ = evaluate(prices.drop(last))
    assert incomplete.empty
    assert reasons.iloc[0]["Reason"] == "Missing asset prices on final scheduled OOS session 2025-12-30"
