import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.hybrid_portfolio import GLOBAL, REGIONAL, annotate_allocation, effective_universe
from src.portfolio_context import configure_portfolio, portfolio_scope
from src.factor_covariance import estimate_factor_covariance
from src.factor_currency import estimate_factor_returns, factor_covariance_available
from src.optimization import optimize_maximum_sharpe, optimize_minimum_volatility, optimize_risk_parity, weight_bounds
from src.screening import screen_prices_point_in_time
from src.time_series_portfolio import allocation, prices_in_currency


class HybridPortfolioTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(73)
        self.dates = pd.bdate_range("2022-01-03", periods=820)
        self.italian = [f"TEST{i:02}.MI" for i in range(22)]
        shared = rng.normal(0, .009, size=len(self.dates))
        asset = shared[:, None] + rng.normal(.0003, .009, size=(len(shared), len(self.italian)+1))
        self.prices = pd.DataFrame(100*np.exp(np.cumsum(asset, axis=0)), index=self.dates,
                                   columns=self.italian+["NVDA"])
        self.native = self.prices[self.italian]
        fx = pd.Series(1.09*np.exp(np.cumsum(rng.normal(0, .002, size=len(shared)))), index=self.dates)
        self.fx = fx
        self.prices["NVDA"] = prices_in_currency(self.prices.NVDA, 1/fx, prior_day=True)
        self.rates = pd.Series(2., index=self.dates)
        columns = ["MKT-RF", "SMB", "HML", "RMW", "CMA"]
        self.factors = {region: pd.DataFrame(rng.normal(.0002, .007, (len(shared), 5)),
                                            index=self.dates, columns=columns).assign(RF=.0001)
                        for region in ("Borsa Italiana", "USA", "Developed")}

    def config(self, minimum=.05, policy=REGIONAL, effective="2022-01-03"):
        _, ranked = screen_prices_point_in_time(self.native, top_n=20)
        manual = {"NVDA": dict(name="NVIDIA", region="USA", exchange="NASDAQ", quote_currency="USD",
                               minimum=minimum, effective_from=effective,
                               inclusion="Mandatory Inclusion" if minimum else "Optimizer Discretion")}
        return effective_universe([(t, t) for t in ranked], manual, "Borsa Italiana", "EUR", policy)

    def test_join_and_dedup_stable_ranking(self):
        before, selected = screen_prices_point_in_time(self.native, top_n=20)
        config = self.config()
        after, reused = screen_prices_point_in_time(self.native, top_n=20)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(selected, reused)
        self.assertEqual([a["ticker"] for a in config["assets"]], selected + ["NVDA"])
        duplicate = effective_universe([("NVDA", "NVIDIA")], {"NVDA": config["assets"][-1]}, "USA")
        self.assertEqual(len(duplicate["assets"]), 1)
        self.assertEqual(duplicate["assets"][0]["sources"], ["Quantitative", "Manual"])

    def test_regional_global_ff3_ff5_and_constraint(self):
        data = self.prices[[item["ticker"] for item in self.config()["assets"]]].iloc[20:]
        with patch("src.factor_eur.load_estr_rates", return_value=self.rates):
            for policy in (REGIONAL, GLOBAL):
                for model in ("ff3", "ff5"):
                    for floor in (0., .05):
                        with self.subTest(policy=policy, model=model, floor=floor), portfolio_scope():
                            configure_portfolio(self.config(floor, policy))
                            data.attrs["portfolio_currency"] = "EUR"
                            with patch("src.time_series_market.load_factors", side_effect=lambda _, region: self.factors[region]):
                                mu = estimate_factor_returns(data, self.factors["USA"], self.fx,
                                                             market="Borsa Italiana", model=model)
                                if policy == GLOBAL:
                                    implied = estimate_factor_covariance(
                                        data, self.factors["USA"], model=model, eurusd_prices=self.fx)
                                    self.assertEqual(implied.index.tolist(), data.columns.tolist())
                                    self.assertGreaterEqual(np.linalg.eigvalsh(implied).min(), -1e-10)
                                    self.assertNotEqual(implied.loc["NVDA", data.columns[0]], 0.)
                                else:
                                    with self.assertRaisesRegex(ValueError, "common factor set"):
                                        estimate_factor_covariance(data, self.factors["USA"], model=model,
                                                                   eurusd_prices=self.fx)
                            cov = data.pct_change().dropna().cov()*252
                            w = optimize_maximum_sharpe(mu, cov, .02)
                            self.assertGreaterEqual(w.NVDA, floor-1e-8)
                            self.assertAlmostEqual(w.sum(), 1.)
                            self.assertTrue(np.isfinite(mu).all())
                            self.assertNotEqual(cov.loc["NVDA", data.columns[0]], 0.)
                            self.assertEqual(factor_covariance_available(data.columns), policy == GLOBAL)

    def test_bounds_and_post_rounding(self):
        mu = pd.Series([.07, .01, -.05], index=list("ABC"))
        cov = pd.DataFrame(np.diag([.1, .2, .3]), index=mu.index, columns=mu.index)
        funcs = (lambda **kw: optimize_minimum_volatility(mu, cov, .7, **kw),
                 lambda **kw: optimize_risk_parity(cov, .7, **kw),
                 lambda **kw: optimize_maximum_sharpe(mu, cov, 0., .7, **kw),
                 lambda **kw: optimize_maximum_sharpe(mu, cov, .5, .7, **kw))
        for solve in funcs:
            np.testing.assert_allclose(solve(), solve(minimum_weights={"C": 0.}), atol=1e-10)
            w = solve(minimum_weights={"C": .5})
            self.assertGreaterEqual(w.C, .5-1e-8)
            self.assertAlmostEqual(w.sum(), 1.)
        with self.assertRaises(ValueError):
            weight_bounds(mu.index, .6, {"B": .6, "C": .6})
        raw, _ = allocation(pd.Series({"A": .94, "B": .06}), pd.Series({"A": 50., "B": 80.}), 100)
        annotated = annotate_allocation(raw, 100, {"assets": [{"ticker": "B", "minimum": .05}]})
        self.assertTrue(annotated.loc["B", "Below minimum"])

    def test_historical_eligibility(self):
        from src.hybrid_portfolio import eligible_manual, manual_records

        selected = self.italian[:5]
        ready = manual_records(self.config())
        names, status = eligible_manual(self.prices, selected, ready)
        self.assertEqual(names, [*selected, "NVDA"])
        self.assertTrue(status["NVDA"].startswith("Eligible"))
        with self.assertRaisesRegex(ValueError, "not effective"):
            eligible_manual(self.prices, selected, manual_records(self.config(effective="2030-01-01")))
        with self.assertRaisesRegex(ValueError, "missing pre-decision history"):
            eligible_manual(self.native, selected, ready)

    def test_calendar_has_no_nyse_holiday_but_keeps_native_ranking_dates(self):
        from src.hybrid_portfolio import hybrid_sessions

        dates = pd.bdate_range("2025-07-03", "2025-07-08")
        scheduled = hybrid_sessions(dates, self.config())
        self.assertNotIn(pd.Timestamp("2025-07-04"), scheduled)
        self.assertIn(pd.Timestamp("2025-07-04"), dates)
        self.assertIn(pd.Timestamp("2025-07-07"), scheduled)

    def test_fx_strict_prior_and_inverse(self):
        from src.time_series_market import reporting_history

        dates = self.dates[:4]
        native = pd.DataFrame({"Close": [120., 120., 120., 120.], "Adj Close": [110., 115., 120., 120.]}, index=dates)
        fx = pd.DataFrame({"Close": [1.2, 1.1, 1., 4.]}, index=dates)
        def provider(ticker):
            if ticker == "NVDA":
                return native, "USD"
            self.assertEqual(ticker, "EURUSD=X")
            return fx, "USD"
        with patch("src.time_series_market.daily_bars", side_effect=provider):
            converted, original, _, rate = reporting_history("NVDA", "EUR", prior_fx=True)
        pd.testing.assert_frame_equal(original, native)
        np.testing.assert_allclose(converted.Close, [np.nan, 100., 120/1.1, 120.])
        np.testing.assert_allclose(converted["Adj Close"], native["Adj Close"]*rate)
        self.assertEqual(converted.attrs["fx"]["direction"], "divide")

@unittest.skipUnless(Path("data/instruments.csv").exists(), "Run app integration from the repository.")
class HybridApplicationTests(unittest.TestCase):
    italian = ("AMP.MI", "AZM.MI", "BMED.MI")

    def check_backtests(self, app, minimum):
        app.session_state["main_navigation"] = "Backtesting"
        app.run()
        for key, value in (("standard_lookback", "1 year"), ("fixed_lookback", "1 year"),
                           ("fixed_backtest", "2 months"), ("standard_rebalance", "Monthly")):
            app.selectbox(key=key).select(value)
        next(w for w in app.button if w.label == "Run Backtests").click().run()
        self.assertFalse(app.exception)
        fixed = app.session_state["fixed_backtest_results"]
        rolling = app.session_state["standard_backtest_results"]
        self.assertTrue(fixed and rolling, [w.value for w in app.error])
        if "max_sharpe_bt" not in rolling:
            rolling = next(iter(rolling.values()))
        for result in (fixed, rolling):
            self.assertEqual(len(result["max_sharpe_bt"]), 43)
            self.assertTrue(np.isfinite(result["max_sharpe_bt"]["Portfolio Value"]).all())
            if minimum is not None:
                self.assertTrue(result["max_sharpe_weights_history"]["NVDA"].ge(minimum-1e-8).all())
        self.assertEqual(len(fixed["max_sharpe_weights_history"]), 1)
        self.assertGreater(len(rolling["max_sharpe_weights_history"]), 1)
        self.assertTrue(app.get("plotly_chart"))
        app.session_state["main_navigation"] = "Portfolio Optimization"
        app.run()

    def check_original_analysis_pages(self, app, minimum):
        app.session_state["main_navigation"] = "Factor Models"
        app.run()
        self.assertFalse(app.exception)
        app.session_state["factor_model_view"] = "Fama–French FF3"
        app.run()
        app.session_state["factor_model_view"] = "Fama–French FF3"
        next(w for w in app.button if w.label == "Run FF3").click().run()
        factor = app.session_state["ff3_results"]
        self.assertIsNotNone(factor, ([w.value for w in app.error], [w.message for w in app.exception], [w.value for w in app.info], [w.label for w in app.button]))
        self.assertTrue(set(self.italian).issubset(factor["stats"].index))
        self.assertTrue(np.isfinite(factor["expected_returns_eur"]).all())
        if minimum is not None:
            self.assertGreaterEqual(factor["weights"]["NVDA"], minimum-1e-8)
        app.session_state["factor_model_view"] = "Fama–French FF3"
        app.session_state["ff3_covariance"] = "Factor Cov"
        app.run()
        self.assertEqual(next(w for w in app.button if w.label == "Run FF3").disabled, minimum is not None)
        app.session_state["main_navigation"] = "Time Series Portfolio"
        app.run()
        next(w for w in app.checkbox if w.label == "Includi asset da Quant Selection").check()
        app.session_state["main_navigation"] = "Time Series Portfolio"
        next(w for w in app.button if w.label == "Calcola allocazione corrente").click().run()
        self.assertFalse(app.exception)
        self.assertTrue("ts_basket" in app.session_state, [w.value for w in app.error])
        basket = app.session_state["ts_basket"]
        self.assertTrue(set(self.italian).issubset(basket["table"].index))
        self.assertTrue(app.get("plotly_chart"))
        if minimum is not None:
            self.assertGreaterEqual(basket["weights"].loc["NVDA"], minimum-1e-8)
        app.session_state["main_navigation"] = "Time Series Portfolio"
        next(w for w in app.selectbox if w.label == "Timeframe").select("1D").run()
        self.assertTrue(any("Synthetic OHLC:" in w.value for w in app.caption))
        self.assertTrue(any(trace["type"] == "candlestick"
                            for chart in app.get("plotly_chart") for trace in json.loads(chart.proto.spec)["data"]))
        self.assertFalse(app.exception)
        app.session_state["main_navigation"] = "Time Series Portfolio"
        app.session_state["ts_portfolio_analysis"] = "K — AI Portfolio Analyst"
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("Chiedi un’analisi a K", [w.label for w in app.button])
        with (patch("src.time_series_ui.st.secrets", {"OPENAI_API_KEY": "fixture"}),
              patch("src.time_series_ui._request_analysis", return_value="Analisi sintetica verificata") as request):
            app.session_state["main_navigation"] = "Time Series Portfolio"
            app.session_state["ts_portfolio_analysis"] = "K — AI Portfolio Analyst"
            next(w for w in app.button if w.label == "Chiedi un’analisi a K").click().run()
            self.assertEqual(request.call_args.args[0]["currency"], "EUR")
            self.assertTrue(request.call_args.args[0]["positions"])
            self.assertEqual(app.session_state["ts_k_report"][1], "Analisi sintetica verificata")

        app.session_state["main_navigation"] = "Monte Carlo Simulation"
        app.session_state["monte_carlo_views"] = "Forward-Looking Simulation"
        app.run()
        for label, value in (("Expected Return Model", "Historical Mean"), ("Simulation Horizon (Years)", .25),
                             ("Number of Simulations", 1000), ("Estimation Window (Sessions)", 252)):
            next(w for w in app.selectbox if w.label == label).select(value)
        app.session_state["main_navigation"] = "Monte Carlo Simulation"
        app.session_state["monte_carlo_views"] = "Forward-Looking Simulation"
        next(w for w in app.button if w.label == "Run Forward-Looking Simulation").click().run()
        self.assertFalse(app.exception)
        self.assertTrue("forward_mc_result" in app.session_state, [w.value for w in app.error])
        config, _, weights, *_ = app.session_state["forward_mc_result"]
        self.assertEqual(config["currency"], "EUR")
        self.assertTrue(np.isfinite(weights.to_numpy()).all())
        if minimum is not None:
            self.assertTrue(weights.loc["NVDA"].ge(minimum-1e-8).all())

        next(w for w in app.button if w.label == "Save Experiment").click().run()
        self.assertTrue("last_saved_experiment" in app.session_state, [w.value for w in app.error])
        archive = Path(app.session_state["last_saved_experiment"]["path"])
        saved = json.loads(archive.read_text())
        self.assertEqual(saved["portfolio_parameters"]["number_of_selected_assets"], 3 if minimum is None else 4)
        self.assertEqual(saved["experiment"]["schema_version"], 1)
        if minimum is not None:
            self.assertEqual(saved["asset_selection"]["hybrid_universe"]["assets"][-1]["minimum"], minimum)

        app.session_state["main_navigation"] = "Saved Experiments"
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.dataframe)

        app.session_state["main_navigation"] = "ML Return Forecasting"
        app.run()
        app.session_state["main_navigation"] = "ML Return Forecasting"
        next(w for w in app.button if w.label == "Estimate ML models").click().run()
        self.assertFalse(app.exception)
        self.assertTrue("ml_research" in app.session_state, [w.value for w in app.error])
        forecasts = app.session_state["ml_research"]["forecasts"]
        self.assertEqual(len(forecasts), 3 if minimum is None else 4)
        self.assertTrue(np.isfinite(forecasts["Horizon Return"]).all())

        app.session_state["main_navigation"] = "Volatility Analytics"
        app.run()
        app.session_state["main_navigation"] = "Volatility Analytics"
        next(w for w in app.button if w.label == "Run Volatility Analysis").click().run()
        self.assertFalse(app.exception)
        self.assertTrue("volatility_analytics_portfolio" in app.session_state, [w.value for w in app.error])
        self.assertTrue(app.get("plotly_chart"))

    def test_original_optimization_controls_and_results(self):
        from streamlit.testing.v1 import AppTest

        dates = pd.bdate_range(end=pd.Timestamp.now().normalize()-pd.Timedelta(days=2), periods=820)
        rng = np.random.default_rng(30)
        tickers = [*self.italian, "NVDA"]
        panel = pd.DataFrame(75*np.exp(np.cumsum(rng.normal(
            [.0004, .0003, .0005, -.001], .012, (len(dates),len(tickers))), axis=0)),
            index=dates, columns=tickers)
        usd = 1.1 + pd.Series(np.sin(np.arange(len(dates))/17)*.04,index=dates)
        converted = panel.assign(NVDA=panel.NVDA.div(usd.shift())).dropna()
        execution = converted*1.2
        benchmark = panel.iloc[:, :3].mean(axis=1)
        factors = pd.DataFrame(rng.normal(0, .006, (len(dates), 5)), index=dates,
                               columns=["MKT-RF", "SMB", "HML", "RMW", "CMA"]).assign(RF=.0001)
        rates = pd.Series(2., index=dates)

        def feedback(symbol, *args, **kwargs):
            result = usd if symbol == "EURUSD=X" else panel.get(symbol, benchmark)
            return result, []

        def quote(symbol, *args, **kwargs):
            series = usd if symbol == "EURUSD=X" else execution.get(symbol, benchmark)
            adjusted = usd if symbol == "EURUSD=X" else converted.get(symbol, benchmark)
            bars = pd.DataFrame({"Open": series*.99, "High": series*1.01, "Low": series*.98,
                                 "Close": series, "Adj Close": adjusted})
            return bars, bars, "USD" if symbol == "NVDA" else "EUR", None

        def history(symbol):
            bars = quote(symbol)[0]
            if symbol == "NVDA":
                bars.attrs["quote"] = {"name": "NVIDIA", "symbol": "NVDA", "exchange": "NASDAQ",
                                        "type": "EQUITY", "original_currency": "USD"}
            return bars, "USD" if symbol in {"NVDA", "EURUSD=X"} else "EUR"

        def build(root, minimum=None, *, verify_analysis=True):
            workspace = tempfile.TemporaryDirectory(prefix="portfolio-restore-test-")
            self.addCleanup(workspace.cleanup)
            test_root = Path(workspace.name)
            for name in ("app.py", "app_core.py"):
                (test_root / name).write_text((root / name).read_text())
            (test_root / "data").symlink_to(Path.cwd() / "data", target_is_directory=True)
            app = AppTest.from_file(str(test_root / "app.py"), default_timeout=60)
            app.session_state["kangemi_authenticated"] = True
            app.session_state["main_navigation"] = "Portfolio Optimization"
            app.session_state["quant_selected_tickers"] = list(self.italian)
            ranking = pd.DataFrame({"Ticker": self.italian, "Instrument": self.italian,
                                    "Selected": True, "Rank": [1, 2, 3], "Status": "Eligible", "Reason": ""})
            for name in ("Momentum", "Risk", "Value", "Quality", "Quant"):
                ranking[f"{name} Score"] = 80.
            app.session_state["screening_results"] = {"market": "Borsa Italiana",
                "table": ranking, "period": "5y", "top_n": 3}
            with (patch("yfinance.set_tz_cache_location", create=True),
                  patch("src.data._fetch_one", side_effect=feedback),
                  patch("src.hybrid_portfolio.load_hybrid_prices", side_effect=lambda names, _:
                        (execution.reindex(columns=names), pd.DataFrame({t: converted.get(t, benchmark) for t in names}), panel, {})),
                  patch("src.screening.build_price_matrix", return_value=panel[list(self.italian)]),
                  patch("src.fama_french.load_europe_ff3", return_value=factors),
                  patch("src.fama_french_5.load_europe_ff5", return_value=factors),
                  patch("src.backtest_inputs.load_europe_ff3", return_value=factors),
                  patch("src.backtest_inputs.load_europe_ff5", return_value=factors),
                  patch("src.time_series_market.load_factors", return_value=factors),
                  patch("src.time_series_market.daily_bars", side_effect=history),
                  patch("src.hybrid_ui.daily_bars", side_effect=history),
                  patch("src.time_series_market.reporting_history", side_effect=quote),
                  patch("src.factor_eur.load_estr_rates", return_value=rates)):
                app.run()
                self.assertFalse(app.exception)
                next(w for w in app.radio if w.label == "Asset selection").set_value("Quant Selection").run()
                if minimum is not None:
                    app.session_state["main_navigation"] = "Asset Selection"
                    app.run()
                    next(w for w in app.text_input if w.label == "Ticker").set_value("NVDA")
                    next(w for w in app.selectbox if w.label == "Factor region").select("USA")
                    next(w for w in app.date_input if w.label == "Historical eligibility from").set_value(dates[0].date())
                    next(w for w in app.radio if w.label == "Inclusion").set_value(
                        "Mandatory Inclusion" if minimum else "Optimizer Discretion")
                    app.session_state["main_navigation"] = "Asset Selection"
                    next(w for w in app.button if w.label == "Add or update ticker").click().run()
                    self.assertFalse(app.exception)
                    self.assertTrue("hybrid_manual" in app.session_state, [w.value for w in app.error])
                    self.assertEqual(app.session_state["hybrid_manual"]["NVDA"]["minimum"], minimum)
                    self.assertEqual(app.session_state["hybrid_manual"]["NVDA"]["effective_from"],
                                     dates[0].date().isoformat())
                    app.session_state["main_navigation"] = "Portfolio Optimization"
                    app.run()
                next(w for w in app.button if w.label == "Run Portfolio Optimization").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error, [w.value for w in app.error])
                self.check_backtests(app, minimum)
                if verify_analysis:
                    self.check_original_analysis_pages(app, minimum)
                    app.session_state["main_navigation"] = "Portfolio Optimization"
                    app.run()
            result = app.session_state["optimization_results"]
            self.assertIsNotNone(result)
            self.assertTrue(app.get("plotly_chart"))
            self.assertTrue(app.dataframe)
            self.assertFalse(any("hybrid optimization" in w.label.lower() for w in app.button))
            return app, result

        root = Path(__file__).resolve().parents[1]
        results = {}
        for name, minimum in (("A", None), ("B", 0.), ("C", .05)):
            with self.subTest(scenario=name):
                app, result = build(root, minimum)
                prices = result["prices"]
                expected = panel.loc[:, prices.columns] if minimum is None else converted.loc[:, prices.columns]
                pd.testing.assert_frame_equal(prices, expected)
                w = result["max_weights"]
                self.assertAlmostEqual(w.sum(), 1.)
                self.assertTrue(set(w.index) >= set(self.italian))
                if minimum is not None:
                    self.assertGreaterEqual(w.NVDA, minimum-1e-8)
                    self.assertAlmostEqual(result["max_table"].loc["NVDA", "Price"], execution.NVDA.iloc[-1])
                results[name] = (app, result)
                print(f"{name}: Max Sharpe weights {w.to_dict()}, return {result['max_return']:.8f}, "
                      f"volatility {result['max_volatility']:.8f}")
        baseline = os.environ.get("PORTFOLIO_BASELINE")
        if baseline:
            baseline_root = Path(baseline)
            spec = importlib.util.spec_from_file_location("baseline_optimization", baseline_root / "src/optimization.py")
            baseline_optimizer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(baseline_optimizer)
            with ExitStack() as stack:
                for name in ("optimize_maximum_sharpe", "optimize_minimum_volatility", "optimize_risk_parity",
                             "optimize_hrp", "efficient_frontier"):
                    stack.enter_context(patch(f"src.optimization.{name}", getattr(baseline_optimizer, name)))
                baseline_app, reference = build(baseline_root, verify_analysis=False)
            for key in ("max_weights", "min_weights", "risk_parity_weights", "hrp_weights", "covariance", "frontier"):
                assert_equal = pd.testing.assert_frame_equal if isinstance(reference[key], pd.DataFrame) else pd.testing.assert_series_equal
                assert_equal(results["A"][1][key], reference[key])
            for key in ("fixed_backtest_results", "standard_backtest_results"):
                actual = results["A"][0].session_state[key]
                expected = baseline_app.session_state[key]
                if "max_sharpe_bt" not in actual:
                    actual = next(iter(actual.values()))
                    expected = next(iter(expected.values()))
                pd.testing.assert_frame_equal(actual["max_sharpe_bt"], expected["max_sharpe_bt"])
        self.assertEqual([w.label for w in results["A"][0].button],
                         [w.label for w in results["C"][0].button])
