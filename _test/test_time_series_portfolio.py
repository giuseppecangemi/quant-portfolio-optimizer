import ast
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.time_series_analytics import breadth, portfolio_ohlc, price_zones, relative_strength, risk_dynamics, technical
from src.time_series_portfolio import CASH, allocation, current_model, linked_attribution, prices_in_currency, reconstruct, synthetic_daily_ohlc


class TimeSeriesAccounting(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2025-01-06", periods=12)
        self.prices = pd.DataFrame({"A": np.linspace(79, 119, 12), "B": np.linspace(60, 32, 12)}, index=self.dates)
        self.positions = pd.Series({"A": 3, "B": 7, "UNHELD": 0})
        self.end = self.dates[-1]

    def test_fixed_quantities_and_attribution(self):
        cash = 19.
        path = reconstruct(self.prices, self.positions, cash, self.end)
        marks = self.prices.mul(self.positions.loc[["A", "B"]])
        pd.testing.assert_series_equal(path.value, marks.sum(axis=1)+cash)
        pd.testing.assert_frame_equal(path.weights[["A", "B"]], marks.div(path.value, axis=0))
        nav = path.value.pct_change(fill_method=None)
        np.testing.assert_allclose(path.contributions.iloc[1:].sum(axis=1), nav.iloc[1:])
        linked = linked_attribution(path.contributions, nav, "Weekly")
        compounded = (nav+1).groupby(self.dates.to_period("W")).prod()-1
        np.testing.assert_allclose(linked.sum(axis=1), compounded)
        single = reconstruct(self.prices, pd.Series({"A": 3}), cash, self.end)
        np.testing.assert_allclose(single.value, self.prices.A*3+cash)
        self.assertEqual((path.value / path.value.iloc[0] * 100).iloc[0], 100.)

    def test_rebalance_uses_weights_after_close_and_anchors_endpoint(self):
        positions = self.positions[["A", "B"]]
        target = pd.Series({"A": .4, "B": .4, CASH: .2})
        daily = reconstruct(self.prices, positions, 19, self.end, targets=target, frequency="Daily")
        np.testing.assert_allclose(daily.value.pct_change().iloc[1:], self.prices.pct_change().iloc[1:] @ target[["A", "B"]])
        weekly = reconstruct(self.prices, positions, 19, self.end, targets=target, frequency="Weekly")
        self.assertEqual(list(weekly.rebalanced[weekly.rebalanced].index), [self.dates[4], self.dates[9]])
        np.testing.assert_allclose(weekly.weights.loc[self.dates[4]], target)
        self.assertNotAlmostEqual(weekly.weights.A.iloc[3], target.A)
        self.assertAlmostEqual(weekly.value.iloc[-1], self.prices.iloc[-1] @ positions + 19)
        np.testing.assert_allclose(weekly.contributions.iloc[1:].sum(axis=1), weekly.value.pct_change().iloc[1:])

    def test_synthetic_daily_bounds_missing_and_adjusted_basis(self):
        close = self.prices
        raw = {"Open": close * .98, "High": close * 1.07, "Low": close * .94, "Close": close}
        path = reconstruct(close, self.positions, 19., self.end)
        bars = synthetic_daily_ohlc(path, close, raw, close.index)
        for field in ("open", "high", "low", "close"):
            np.testing.assert_allclose(bars[field], raw[field.title()] @ self.positions[["A", "B"]] + 19.)
        broken = dict(raw, High=raw["High"].copy())
        broken["High"].loc[self.dates[2], "A"] = np.nan
        bars = synthetic_daily_ohlc(path, close, broken, close.index)
        self.assertNotIn(self.dates[2], bars.index)
        self.assertEqual(len(bars), len(close)-1)
        adjusted = close.copy()
        adjusted.loc[self.dates[:5], "A"] *= .9
        total = reconstruct(adjusted, self.positions, 19., self.end)
        bars = synthetic_daily_ohlc(total, adjusted, raw, close.index)
        np.testing.assert_allclose(bars.open, adjusted.mul(.98) @ self.positions[["A", "B"]] + 19.)
        np.testing.assert_allclose(bars.close, total.value)

    def test_synthetic_rebalance_uses_pretrade_units_and_cash(self):
        close = self.prices
        path = reconstruct(close, self.positions[["A", "B"]], 19., self.end,
                           targets=pd.Series({"A": .4, "B": .4, CASH: .2}), frequency="Daily")
        raw = {"Open": close.mul([.97, 1.02]), "High": close*1.07, "Low": close*.94, "Close": close}
        bars = synthetic_daily_ohlc(path, close, raw, close.index)
        units = path.marks[["A", "B"]].iloc[0] / close.iloc[0]
        self.assertAlmostEqual(bars.open.iloc[1], units @ raw["Open"].iloc[1] + path.marks[CASH].iloc[0])
        after_trade = path.marks[["A", "B"]].iloc[1] / close.iloc[1]
        self.assertNotAlmostEqual(bars.open.iloc[1], after_trade @ raw["Open"].iloc[1] + path.marks[CASH].iloc[1])
        np.testing.assert_allclose(bars.close, path.value)

    def test_currency_splits_total_return_and_unequal_history(self):
        native = self.prices.copy()
        native.loc[self.dates[:3], "B"] = np.nan
        native.loc[self.dates[7], "B"] = np.nan
        fx = pd.Series([.9, .8], index=[self.dates[1], self.dates[4]])
        converted = prices_in_currency(native.A, fx)
        self.assertTrue(pd.isna(converted.iloc[0]))
        self.assertAlmostEqual(converted.iloc[3], native.A.iloc[3]*.9)
        self.assertAlmostEqual(converted.iloc[4], native.A.iloc[4]*.8)
        self.assertTrue(pd.isna(converted.iloc[-1]))
        base = reconstruct(native, self.positions, 19., self.end)
        self.assertEqual(base.value.index[0], self.dates[3])
        self.assertNotIn(self.dates[7], base.value.index)
        # Provider Close is already split-normalized; an adjusted dividend ratio
        # changes prior levels only when anchored to the same final quote.
        close = pd.DataFrame({"A": [50., 50., 49., 51.]}, index=self.dates[:4])
        adjusted = pd.DataFrame({"A": [49., 49., 49., 51.]}, index=close.index)
        end = close.index[-1]
        total = reconstruct(adjusted.div(adjusted.loc[end]).mul(close.loc[end]),
                            pd.Series({"A": 2}), 0., end)
        self.assertEqual(total.value.loc[close.index[2]], 98)
        self.assertEqual(total.value.iloc[-1], 102)
        self.assertEqual(total.value.pct_change().iloc[2], 0)

    def test_allocation_order_and_weight_constraints(self):
        w, _, _ = current_model(self.prices, "Historical", "Equal Weight", .02, .6)
        table, cash = allocation(w, self.prices.iloc[-1], 100.)
        self.assertEqual(table.loc["A", "Quantity"], 0)
        self.assertEqual(table.loc["B", "Quantity"], 1)
        self.assertEqual(cash, 68.)
        with self.assertRaises(ValueError):
            current_model(self.prices, "Historical", "Equal Weight", .02, .4)


    def test_market_adapter_preserves_completed_split_basis(self):
        from src.time_series_market import daily_bars

        day = pd.Timestamp.now(tz="Europe/London").normalize()
        index = pd.date_range(end=day, periods=3)
        fixture = pd.DataFrame({"Open": [48., 49., 52.], "High": [52., 53., 52.], "Low": [47., 47., 52.],
                                "Close": [50., 50., 52.], "Adj Close": [49., 50., 52.],
                                "Stock Splits": [0., 0., 2.]}, index=index)
        with (patch("src.time_series_market.yf.Ticker", create=True) as provider,
              patch("src.time_series_market.exchange_session", return_value=("OPEN", None, None, None))):
            instrument = provider.return_value
            instrument.history.return_value = fixture
            instrument.history_metadata = {"currency": "GBp", "exchangeTimezoneName": "Europe/London"}
            bars, currency = daily_bars.__wrapped__("TEST.L")
        self.assertEqual(currency, "GBP")
        self.assertEqual(bars.Close.tolist(), [1., 1.])
        self.assertEqual(bars["Adj Close"].tolist(), [.98, 1.])
        np.testing.assert_allclose(bars.High, [1.04, 1.06])
        self.assertNotIn(day.tz_localize(None), bars.index)


class TimeSeriesAnalytics(unittest.TestCase):
    def test_candles_use_only_observed_portfolio_closes(self):
        dates = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05",
                                "2026-01-06", "2026-03-30", "2026-04-01"])
        value = pd.Series([100., 105., 98., 96., 102., 108., 110.], index=dates)
        weekly = portfolio_ohlc(value, "1W")
        np.testing.assert_allclose(weekly.iloc[0], [100., 105., 98., 98.])
        self.assertEqual(weekly.index[0], dates[2])
        self.assertEqual(weekly.index[1], dates[4])
        self.assertEqual(len(portfolio_ohlc(value, "1M")), 4)
        self.assertEqual(len(portfolio_ohlc(value, "1Q")), 3)
        with self.assertRaises(ValueError):
            portfolio_ohlc(value, "1D")

    def test_indicators_are_on_basket_and_causal(self):
        index = pd.bdate_range("2024-01-01", periods=400)
        t = np.arange(len(index))
        values = pd.Series(100 + 9*np.sin(t/8) + .02*t, index=index)
        params = ((20, 50), (12, 26), (21, 63), 14, (12, 26, 9), 20)
        stats = technical(values, *params)
        pd.testing.assert_frame_equal(stats.iloc[:300], technical(values.iloc[:300], *params))
        self.assertAlmostEqual(stats["SMA 20"].iloc[-1], values.tail(20).mean())
        zones = price_zones(values, 1., 10, 2, 2., 8)
        scaled = price_zones(values*3, 1., 10, 2, 2., 8)
        self.assertFalse(zones.empty)
        np.testing.assert_allclose(scaled.Level, zones.Level*3)
        self.assertTrue((zones.Last <= values.index[-1]).all())
        self.assertTrue(price_zones(pd.Series(100., index=index)).empty)
        self.assertEqual(technical(pd.Series(100., index=index), *params).RSI.iloc[-1], 50.)
        risks = risk_dynamics(values, 21, .02)
        self.assertTrue(risks["EWMA"].iloc[:22].isna().all())
        self.assertTrue(risks.Drawdown.le(0).all())

    def test_breadth_weights_and_relative_normalization(self):
        dates = pd.bdate_range("2024-01-01", periods=220)
        prices = pd.DataFrame({"A": np.linspace(10, 15, 220), "B": np.linspace(15, 10, 220),
                               "C": np.linspace(10, 30, 220)}, index=dates)
        weights = pd.DataFrame({"A": .1, "B": .4, "C": 0., CASH: .5}, index=dates)
        equal, weighted, heat = breadth(prices, weights, 14, (30, 70), (12, 26, 9))
        self.assertEqual(equal["Above SMA 200"].iloc[-1], .5)
        self.assertAlmostEqual(weighted["Above SMA 200"].iloc[-1], .2)
        self.assertTrue(weighted.stack().dropna().between(0, 1).all())
        self.assertTrue(heat.loc["C"].isna().all())
        matched = relative_strength(prices.A, prices.B.iloc[3:], 20)
        self.assertEqual(matched.index[0], dates[3])
        self.assertEqual(matched.Portfolio.iloc[0], 100.)
        self.assertEqual(matched.Benchmark.iloc[0], 100.)


class TimeSeriesPage(unittest.TestCase):
    def test_page_and_window_isolation(self):
        from src import time_series_ui as ui

        dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=1500)
        rng = np.random.default_rng(8)
        close = pd.Series(100*np.exp(np.cumsum(rng.normal(.0003, .01, len(dates)))), index=dates)
        daily = pd.DataFrame({"Open": close*.98, "High": close*1.02, "Low": close*.96,
                              "Close": close, "Adj Close": close})
        script = """
            import pandas as pd
            import streamlit as st
            from src.time_series_ui import render_time_series_portfolio

            render_time_series_portfolio(
                pd.DataFrame({"ticker": ["AAA", "BBB"], "name": ["One", "Two"]}), "USA", [],
                lambda label, text, color: st.metric(label, text), lambda table, whole: table.style,
                lambda ticker: ticker,
            )
        """
        app = AppTest.from_string(script)
        ui.fit_basket.clear()
        real_fit = ui.fit_basket
        with (patch.object(ui, "reporting_history", return_value=(daily, daily, "USD", None)),
              patch.object(ui, "fit_basket", wraps=real_fit) as fit):
            app.run()
            self.assertFalse(app.exception)
            app.multiselect[0].set_value(["AAA", "BBB"])
            optimizer = next(widget for widget in app.selectbox if widget.label == "Ottimizzatore")
            optimizer.select("Equal Weight")
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.get("plotly_chart")), 1)
            def traces():
                return json.loads(app.get("plotly_chart")[0].proto.spec)["data"]

            def widget(elements, label):
                return next(w for w in elements if w.label == label)

            self.assertEqual(traces()[0]["type"], "candlestick")
            self.assertIn("Close/Open", traces()[0]["hovertemplate"])
            widget(app.selectbox, "Timeframe").select("1D").run()
            self.assertFalse(app.exception)
            self.assertEqual(traces()[0]["type"], "candlestick")
            self.assertEqual(traces()[0]["name"], "Synthetic OHLC")
            self.assertEqual(widget(app.selectbox, "Daily OHLC Method").value, "Synthetic OHLC")
            self.assertTrue(any("limiti teorici" in x.value for x in app.caption))
            widget(app.selectbox, "Daily OHLC Method").select("Synchronized Intraday OHLC").run()
            self.assertFalse(app.get("plotly_chart"))
            self.assertTrue(any("intraday sincronizzato" in x.value for x in app.info))
            widget(app.selectbox, "Daily OHLC Method").select("Close Only").run()
            self.assertEqual(traces()[0]["type"], "scatter")
            widget(app.selectbox, "Timeframe").select("1M")
            widget(app.selectbox, "Chart Type").select("Area Chart")
            widget(app.selectbox, "Frequenza indicatori").select("Timeframe del grafico")
            widget(app.checkbox, "RSI sincronizzato").check()
            widget(app.checkbox, "MACD sincronizzato").check()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(traces()[0]["fill"], "tozeroy")
            self.assertTrue(any(t["name"] == "RSI · mesi" for t in traces()))
            self.assertTrue(any(t["name"] == "MACD Histogram · mesi" for t in traces()))
            self.assertEqual(fit.call_count, 1)
            original = app.session_state["ts_basket"]["weights"].copy()
            app.selectbox(key="ts_reconstruction").select("1 anno").run()
            self.assertFalse(app.exception)
            pd.testing.assert_series_equal(app.session_state["ts_basket"]["weights"], original)
            self.assertEqual(fit.call_count, 1)
            app.radio[0].set_value("Base 100").run()
            self.assertFalse(app.exception)
            pd.testing.assert_series_equal(app.session_state["ts_basket"]["weights"], original)
            self.assertEqual(fit.call_count, 1)
            app.selectbox(key="ts_estimation").select("2 anni").run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(fit.call_args.args[7], "2 anni")
            for label in ("Technical Analysis", "Momentum Analysis", "Risk Dynamics", "Market Breadth",
                          "Portfolio Attribution", "Benchmark Analysis", "K — AI Portfolio Analyst"):
                app.session_state["ts_portfolio_analysis"] = label
                app.run()
                self.assertFalse(app.exception, label)
            self.assertTrue(any(button.label == "Chiedi un’analisi a K" for button in app.button))

    def test_registered_navigation(self):
        source = Path("app_core.py").read_text()
        syntax = ast.parse(source)
        self.assertTrue(any(isinstance(node, ast.Constant) and node.value == "Time Series Portfolio"
                            for node in ast.walk(syntax)))
        self.assertTrue(any(isinstance(node, ast.If) and ast.unparse(node.test) == "time_series_tab.open"
                            for node in syntax.body))
