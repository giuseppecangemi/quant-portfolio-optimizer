import ast
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.vix_analytics import compare_markets, date_bounds, rolling_statistics
from src.vix_data import load_history, parse_history
from src.vix_ui import index_chart, initial_levels


class VixTest(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2023", periods=330)
        self.vix = pd.Series(15 + np.arange(len(dates)) / 100, index=dates, name="VIX")
        self.stoxx = (2 * self.vix.iloc[1:] + 1).rename("VSTOXX")

    def test_stats_and_prefix(self):
        values = pd.Series([4., 1., np.nan, 2., 3., 3.])
        frame = rolling_statistics(values, 3)
        self.assertTrue(frame["Mean"].iloc[:2].isna().all())
        row = frame.loc[4]
        self.assertAlmostEqual(row["Mean"], 2)
        self.assertAlmostEqual(row["Std"], 1)
        self.assertAlmostEqual(row["Z"], 1)
        self.assertEqual(row["Percentile"], 100)
        for k in (1, 2, 3):
            self.assertEqual(row[f"Lower {k}"], 2-k)
            self.assertEqual(row[f"Upper {k}"], 2+k)
        self.assertAlmostEqual(frame.loc[5, "Change"], 0)
        self.assertEqual(frame.loc[5, "Percentile"], 100)
        self.assertTrue(rolling_statistics(pd.Series([1., 1., 1.]), 2)["Z"].isna().all())
        pd.testing.assert_frame_equal(frame.iloc[:-1], rolling_statistics(values.iloc[:-1], 3))

    def test_comparison_and_dates(self):
        common, corr, spread = compare_markets(self.vix, self.stoxx, 21)
        self.assertTrue(common.index.equals(self.stoxx.index))
        self.assertAlmostEqual(corr.iloc[-1], 1)
        self.assertAlmostEqual(spread.iloc[-1], self.stoxx.iloc[-1]-self.vix.iloc[-1])
        for period in ("1M", "1Y", "MAX"):
            start, end = date_bounds(common.index, period)
            self.assertEqual(end, common.index[-1])
            self.assertTrue(start <= end)
        start, end = date_bounds(common.index, "Custom", ("2023-03-01", "2023-05-01"))
        self.assertEqual(start.month, 3)
        self.assertEqual(end.month, 5)

    def test_period_plot_and_custom_levels(self):
        frame = rolling_statistics(self.vix, 252)
        bounds = date_bounds(frame.index, "1M")
        levels = [{"Visible": True, "Level": 22., "Label": "Custom"},
                  {"Visible": False, "Level": 32., "Label": "Hidden"}]
        figure = index_chart("VIX", frame, bounds, True, [1, 2, 3], levels, True)
        self.assertEqual(tuple(figure.layout.xaxis.range), bounds)
        self.assertEqual(len(figure.layout.shapes), 1)
        self.assertEqual(figure.layout.shapes[0].y0, 22)
        self.assertTrue(figure.layout.xaxis.rangeslider.visible)
        self.assertTrue(pd.notna(frame.loc[bounds[0]:, "Mean"]).all())
        self.assertNotEqual(initial_levels("VIX", self.vix), initial_levels("VSTOXX", self.stoxx))

    def test_official_formats_and_network_errors(self):
        parsed = parse_history("VSTOXX", "Date;Symbol;Indexvalue\n06.01.1999;V2TX;25\n04.01.1999;V2TX;18\n05.01.1999;OTHER;12\n07.01.1999;V2TX;\n")
        self.assertEqual(parsed.tolist(), [18., 25.])
        loader = load_history.__wrapped__
        with patch("src.vix_data.urlopen", side_effect=[URLError("DNS"), BytesIO(b"DATE,CLOSE\n01/03/2024,16\n01/02/2024,15\n")]) as req:
            self.assertEqual(loader("VIX").tolist(), [15., 16.])
            self.assertEqual(req.call_count, 2)
            self.assertEqual(req.call_args.kwargs["timeout"], 10)
        with patch("src.vix_data.urlopen", side_effect=URLError("DNS")) as req:
            with self.assertRaisesRegex(ValueError, "STOXX daily history unavailable"):
                loader("VSTOXX")
            self.assertEqual(req.call_count, 2)

    def test_navigation_single_provider_and_complete_dashboard(self):
        path = Path(__file__).resolve().parents[1] / "app_core.py"
        source = path.read_text()
        main = next(node for node in ast.parse(source).body if isinstance(node, ast.With)
                    and "kangemi_main_navigation" in ast.unparse(node.items[0]))
        vix = next(node for node in ast.parse(source).body if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "vix_tab.open")
        script = "import streamlit as st\ncolored_metric = lambda label, value, color: st.metric(label, value)\n"
        script += ast.get_source_segment(source, main) + "\n" + ast.get_source_segment(source, vix)
        app = AppTest.from_string(script)
        with patch("src.vix_ui.load_history", side_effect=lambda key: self.vix if key == "VIX" else self.stoxx) as fetch:
            app.run()
            fetch.assert_not_called()
            tab = next(tab for tab in app.tabs if tab.label == "VIX")
            self.assertEqual(tab.label, "VIX")
            app.session_state["main_navigation"] = "VIX"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("plotly_chart")), 5)
        with patch("src.vix_ui.load_history", side_effect=[self.vix, ValueError("STOXX unavailable")]):
            app.session_state["main_navigation"] = "VIX"

            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("plotly_chart")), 1)
            self.assertEqual(app.warning[0].value, "STOXX unavailable")
