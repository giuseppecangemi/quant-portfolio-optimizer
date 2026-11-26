import base64
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.component_var import dynamic_component_var


class ComponentVaRDisplayTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_string("""
            import pandas as pd
            from src.component_var_ui import render_dynamic_component_var

            dates = pd.bdate_range("2026-01-29", "2026-03-02")
            curve = pd.DataFrame({"Portfolio Value": 1000.}, index=dates)
            curve.attrs["risk_position_marks"] = pd.DataFrame({"A": 100., "B": 200.}, index=dates)
            result = {
                "ff3_bt": curve, "ff3_weights_history": pd.DataFrame({"A": [.33], "B": [.67]}),
                "risk_prices": pd.DataFrame({"A": 10., "B": 20.}, index=dates),
            }
            render_dynamic_component_var(result, "test", lambda asset: "Company " + asset)
        """)

    def test_ff3_frequencies_and_arrow_metadata(self):
        dates = pd.DatetimeIndex(["2026-01-30", "2026-02-02", "2026-02-03",
                                  "2026-02-27", "2026-03-02"])
        daily = pd.DataFrame({
            "Date": dates.take([0, 1, 1, 2, 2, 3, 4]),
            "Asset": pd.Series(["A", "A", "B", "A", "B", "B", "A"],
                               dtype=pd.StringDtype(storage="pyarrow")),
            "Source": pd.Categorical(["FF3"] * 7),
            "Invested Amount (EUR)": [100., 100., 200., 100., 200., 200., 100.],
            "Component VaR (EUR)": [10., 4., 6., np.nan, 6., 10., 0.],
            "Component VaR Contribution (%)": [100., 40., 60., np.nan, 60., 100., np.nan],
            "Total Portfolio VaR (EUR)": [10., 10., 10., 10., 10., 10., 0.],
        })
        original = daily.copy(deep=True)

        def values(data):
            if isinstance(data, list):
                return np.asarray(data, dtype=float)
            return np.frombuffer(base64.b64decode(data["bdata"]), dtype=np.dtype(data["dtype"]))

        with patch("src.component_var_ui.cached_dynamic_var", return_value=daily):
            for frequency, selected in [("Daily", [0, 1, 2, 3, 4]),
                                        ("Weekly", [0, 2, 3, 4]), ("Monthly", [0, 3, 4])]:
                with self.subTest(frequency=frequency):
                    self.app.run()
                    self.app.selectbox(key="component_var_frequency_test").select(frequency).run()
                    self.assertFalse(self.app.exception)
                    self.assertEqual(self.app.selectbox(key="component_var_strategy_test").value,
                                     "FF3 Maximum Sharpe")
                    fig = json.loads(self.app.get("plotly_chart")[0].proto.spec)
                    a, b = fig["data"]
                    self.assertEqual([a["name"], b["name"]], ["Company A (A)", "Company B (B)"])
                    pd.testing.assert_index_equal(pd.DatetimeIndex(a["x"]), dates.take(selected))
                    np.testing.assert_allclose(values(a["y"]), np.array([100., 40., np.nan, 0., np.nan])[selected])
                    np.testing.assert_allclose(values(b["y"]), np.array([0., 60., 60., 100., np.nan])[selected])
                    np.testing.assert_allclose(values(a["customdata"]), np.array([10., 4., np.nan, 0., 0.])[selected])
                    np.testing.assert_allclose(values(b["customdata"]), np.array([0., 6., 6., 10., 0.])[selected])
        pd.testing.assert_frame_equal(daily, original)

    def test_empty_estimates_display_message(self):
        with patch("src.component_var_ui.cached_dynamic_var", return_value=pd.DataFrame()):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.get("plotly_chart"))
        self.assertIn("No dates have 252 complete daily returns", self.app.info[0].value)

    def test_dynamic_estimates_reconcile_on_available_dates(self):
        dates = pd.bdate_range("2026-02-02", periods=7)
        prices = pd.DataFrame({"A": [10., 12., 11., 10., 13., 14., 15.],
                               "B": [15., 14., 17., 18., 19., np.nan, 20.]}, index=dates)
        marks = pd.DataFrame({"A": 100., "B": 200.}, index=dates)
        marks.loc[dates[2], "B"] = 0.  # A valid date where only A is invested.
        marks.loc[dates[4]] = 0.      # Skipped because nobody is invested.
        daily = dynamic_component_var(marks, prices, window=2)
        self.assertEqual(daily["Date"].unique().tolist(), [dates[2], dates[3]])
        self.assertEqual(daily["Asset"].tolist(), ["A", "A", "B"])
        by_date = daily.groupby("Date")
        np.testing.assert_allclose(
            by_date["Component VaR (EUR)"].sum(),
            by_date["Total Portfolio VaR (EUR)"].first(), rtol=1e-12, atol=1e-12,
        )
