import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from scipy.integrate import quad
from scipy.stats import norm, t

from src.return_distribution import (
    GAUSSIAN, HISTORICAL, LOGNORMAL, STUDENT, FittedDistribution,
    backtest_return_samples, distribution_statistics,
)
from src.return_distribution_ui import distribution_chart


class DistributionTests(unittest.TestCase):
    def test_inverse_quantile_and_fractional_es(self):
        _, stats = distribution_statistics(pd.Series(-np.arange(1., 31.)))
        np.testing.assert_allclose(stats["models"][HISTORICAL]["risk"][.95], (29., (30. + .5 * 29.) / 1.5))
        _, stats = distribution_statistics(pd.Series(-np.arange(1., 21.)))
        np.testing.assert_allclose(stats["models"][HISTORICAL]["risk"][.95], (19., 20.))
        _, stats = distribution_statistics(pd.Series([.1, .2, .3]))
        np.testing.assert_allclose(stats["models"][HISTORICAL]["risk"][.99], (-.1, -.1))

    def test_gaussian_mle_risk_and_information_criteria(self):
        raw = pd.Series([-.01, .01, .03, np.nan, np.inf])
        sample, stats = distribution_statistics(raw)
        gaussian = stats["models"][GAUSSIAN]
        fit = gaussian["fit"]
        sigma = np.sqrt(2 / 3) * .02
        self.assertEqual(len(sample), 3)
        self.assertEqual(stats["Excluded"], 2)
        self.assertAlmostEqual(fit.location, .01)
        self.assertAlmostEqual(fit.scale, sigma)
        self.assertAlmostEqual(stats["Volatility"], .02)
        np.testing.assert_allclose(gaussian["risk"][.95], [-.01 + sigma * 1.6448536269514722,
                                                         -.01 + sigma * 2.0627128075074257])
        likelihood = norm.logpdf(sample, .01, sigma).sum()
        self.assertAlmostEqual(gaussian["Log-likelihood"], likelihood)
        self.assertAlmostEqual(gaussian["AIC"], 4 - 2 * likelihood)
        self.assertAlmostEqual(gaussian["BIC"], 2 * np.log(len(sample)) - 2 * likelihood)
        self.assertTrue(np.isnan(stats["Excess kurtosis"]))
        self.assertTrue(np.isnan(stats["JB p-value"]))

    def test_student_mle_and_analytic_tail_against_integration(self):
        rng = np.random.default_rng(73)
        returns = pd.Series(.015 + .025 * rng.standard_t(5, 4500))
        _, stats = distribution_statistics(returns)
        row = stats["models"][STUDENT]
        fit = row["fit"]
        self.assertAlmostEqual(fit.location, .015, delta=.002)
        self.assertAlmostEqual(fit.scale, .025, delta=.003)
        self.assertAlmostEqual(fit.df, 5, delta=1.)
        self.assertAlmostEqual(row["BIC"], 3 * np.log(len(returns)) - 2 * row["Log-likelihood"])
        for confidence, (var, es) in row["risk"].items():
            p = 1 - confidence
            self.assertAlmostEqual(t.cdf(-var, fit.df, loc=fit.location, scale=fit.scale), p)
            q = t.ppf(p, fit.df)
            expected, _ = quad(lambda x: -(fit.location + fit.scale * x) * t.pdf(x, fit.df), -np.inf, q)
            self.assertAlmostEqual(es, expected / p, places=8)
        self.assertTrue(np.isposinf(FittedDistribution(STUDENT, .01, .02, 1).risk(.95)[1]))
        self.assertTrue(np.isfinite(FittedDistribution(STUDENT, .01, .02, 1.5).risk(.99)).all())

    def test_lognormal_mle_jacobian_and_truncated_expectation(self):
        log_returns = np.linspace(-.4, .6, 99)
        raw = pd.Series(np.expm1(log_returns))
        sample, stats = distribution_statistics(raw)
        row = stats["models"][LOGNORMAL]
        fit = row["fit"]
        self.assertAlmostEqual(fit.location, log_returns.mean())
        self.assertAlmostEqual(fit.scale, log_returns.std(ddof=0))
        self.assertAlmostEqual(row["Log-likelihood"],
                               (norm.logpdf(log_returns, fit.location, fit.scale) - log_returns).sum())
        self.assertAlmostEqual(row["AIC"], 4 - 2 * row["Log-likelihood"])
        for confidence, (var, es) in row["risk"].items():
            z = norm.ppf(1 - confidence)
            self.assertAlmostEqual(var, -np.expm1(fit.location + fit.scale * z))
            integral, _ = quad(lambda x: -np.expm1(fit.location + fit.scale * x) * norm.pdf(x), -np.inf, z)
            self.assertAlmostEqual(es, integral / (1 - confidence))
        for family in (GAUSSIAN, STUDENT, LOGNORMAL):
            model = stats["models"][family]
            self.assertAlmostEqual(model["Log-likelihood"], model["fit"].log_density(sample).sum())
        self.assertNotIn("AIC", stats["models"][HISTORICAL])
        self.assertTrue(np.isneginf(fit.log_density([-1.])[0]))
        pd.testing.assert_series_equal(sample, raw)

    def test_invalid_support_outliers_and_optimizer_failure(self):
        raw = pd.Series(np.r_[np.linspace(-.05, .07, 200), -1., -.999999, 3.])
        sample, stats = distribution_statistics(raw)
        self.assertNotIn(LOGNORMAL, stats["models"])
        self.assertIn("−100%", stats["unavailable"][LOGNORMAL])
        self.assertEqual(len(sample), len(raw))
        self.assertTrue(np.isfinite(stats["models"][GAUSSIAN]["risk"][.99]).all())
        with patch("src.return_distribution.minimize", side_effect=RuntimeError("fit failed")):
            _, stats = distribution_statistics(pd.Series(np.linspace(-.1, .2, 50)))
        self.assertEqual(stats["unavailable"][STUDENT], "fit failed")
        self.assertIn(LOGNORMAL, stats["models"])

    def test_normality_diagnostics_on_large_sample(self):
        sample = pd.Series(np.random.default_rng(19).standard_normal(2500))
        _, stats = distribution_statistics(sample)
        self.assertTrue(np.isfinite(stats["JB statistic"]))
        self.assertTrue(0 <= stats["JB p-value"] <= 1)

    def test_constant_and_insufficient(self):
        sample, stats = distribution_statistics(pd.Series([.01] * 12))
        self.assertEqual(stats["Volatility"], 0.)
        self.assertTrue(np.isnan(stats["Skewness"]))
        self.assertEqual(set(stats["models"]), {HISTORICAL})
        self.assertEqual(len(stats["unavailable"]), 3)
        self.assertIsNone(distribution_statistics(pd.Series([np.nan, .02]))[1])
        self.assertIsNone(distribution_statistics(pd.Series(dtype=float))[1])

    def test_backtest_alignment_and_input_immutability(self):
        dates = pd.bdate_range("2026-04-01", periods=7)
        curve = pd.DataFrame({"Daily Return": [-.001, .02, .03, .04, -.02],
                              "Portfolio Value": 100.}, index=dates[1:-1])
        prices = pd.DataFrame({"A": [50., 100., 110., np.nan, 120., 150., 500.],
                               "B": [50., 100., 105., 107., 111., 115., 500.]}, index=dates)
        prior = prices.copy(deep=True)
        p, a = backtest_return_samples(curve, prices, pd.DataFrame(columns=["A"]))
        pd.testing.assert_series_equal(p, curve["Daily Return"])
        pd.testing.assert_index_equal(a.index, curve.index)
        np.testing.assert_allclose(a["A"], [np.nan, .1, np.nan, np.nan, .25])
        sample, stats = distribution_statistics(p)
        fig = distribution_chart(sample, stats, GAUSSIAN, .95, (GAUSSIAN,))
        self.assertEqual(fig.layout.xaxis.tickformat, ".1%")
        self.assertAlmostEqual(sum(fig.data[0].y * fig.data[0].width * 100), 1.)
        self.assertEqual(fig.data[2].x[0], -stats["models"][GAUSSIAN]["risk"][.95][0])
        pd.testing.assert_frame_equal(prices, prior)
        pd.testing.assert_series_equal(sample, curve["Daily Return"])

    def test_overlay_and_tail_zoom(self):
        series = pd.Series(np.random.default_rng(3).normal(.001, .02, 1000))
        sample, stats = distribution_statistics(series)
        fits = (GAUSSIAN, STUDENT, LOGNORMAL)
        full = distribution_chart(sample, stats, STUDENT, .95, fits)
        zoom = distribution_chart(sample, stats, LOGNORMAL, .99, fits, True)
        self.assertEqual(tuple(trace.name for trace in full.data[1:4]), fits)
        self.assertLess(zoom.layout.xaxis.range[1], full.layout.xaxis.range[1])
        self.assertAlmostEqual(zoom.data[4].x[0], -stats["models"][LOGNORMAL]["risk"][.99][0])

    def test_strategy_switch_and_assets_in_streamlit(self):
        app = AppTest.from_string("""
            import numpy as np
            import pandas as pd
            from src.component_var_ui import render_dynamic_component_var

            dates = pd.bdate_range("2026-01-01", periods=15)
            prices = pd.DataFrame({k: 100 + np.cumsum(np.cos(np.arange(15) + j))
                                   for j, k in enumerate("ABC")}, index=dates)
            weights = pd.DataFrame([[.3, .3, .4]], columns=list("ABC"))
            first = pd.DataFrame({"Daily Return": .01, "Portfolio Value": 100.}, index=dates[3:])
            second = pd.DataFrame({"Daily Return": -.02, "Portfolio Value": 100.}, index=dates[4:])
            for curve in (first, second):
                curve.attrs["risk_position_marks"] = prices.loc[curve.index]
            result = {"max_sharpe_bt": first, "max_sharpe_weights_history": weights,
                      "ff3_bt": second, "ff3_weights_history": weights, "risk_prices": prices}
            render_dynamic_component_var(result, "dist", lambda ticker: "Company " + ticker)
        """).run()
        self.assertFalse(app.exception)
        self.assertIn("1.00%", [m.value for m in app.metric if m.label == "Mean daily return"])
        app.number_input(key="component_var_window_dist").set_value(2).run()
        self.assertEqual(len(app.get("plotly_chart")), 2)
        app.selectbox(key="component_var_strategy_dist").select("FF3 Maximum Sharpe").run()
        self.assertFalse(app.exception)
        self.assertIn("-2.00%", [m.value for m in app.metric if m.label == "Mean daily return"])
        app.selectbox(key="dist_confidence_dist").select(.99).run()
        self.assertIn("2.00%", [m.value for m in app.metric if m.label == "VaR 99%"])
        app.date_input[0].set_value((pd.Timestamp("2026-01-08").date(),
                                     pd.Timestamp("2026-01-16").date())).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("7 daily observations" in caption.value for caption in app.caption))
        app.session_state["dist_tab_dist"] = "Asset Distribution"
        app.run()
        self.assertFalse(app.exception)
        app.multiselect[0].set_value(["A", "B", "C"])
        app.session_state["dist_tab_dist"] = "Asset Distribution"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 4)
        self.assertEqual(len(app.dataframe[-1].value), 3)
