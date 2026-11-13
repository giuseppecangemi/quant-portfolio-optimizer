import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.historical_activation import PortfolioConfiguration, reconstruct


class HistoricalConfigurationTests(unittest.TestCase):
    def test_configuration_round_trip(self):
        config = PortfolioConfiguration(dict(
            universe=["A.MI", "B.MI"], return_model="Historical", period="2y",
            screening_period="2y", selection_mode="Quant Selection", top_n=1,
            screening_market_matches=True, active_factors=["Momentum", "Risk"],
        ))
        self.assertEqual(json.loads(json.dumps(config)), config)
        self.assertIsNone(config.replay_blocker())
        self.assertIsNotNone(PortfolioConfiguration(config, return_model="FF3").replay_blocker())

    def test_selection_uses_training_slice_and_imported_universe(self):
        class Calendar:
            name = "test"
            def date_to_session(self, day, direction):
                return day
            def previous_session(self, day):
                return day - pd.offsets.BDay()
            def sessions_in_range(self, start, end):
                return pd.bdate_range(start, end)
            def session_close(self, day):
                return day + pd.Timedelta(hours=17)

        execution = pd.Timestamp("2025-01-06")
        dates = pd.bdate_range("2022-12-01", execution)
        rng = np.random.default_rng(45)
        history = {}
        for ticker in ("A.MI", "B.MI"):
            prices = 100 * np.cumprod(1 + rng.normal(.001, .01, len(dates)))
            history[ticker] = pd.DataFrame({"Close": prices, "Adj Close": prices,
                                           "Stock Splits": 0}, index=dates)
        with patch("src.historical_activation.screen_prices_point_in_time",
                   return_value=(None, ["B.MI"])) as screen:
            snapshot, orders, _ = reconstruct(history, Calendar(), execution, 2, 1,
                                              "Maximum Sharpe", 10000, 3, True, 1, 0, 2)
        training = screen.call_args.args[0]
        self.assertEqual(list(training), ["A.MI", "B.MI"])
        self.assertLess(training.index.max(), execution)
        self.assertEqual(snapshot["weights"], {"B.MI": 1.0})
        self.assertEqual(orders[0]["date"], str(execution.date()))
