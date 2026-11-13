import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from src import screening


class ScreenerDownloadAudit(unittest.TestCase):
    def test_clean_and_fallback(self):
        dates = pd.bdate_range("2024-01-01", periods=101)
        raw = pd.Series(np.arange(101)+10., index=dates)
        raw.iloc[-1] = np.inf
        raw = pd.concat([raw, raw.iloc[:1]])
        cleaned = screening._clean_downloaded_prices(raw)
        self.assertEqual(len(cleaned), 100)
        self.assertEqual(screening._calculate_price_signals(cleaned)["momentum_window"], "available")

    def test_retry_classifies_failure_and_recovers(self):
        universe = pd.DataFrame({"ticker":["A.MI","B.MI","C.MI"],"name":["A","B","C"]})
        prices = pd.DataFrame({"Close":range(1,101)},index=pd.bdate_range("2024-01-01",periods=100))
        with patch.object(screening,"_download_price_batch",return_value=(pd.DataFrame(),"empty")), patch.object(screening.yf,"Ticker") as ticker:
            ticker.return_value.history.side_effect = [prices,pd.DataFrame(),TimeoutError("private")]
            got,status=screening.download_universe_prices(universe)
        self.assertEqual(list(got),["A.MI"])
        self.assertIn("Recovered",status["A.MI"])
        self.assertIn("Empty provider response",status["B.MI"])
        self.assertIn("TimeoutError",status["C.MI"])
        self.assertNotIn("private",status["C.MI"])

    def test_multiindex_both_orders(self):
        raw=pd.DataFrame([[1.,2.]],columns=pd.MultiIndex.from_tuples([("Close","A.MI"),("Close","B.MI")]))
        a=screening._extract_close_frame(raw,["A.MI","B.MI"])
        b=screening._extract_close_frame(raw.swaplevel(axis=1),["A.MI","B.MI"])
        pd.testing.assert_frame_equal(a,b)
