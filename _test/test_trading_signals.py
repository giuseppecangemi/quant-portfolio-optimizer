import tempfile
import unittest

import numpy as np
import pandas as pd

from src.trading_signals import Store, coverage_tests, forecast, indicators, monitor, paper_orders, signal, tail


class TradingSignalsTests(unittest.TestCase):
    def test_activation_and_isolation(self):
        orders, cash = paper_orders({"A": .6, "B": .4}, {"A": 17, "B": 13}, 100, 2, True)
        self.assertEqual([o["quantity"] for o in orders], [3, 2])
        self.assertEqual(cash, 19)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/journal.db"
            store = Store(path, "owner")
            store.put("p", [("portfolio","initial",{}),("transaction","0",orders[0])])
            store.put("p", [("portfolio","initial",{}),("transaction","1",orders[1])])
            self.assertEqual(Store(path,"owner").read("transaction"), [("p","0",orders[0])])
            self.assertEqual(Store(path,"other").read("transaction"), [])
        with self.assertRaises(ValueError):
            paper_orders({"A":1}, {}, 100, 0, False)

    def test_fixed_shares_and_dividend(self):
        dates = pd.date_range("2025-01-02", periods=3)
        close = pd.DataFrame({"A":[10.,12.,11.]}, index=dates)
        div = pd.DataFrame({"A":[0.,0.,1.]}, index=dates)
        splits = pd.DataFrame({"A":[0.,2.,0.]}, index=dates)
        order = dict(ticker="A", quantity=2., price=20., fee=1., date="2025-01-02")
        nav, r, dd, marks, qty, _ = monitor([order], close, div, splits, 10.)
        np.testing.assert_allclose(nav, [50,58,58])
        np.testing.assert_allclose(r, [-1/51,.16,0])
        self.assertAlmostEqual(marks.iloc[-1,0]/nav.iloc[-1],44/58)
        self.assertEqual(qty.iloc[-1,0],4)

    def test_tail_and_forecasts(self):
        np.testing.assert_allclose(tail(0,1,.95),[1.644853627,2.062712808])
        np.testing.assert_allclose(tail(0,1,.95,5),[1.560849758,2.238684256],rtol=1e-6)
        rng = np.random.default_rng(48)
        shocks = rng.standard_t(7, 1500)*np.sqrt(5/7)
        v, r = 1., []
        for shock in shocks:
            r.append(np.sqrt(v)*shock)
            v=.05+.1*r[-1]**2+.85*v
        for dist in ("normal","t"):
            f = forecast(pd.Series(r)/100, dist)
            self.assertIsNotNone(f)
            self.assertGreater(f["sigma"],0)
        self.assertIsNone(forecast(pd.Series(r[:50]),"normal"))
        self.assertIsNone(coverage_tests([0,1],.95))
        self.assertIsNotNone(coverage_tests([0]*19+[1]+[0]*39+[1]+([0]*19+[1])*20,.95))

    def test_causal_signals(self):
        p = pd.Series(np.linspace(100,50,230),index=pd.date_range("2025-01-01",periods=230))
        a = indicators(p)
        pd.testing.assert_frame_equal(a.iloc[:220],indicators(p.iloc[:220]))
        self.assertEqual(signal(a.iloc[:10])["state"],"INSUFFICIENT DATA")
        self.assertEqual(signal(a)["state"],"EXIT CANDIDATE")
