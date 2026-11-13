"""Independent Monte Carlo checks. Run as a unittest module."""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.monte_carlo import aligned_returns, empirical_tail_mean, extract_equity, simulate


METHODS = ("Block bootstrap", "IID bootstrap", "Multivariate Gaussian")


class MonteCarloAudit(unittest.TestCase):
    def test_A_B_C_constants(self):
        for method in METHODS:
            for rate in (0., .001, -.002):
                m, p, f = simulate(pd.DataFrame({"A": [rate]*40}), method=method, n_paths=100)
                expected = 10000*(1+rate)**252
                np.testing.assert_allclose(f.A, expected, rtol=1e-12)
                self.assertEqual(p["A"].iloc[0, 0], 10000)
                self.assertEqual(m.iloc[0]["Probability MDD ≤ -20% (%)"], 100 if rate < 0 else 0)

    def test_D_E_F_I_L_M_bootstrap(self):
        history = np.linspace(-.03, .025, 40)**3
        frame = pd.DataFrame({"A":history, "B":history*2, "C":history})
        for method, length in ((METHODS[0], 5), (METHODS[1], 1)):
            _, _, f, paths = simulate(frame, method=method,n_paths=100,seed=12,return_paths=True)
            starts = np.random.default_rng(12).integers(0,41-length,size=(100,(252+length-1)//length))
            idx = np.array([[j for start in row for j in range(start,start+length)][:252] for row in starts])
            reference = np.array([10000*np.prod(1+history[row]) for row in idx])
            np.testing.assert_allclose(f.A, reference, rtol=1e-13)
            np.testing.assert_array_equal(f.A,f.C)
            np.testing.assert_allclose(f.B,[10000*np.prod(1+2*history[row]) for row in idx],rtol=1e-13)
            self.assertEqual(paths["A"].shape,(100,253))
            self.assertTrue(f.equals(simulate(frame,method=method,n_paths=100,seed=12)[2]))
            self.assertFalse(f.equals(simulate(frame,method=method,n_paths=100,seed=13)[2]))

    def test_G_H_missing_short_and_gap(self):
        d = pd.date_range("2025-01-01",periods=45)
        a = pd.Series(100*1.01**np.arange(45),index=d)
        b = a.drop(d[5])
        actual = aligned_returns({"A":a,"B":b})
        self.assertEqual(len(actual), 42)
        self.assertNotIn(d[5], actual.index)
        self.assertNotIn(d[6], actual.index)
        np.testing.assert_allclose(actual, .01)
        with self.assertRaisesRegex(ValueError, "Nessun blocco"):
            simulate(actual, n_paths=100, block_size=40)
        self.assertIsNone(extract_equity({"A":a.mask(a==a.iloc[3])},"A"))
        with self.assertRaises(ValueError):
            simulate(pd.DataFrame({"A":[0.]*39}), n_paths=100)

    def test_J_N_drift(self):
        for drift in (0.,.10,-.10):
            f = simulate(pd.DataFrame({"A":[0.]*40}),n_paths=100,annual_drift_adjustment=drift)[2]
            np.testing.assert_allclose(f.A,10000*(1+drift/252)**252,rtol=1e-12)

    def test_K_metrics_and_ties(self):
        h = pd.DataFrame({"A":[-.25]+[0.]*39})
        m, pct, f, paths = simulate(h,n_paths=101,method=METHODS[1],return_paths=True)
        x = sorted(f.A)
        q = x[5]
        lower = [v for v in x if v <= q]

        strict_es = 10000-(sum(x[:5])+.05*x[5])/5.05
        self.assertGreater(len(lower),5.05)
        self.assertAlmostEqual(m.iloc[0]["ES 95% (€)"],strict_es)
        self.assertAlmostEqual(m.iloc[0]["VaR 95% (€)"],10000-q)
        self.assertAlmostEqual(m.iloc[0]["Mean final (€)"],sum(x)/len(x))
        self.assertAlmostEqual(m.iloc[0]["Median final (€)"],x[50])
        self.assertAlmostEqual(m.iloc[0]["P5 final (€)"],q)
        self.assertAlmostEqual(m.iloc[0]["P95 final (€)"],x[95])
        self.assertAlmostEqual(m.iloc[0]["Probability of loss (%)"],100*sum(v<10000 for v in x)/len(x))
        depths = [min(v/max(path[:i+1])-1 for i,v in enumerate(path)) for path in paths["A"]]
        self.assertAlmostEqual(m.iloc[0]["Probability MDD ≤ -20% (%)"],100*np.mean(np.array(depths)<=-.2))
        self.assertAlmostEqual(pct["A"].iloc[-1].P5,q)
        print(f"ES threshold={m.iloc[0]['ES 95% (€)']:.6f}; exact-tail={strict_es:.6f}; tail rows={len(lower)}/101")

    def test_O_gaussian_moments_and_singularity(self):
        rng = np.random.default_rng(15)
        a = rng.normal(0,.01,252)
        frame = pd.DataFrame({"A":a,"B":.7*a+rng.normal(0,.004,252),"C":a})
        _, _, f, paths = simulate(frame,n_paths=500,method=METHODS[2],return_paths=True)
        shocks = np.column_stack([(p[:,1:]/p[:,:-1]-1).ravel() for p in paths.values()])
        np.testing.assert_allclose(shocks.mean(axis=0),frame.mean(),atol=1.2e-4)
        np.testing.assert_allclose(np.cov(shocks,rowvar=False),frame.cov(),rtol=.025)
        np.testing.assert_allclose(f.A,f.C,rtol=1e-7)
        print(f"Gaussian correlation input={frame.corr().iloc[0,1]:.5f}, simulated={np.corrcoef(shocks.T)[0,1]:.5f}")

    def test_dependence_sensitivity(self):
        rng=np.random.default_rng(415)
        x=np.zeros(252)
        for i in range(1,len(x)):
            x[i]=.45*x[i-1]+rng.standard_t(5)*(.018 if i>=126 else .006)
        frame=pd.DataFrame({"A":x,"B":x*.9+rng.normal(0,.003,252)})
        def lag(a):
            return np.corrcoef(a[...,1:].ravel(),a[...,:-1].ravel())[0,1]
        print(f"Synthetic n=252 skew={skew(x):.3f}, excess kurtosis={kurtosis(x):.3f}, lag1={lag(x):.3f}, squared lag1={lag(x*x):.3f}, half vol={x[:126].std():.4f}/{x[126:].std():.4f}")
        for size in (1,3,5,10,20):
            m,_,_,paths=simulate(frame,n_paths=500,block_size=size,seed=42,return_paths=True)
            p=paths["A"]
            r=p[:,1:]/p[:,:-1]-1
            print(f"L={size} lag1={lag(r):.3f} squared={lag(r*r):.3f} ES={m.iloc[0]['ES 95% (€)']:.2f}")
            self.assertGreater(lag(r),.10) if size>1 else self.assertLess(abs(lag(r)),.03)

    def test_input_validation(self):
        h = pd.DataFrame({"A": [0.]*40})
        for parameters in ({"annual_drift_adjustment": np.nan}, {"initial_capital": np.inf},
                           {"years": 1.5}, {"n_paths": 100.5}, {"seed": -1}, {"block_size": 0}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                simulate(h, **parameters)
        for bad in (np.nan, np.inf, -np.inf, -1.):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                simulate(pd.DataFrame({"A": [bad]*40}))
        with self.assertRaises(ValueError):
            simulate(pd.DataFrame(index=range(40)))

    def test_fractional_tail_small_samples(self):
        self.assertAlmostEqual(empirical_tail_mean([-8., -2., 5.], .05), -8.)
        self.assertAlmostEqual(empirical_tail_mean([-8., -2., -2., 5.], .375), -6.)

    def test_initial_cost_once_and_intermediate_cost_in_nav(self):
        dates = pd.bdate_range("2025-01-01", periods=253)
        nav = pd.DataFrame({"Portfolio Value": 9800 * 1.001 ** np.arange(253)}, index=dates)
        nav.attrs["initial_capital"] = 10000.
        returns = aligned_returns({"A": extract_equity({"curve": nav}, "curve")})
        self.assertAlmostEqual(returns.attrs["entry_multipliers"]["A"], .98)
        for method in METHODS:
            _, p, f = simulate(returns, method=method, n_paths=100)
            self.assertEqual(p["A"].Median.iloc[0], 10000.)
            np.testing.assert_allclose(f.A, nav.iloc[-1, 0], rtol=1e-12)
        nav.iloc[101:] *= .99
        returns = aligned_returns({"A": extract_equity({"curve": nav}, "curve")})
        self.assertAlmostEqual(returns.A.iloc[100], 1.001*.99-1)
        self.assertAlmostEqual(10000 * returns.attrs["entry_multipliers"]["A"]
                               * np.prod(1+returns.A), nav.iloc[-1, 0])

    def test_daily_risk_exact_pool(self):
        h = pd.DataFrame({"A": [-.10]+[-.02]*3+[.01]*36})
        h["B"] = h.A
        h.attrs["entry_multipliers"] = {"A": .98, "B": .98}
        for years, capital in ((1, 10000), (2, 20000)):
            n = 101
            m, _, f = simulate(h, method=METHODS[1], years=years, n_paths=n, initial_capital=capital)
            idx = np.random.default_rng(42).integers(0, 40, size=(n, 252*years))
            shocks = h.A.to_numpy()[idx]
            ordered = sorted(shocks.ravel())
            mass = len(ordered)*.05
            k = int(mass)
            es = -(sum(ordered[:k])+(mass-k)*ordered[k])/mass
            j = (len(ordered)-1)*.05
            q = ordered[int(j)] + (j-int(j))*(ordered[int(j)+1]-ordered[int(j)])
            self.assertAlmostEqual(m.loc["A","Daily ES 95% (€)"], capital*es)
            self.assertAlmostEqual(m.loc["A","Daily VaR 95% (€)"], -capital*q)
            self.assertGreaterEqual(m.loc["A","Daily ES 95% (€)"], m.loc["A","Daily VaR 95% (€)"])
            pd.testing.assert_series_equal(m.loc["A"], m.loc["B"], check_names=False)
            np.testing.assert_allclose(f.A, capital*.98*np.prod(1+shocks,axis=1))

    def test_export_roundtrip_and_calendar_gap(self):
        import io
        import json
        import zipfile
        from src.monte_carlo_export import build_audit_archive

        calendar = pd.bdate_range("2025-01-02", periods=50)
        d = calendar.delete(5)
        curve = pd.DataFrame({"Portfolio Value": 99.*1.01**np.arange(len(d)),
                              "Gross Portfolio Value": 100.*1.01**np.arange(len(d))}, index=d)
        curve.attrs["initial_capital"] = 100.
        source = {"a":curve}
        calibration = aligned_returns({"A": extract_equity(source,"a")})
        archive = build_audit_archive(source, {"A":"a"}, calibration, {"seed":42})
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            nav = pd.read_csv(z.open("nav_0.csv"),index_col="Date",parse_dates=True)
            returns = pd.read_csv(z.open("calibration_returns.csv"),index_col="Date",parse_dates=True)
            meta = json.load(z.open("metadata.json"))
        self.assertEqual(list(calendar.difference(nav.index)), [calendar[5]])
        # The observed-only production calendar still cannot exclude this shared gap.
        self.assertIn(calendar[6], returns.index)
        self.assertEqual(meta["config"]["seed"],42)
        self.assertEqual(meta["entry_multipliers"],{"A":.99})
        np.testing.assert_allclose(nav["Portfolio Value"], nav["Gross Portfolio Value"]*.99)
        np.testing.assert_allclose(returns.A, calibration.A)


if __name__ == "__main__":
    unittest.main()
