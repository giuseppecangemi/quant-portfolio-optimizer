"""Deterministic FF3 and FF5 contribution and reconciliation checks."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ff3_attribution import compute_ff3_attribution, compute_ff5_attribution


def test_factor_attribution():
    dates = pd.bdate_range("2025-01-06", periods=5)
    rates = pd.Series(0.0, index=dates)
    fx = pd.Series(1.1, index=dates)
    factors = pd.DataFrame({"MKT-RF": 0.01, "SMB": .002, "HML": -.003,
                            "RMW": .004, "CMA": -.005, "RF": 0.0}, index=dates).drop(dates[3])
    exposures = pd.DataFrame(
        [[1, 2, 3, 4, 5], [1, 2, 3, 8, 10]], index=dates[[0, 1]],
        columns=["Market Beta", "SMB Exposure", "HML Exposure", "RMW Exposure", "CMA Exposure"],
    )
    nav = pd.DataFrame({"Portfolio Value": [99., 100., 103., 102., 106.]}, index=dates)
    nav.attrs["initial_capital"] = 100
    ff3 = compute_ff3_attribution(nav, exposures, factors, fx, rates=rates)
    ff5 = compute_ff5_attribution(nav, exposures, factors, fx, rates=rates)
    for result in (ff3, ff5):
        assert result["covered_days"] == 3
        assert result["missing_dates"].equals(dates[[3]])
        assert np.isclose(result["daily"].loc[dates[3], "Unattributed days"], -.01)
        assert np.isclose(result["daily"].loc[dates[0], "Unexplained"], -.01)
        assert np.isclose(result["summary"].sum(), .06)
    assert "RMW" not in ff3["summary"] and "CMA" not in ff3["summary"]
    assert np.isclose(ff5["daily"].loc[dates[1], "RMW"], 4 * .004 * .99)
    assert np.isclose(ff5["daily"].loc[dates[2], "RMW"], 8 * .004)
    assert np.isclose(ff5["daily"].loc[dates[1], "CMA"], 5 * -.005 * .99)
    assert np.isclose(ff5["daily"].loc[dates[2], "CMA"], 10 * -.005)
    assert np.isclose(ff5["daily"].loc[dates[3], ["RMW", "CMA"]].sum(), 0)


if __name__ == "__main__":
    test_factor_attribution()
    print("PASS: FF3 and FF5 contribution, timing, missing day and net reconciliation")
