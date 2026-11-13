"""Reproducible Monte Carlo calibration export."""
import io
import json
import zipfile

import numpy as np
import pandas as pd


def build_audit_archive(source, strategy_keys, calibration, config, context=None):
    """Keep raw curve rows and exact calibration intervals as separate CSVs."""
    payload = io.BytesIO()
    strategies = []
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i, (label, key) in enumerate(strategy_keys.items()):
            curve = source[key]
            frame = curve if isinstance(curve, pd.DataFrame) else curve.to_frame("Portfolio Value")
            archive.writestr(f"nav_{i}.csv", frame.to_csv(index_label="Date"))
            log = curve.attrs.get("transaction_log")
            if isinstance(log, pd.DataFrame):
                archive.writestr(f"trades_{i}.csv", log.to_csv(index=True))
            weights = source.get(key.removesuffix("_bt") + "_weights_history")
            if isinstance(weights, pd.DataFrame):
                archive.writestr(f"weights_{i}.csv", weights.to_csv())
            strategies.append(dict(id=i, label=label, key=key,
                                   initial_capital=curve.attrs.get("initial_capital"),
                                   trade_log_available=isinstance(log, pd.DataFrame),
                                   columns=list(frame.columns)))
        archive.writestr("calibration_returns.csv", calibration.to_csv(index_label="Date"))
        positions = calibration.attrs.get("session_positions", {})
        archive.writestr("metadata.json", json.dumps({
            "schema_version": 1, "methodology": "daily-simple-net-v1",
            "numpy_version": np.__version__, "pandas_version": pd.__version__,
            "config": config, "context": context or {}, "strategies": strategies,
            "entry_multipliers": calibration.attrs.get("entry_multipliers", {}),
            "observed_calendar": [pd.Timestamp(d).isoformat() for d in positions],
        }, indent=2, allow_nan=False, default=str))
        archive.writestr("README.txt", "nav_N.csv contains the original backtest rows. Blank cells are missing. "
                         "Portfolio Value is net; Gross Portfolio Value is included where available.\n"
                         "calibration_returns.csv contains the exact aligned daily simple returns. "
                         "Entry multipliers are applied once to wealth, separately from daily shocks.\n"
                         "observed_calendar records source valuation dates, not an official exchange calendar. "
                         "Absent logs, cash flows, gross NAV and experiment identifiers are unknown.\n"
                         "Configuration describes the controls at export time, not a previously displayed run.\n")
    return payload.getvalue()
