"""Serializable execution provenance for independent Streamlit result views."""
from datetime import datetime, timezone
import json
from uuid import uuid4


def configuration_snapshot(values):
    return json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))


def new_attempt(section, snapshot):
    return {"run_id": uuid4().hex, "section": section, "configuration": snapshot,
            "timestamp": datetime.now(timezone.utc).isoformat(), "status": "CURRENT"}


def run_status(result, snapshot, attempt=None):
    if result is not None and result.get("status") == "FAILED":
        return "FAILED"
    if result is None:
        return "FAILED" if attempt and attempt.get("error") else None
    if (result["configuration"] != snapshot
            or attempt and result["run_id"] != attempt["run_id"]):
        return "STALE"
    return "CURRENT"
