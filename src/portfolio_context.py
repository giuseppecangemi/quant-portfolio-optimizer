"""Per-run settings for the existing portfolio pipeline."""
from contextlib import contextmanager
from contextvars import ContextVar

_settings = ContextVar("portfolio_settings", default=None)


@contextmanager
def portfolio_scope():
    token = _settings.set(None)
    try:
        yield
    finally:
        _settings.reset(token)


def configure_portfolio(configuration):
    _settings.set(configuration)


def portfolio_settings():
    return _settings.get()


def minimum_for(labels, minimum=None):
    if minimum is not None:
        return minimum
    config = portfolio_settings()
    if config is None:
        return None
    return {row["ticker"]: row["minimum"] for row in config["assets"]
            if row["minimum"] > 0 and row["ticker"] in labels}


class IncompatibleAllocation(ValueError):
    """A specific strategy cannot satisfy this portfolio's constraints."""


def check_minimum(weights):
    bounds = minimum_for(weights.index) or {}
    invalid = [ticker for ticker, lower in bounds.items() if not weights[ticker] >= lower - 1e-8]
    if invalid:
        raise IncompatibleAllocation("Individual minimum weights cannot be met for: " + ", ".join(invalid))
