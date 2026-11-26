"""Current allocation and retrospective basket accounting."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.factors import capm_expected_returns, calculate_capm
from src.portfolio_context import check_minimum
from src.factor_currency import estimate_factor_returns, factors_for_backtest_signal
from src.ml_estimators import horizon_to_daily, optimize_inputs, return_fit, return_panel
from src.optimization import risk_contributions
from src.returns import annualized_returns

SESSIONS = 252
CASH = "Cash"
FREQUENCIES = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q",
               "Semiannual": None, "Annual": "Y"}


def prices_in_currency(price, fx, *, prior_day=False):
    """At most four calendar days of prior FX; never use a later rate."""
    valid = fx.where(np.isfinite(fx) & fx.gt(0)).dropna().sort_index()
    lag = int(prior_day)
    rate = valid.reindex(price.index - pd.Timedelta(days=lag), method="ffill",
                         tolerance=pd.Timedelta(days=4-lag))
    return price * rate.set_axis(price.index)


def current_model(prices, model, optimizer, rf, cap, *, market_returns=None,
                  factors=None, market=None, fx=None, reporting_fx=None, ml_model=None, configuration=None):
    prices = prices.dropna()
    returns = prices.pct_change(fill_method=None).dropna()
    if len(returns) < 2:
        raise ValueError("La finestra di stima richiede almeno due rendimenti comuni.")
    covariance = returns.cov() * SESSIONS
    if configuration and model == "CAPM":
        expected = calculate_capm(returns, None, risk_free_rate=rf)["CAPM Expected Return"]
    elif configuration and model in {"FF3", "FF5"}:
        expected = estimate_factor_returns(
            prices, factors_for_backtest_signal(pd.DataFrame(index=prices.index), prices.index[-1]),
            market="Borsa Italiana", model=model.lower(),
        )
    elif model == "CAPM":
        if market_returns is None or len(market_returns.dropna()) < 2:
            raise ValueError("Storico comune insufficiente per il benchmark CAPM.")
        expected = capm_expected_returns(returns, market_returns, rf)
    elif model in {"FF3", "FF5"}:
        native, raw = factors
        expected = estimate_factor_returns(
            native.dropna(), factors_for_backtest_signal(raw, prices.index[-1]), fx,
            market=market, model=model.lower(),
        ).reindex(prices.columns)
        if reporting_fx is not None:
            change = reporting_fx.reindex(prices.index).pct_change(fill_method=None).dropna()
            if change.empty:
                raise ValueError("Storico insufficiente per stimare il cambio di riferimento.")
            expected = ((1 + expected / SESSIONS) * (1 + change.mean()) - 1) * SESSIONS
    elif model == "ML":
        prediction = return_fit(return_panel(prices, 1), prices.index[-1], ml_model)[0]
        expected = pd.Series(horizon_to_daily(prediction.reindex(prices.columns), 1) * SESSIONS,
                             index=prices.columns)
    else:
        expected = annualized_returns(returns)
    if not np.isfinite(expected).all():
        raise ValueError("Il modello non ha prodotto stime valide per tutti gli asset.")
    if cap * len(expected) < 1 - 1e-9:
        raise ValueError("Il limite per asset non consente di investire il 100% del capitale target.")
    if optimizer == "Equal Weight" or len(expected) == 1:
        weights = pd.Series(1 / len(expected), index=expected.index)
    else:
        weights = optimize_inputs(expected, covariance, optimizer, rf, cap)
        weights = weights.clip(lower=0) / weights.clip(lower=0).sum()
    if configuration:
        check_minimum(weights)
    return weights, expected, covariance


def allocation(weights, prices, capital):
    current = prices.reindex(weights.index)
    if weights.empty or not np.isfinite(weights).all():
        raise ValueError("Pesi non validi per l’allocazione.")
    if (not np.isfinite(current).all() or current.le(0).any() or not np.isfinite(capital)
            or capital <= 0 or weights.lt(0).any() or not np.isclose(weights.sum(), 1.)):
        raise ValueError("Prezzi, pesi o capitale non validi per un portafoglio long-only.")
    target = weights * capital
    qty = np.floor(target / current).astype("int64")
    invested = qty * current
    table = pd.DataFrame({"Weight": weights, "Target Amount": target, "Price": current,
                          "Quantity": qty, "Invested": invested, "Effective weight": invested / capital})
    return table.sort_values("Invested", ascending=False), float(max(0., capital - invested.sum()))


def period_codes(index, frequency):
    if frequency == "Semiannual":
        return index.year * 2 + (index.month > 6).astype(int)
    return index.to_period(FREQUENCIES[frequency])


@dataclass
class PortfolioPath:
    value: pd.Series
    marks: pd.DataFrame
    weights: pd.DataFrame
    contributions: pd.DataFrame
    rebalanced: pd.Series


def reconstruct(prices, positions, cash, end, *, targets=None, frequency=None):
    """Close-to-close basket. A rebalance at close affects the next return."""
    exposure = positions[positions.gt(0)] if targets is None else targets.drop(labels=CASH, errors="ignore")
    active = exposure.index[exposure.gt(0)]
    if active.empty:
        raise ValueError("Il capitale non consente di acquistare asset in questa modalità.")
    close = prices.loc[:end, active]
    close = close.where(np.isfinite(close) & close.gt(0)).dropna()
    if len(close) < 2 or close.index[-1] != end:
        raise ValueError("Servono almeno due chiusure comuni, inclusa la data dell’allocazione.")
    flags = pd.Series(False, index=close.index)
    if targets is None:
        marks = close.mul(positions).loc[:, active].assign(Cash=cash)
        value = marks.sum(axis=1)
        weights = marks.div(value, axis=0)
    else:
        mix = targets.reindex([*active, CASH], fill_value=0.)
        if frequency not in FREQUENCIES or mix.lt(0).any() or not np.isclose(mix.sum(), 1.):
            raise ValueError("Frequenza o pesi target non validi.")
        codes = period_codes(close.index, frequency)
        flags.iloc[:-1] = codes[:-1] != codes[1:]
        flags.iloc[-1] = frequency == "Daily"
        relative = close.pct_change(fill_method=None).add(1).assign(Cash=1.)
        post_marks = np.empty((len(close), len(mix)))
        post_marks[0] = mix
        for i in range(1, len(close)):
            before_trade = post_marks[i - 1] * relative.iloc[i].to_numpy()
            post_marks[i] = before_trade.sum() * mix.to_numpy() if flags.iloc[i] else before_trade
        marks = pd.DataFrame(post_marks, index=close.index, columns=mix.index)
        marks *= (cash + (positions * prices.loc[end]).sum()) / marks.iloc[-1].sum()
        value = marks.sum(axis=1)
        weights = marks.div(value, axis=0)
    component_returns = close.pct_change(fill_method=None).assign(Cash=0.)
    return PortfolioPath(value, marks, weights, weights.shift() * component_returns, flags)


def linked_attribution(contributions, returns, frequency):
    """Forward-link daily contributions by wealth accumulated before each return."""
    codes = period_codes(returns.index, frequency)
    wealth = (1 + returns).groupby(codes).cumprod().groupby(codes).shift().fillna(1.)
    return contributions.mul(wealth, axis=0).groupby(codes).sum(min_count=1)


def endpoint_risk(weights, covariance):
    active = weights.drop(labels=CASH, errors="ignore")
    if active.empty or not active.gt(0).any():
        return pd.Series(dtype=float)
    cov = covariance.reindex(index=active.index, columns=active.index)
    if active @ cov @ active <= 0:
        return pd.Series(dtype=float)
    return risk_contributions(active, cov)


def synthetic_daily_ohlc(path, closes, asset_bars, dates):
    """Value the holdings in force during each session using asset daily bars."""
    held = path.marks.drop(columns=CASH)
    close = closes.reindex_like(held)
    # End-of-close marks reflect any new targets. The next session inherits
    # those quantities and cash. At inception use the initialized basket.
    units = (held / close).shift()
    units.iloc[0] = held.iloc[0] / close.iloc[0]
    cash = path.marks[CASH].shift()
    cash.iloc[0] = path.marks[CASH].iloc[0]
    units, cash = units.reindex(dates), cash.reindex(dates)
    close = close.reindex(dates)
    raw = {field: asset_bars[field].reindex_like(close) for field in ("Open", "High", "Low", "Close")}
    valid = (raw["Low"].gt(0) & raw["Open"].ge(raw["Low"]) & raw["Open"].le(raw["High"])
             & raw["Close"].ge(raw["Low"]) & raw["Close"].le(raw["High"])
             & np.isfinite(raw["High"]))
    # Put the bar on the reconstruction's basis. In Total Return this applies
    # the anchored adjustment factor once, identically to all four prices.
    ratio = close / raw["Close"].where(valid)
    out = pd.DataFrame({
        field.lower(): (units * (raw[field] * ratio)).where(units.ne(0), 0.)
        .sum(axis=1, min_count=len(units.columns)) + cash
        for field in ("Open", "High", "Low", "Close")
    }, index=dates).dropna()
    # A close rebalance transfers value between positions and preserves NAV.
    if not np.allclose(out.close, path.value.reindex(out.index), rtol=1e-10, atol=1e-8):
        raise ValueError("Le chiusure OHLC non si riconciliano con la serie ricostruita.")
    return out
