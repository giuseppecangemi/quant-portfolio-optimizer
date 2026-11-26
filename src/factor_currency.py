"""Market-aware Fama-French expected returns. Prices must be native currency."""
import pandas as pd
from src.eur_risk_free import ESTR_ONLY
from src.fama_french import fit_ff3
from src.factor_eur import convert_window_factors
from src.fama_french_5 import fit_ff5
from src.portfolio_context import portfolio_settings


def factors_for_backtest_signal(factors, signal_date):
    """Apply a two-calendar-month publication buffer to backtest estimates."""
    cutoff = pd.Timestamp(signal_date) - pd.DateOffset(months=2)
    sample = factors.loc[factors.index <= cutoff].copy()
    sample.attrs["publication_cutoff"] = cutoff.isoformat()
    return sample


def factor_regions(labels, *, config=None):
    config = portfolio_settings() if config is None else config
    if config is None:
        return None
    from src.hybrid_portfolio import GLOBAL

    default = "USA" if config["screening_market"] == "USA" else "Europe"
    mapping = {item["ticker"]: item["region"] for item in config["assets"]}
    regions = pd.Series({ticker: mapping.get(ticker, default) for ticker in labels})
    allowed = ["USA", "Europe", "Other developed"] if config["factor_policy"] == GLOBAL else ["USA", "Europe"]
    if not regions.isin(allowed).all():
        raise ValueError("Confirm a supported factor region for " + ", ".join(regions.index[~regions.isin(allowed)]))
    if config["factor_policy"] == GLOBAL:
        regions[:] = "Developed"
    return regions


def factor_covariance_available(labels):
    regions = factor_regions(labels)
    return regions is None or regions.nunique() == 1


def configured_factor_inputs(prices, factors, model, eurusd_prices=None, *, eur_rf_mode=None):
    from src.time_series_market import daily_bars, load_factors
    from src.time_series_portfolio import prices_in_currency

    config = portfolio_settings()
    expected_currency = config["base_currency"]
    market = "Borsa Italiana" if expected_currency == "EUR" else "USA"
    if prices.attrs.get("portfolio_currency") != expected_currency:
        raise ValueError("Factor estimation requires a certified common-currency price matrix.")
    cutoff = factors.attrs.get("publication_cutoff")
    fx = eurusd_prices
    if market == "Borsa Italiana":
        fx = daily_bars("EURUSD=X")[0].Close if fx is None else fx
        if isinstance(fx, pd.DataFrame):
            fx = fx.iloc[:, 0]
        calendar = pd.date_range(prices.index[0]-pd.Timedelta(days=10), prices.index[-1], freq="B")
        fx = prices_in_currency(pd.Series(1., index=calendar), fx, prior_day=True).dropna()
    regions = factor_regions(prices.columns)
    for region, group in regions.groupby(regions):
        family = {"USA": "USA", "Europe": "Borsa Italiana", "Developed": "Developed"}[region]
        data = load_factors(model.upper(), family)
        if cutoff:
            data = data.loc[:pd.Timestamp(cutoff)]
        yield prices[group.index], data, fx, market, eur_rf_mode or config["eur_rf_mode"]


def factor_sample(prices, factors, eurusd_prices, *, market, eur_rf_mode=ESTR_ONLY):
    prices = prices.sort_index()
    if market == "Borsa Italiana":
        if eurusd_prices is None:
            raise ValueError("EURUSD required for European USD-denominated factors")
        available = convert_window_factors(factors, eurusd_prices, prices, eur_rf_mode=eur_rf_mode)
    elif market == "USA":
        available = factors.loc[factors.index < prices.index[-1]]
    else:
        raise ValueError(f"Market {market!r} needs an explicit factor/currency mapping")

    if market == "Borsa Italiana":
        previous_session = pd.Series(available.attrs["interval_starts"], dtype="datetime64[ns]").reindex(prices.index)
    else:
        sessions = prices.index.union(factors.index).sort_values()
        previous_session = sessions.to_series().shift().reindex(prices.index)
    previous_price_date = prices.index.to_series().shift()
    returns = prices.pct_change(fill_method=None).loc[previous_price_date.eq(previous_session)]
    returns = returns.dropna()
    common = returns.index.intersection(available.index)
    return returns.loc[common], available.loc[common]


def _fit_expected(prices, factors, eurusd_prices, market, model, eur_rf_mode):
    returns, available = factor_sample(prices, factors, eurusd_prices, market=market, eur_rf_mode=eur_rf_mode)
    stats = (fit_ff3 if model == "ff3" else fit_ff5)(returns, available)
    column = f"{model.upper()} Expected Return USD"
    expected = stats[column].reindex(prices.columns)
    if expected.isna().any():
        raise ValueError("Missing factor expected returns")
    if market == "Borsa Italiana":
        stats = stats.rename(columns={column: column.replace("USD", "EUR"),
                                      "Historical Return USD": "Historical Return EUR"})
    return expected, stats



def estimate_factor_returns(prices, factors, eurusd_prices=None, *, market, model, return_stats=False, eur_rf_mode=None):
    if model not in ("ff3", "ff5"):
        raise ValueError(f"Unsupported model: {model}")
    if portfolio_settings() is None:
        expected, stats = _fit_expected(prices, factors, eurusd_prices, market, model, eur_rf_mode or ESTR_ONLY)
    else:
        groups = tuple(_fit_expected(group, data, fx, region_market, model, rf)
                       for group, data, fx, region_market, rf in configured_factor_inputs(
                           prices, factors, model, eurusd_prices, eur_rf_mode=eur_rf_mode))
        expected = pd.concat([item[0] for item in groups]).reindex(prices.columns)
        stats = pd.concat([item[1] for item in groups]).reindex(prices.columns)
    return (expected, stats) if return_stats else expected
