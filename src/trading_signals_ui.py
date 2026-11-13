"""Streamlit view for the prospective trading journal."""
import os
import re
from datetime import date, timedelta
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats

from src.trading_signals import Store, coverage_tests, forecast, indicators, monitor, paper_orders, signal, tail


def stage(weights, key, names, metadata):
    def transfer():
        st.session_state["trading_draft"] = {
            **metadata, "id": uuid4().hex, "model": key,
            "weights": {str(t): float(w) for t, w in weights.items() if w > 0},
            "names": dict(names),
        }
        st.session_state.pop("historical_preview", None)
        st.session_state.pop("ts_preview", None)
        st.session_state["main_navigation"] = "Trading Signals"
        st.session_state["ts_workspace"] = "Activate"
        st.session_state["ts_portfolio_type"] = "Historical Activation"

    st.button("Activate in Trading Signals", key=f"stage_trading_{key}", on_click=transfer)


@st.cache_data(ttl=900, show_spinner=False)
def quotes(tickers, start, currency):
    data = {}
    for ticker in tickers:
        if not re.fullmatch(r"[A-Z0-9.^=/-]{1,30}", ticker):
            raise ValueError(f"Invalid ticker: {ticker}")
        instrument = yf.Ticker(ticker)
        h = instrument.history(start=str(start), auto_adjust=False, actions=True, raise_errors=True)
        meta = instrument.history_metadata
        if meta.get("currency") != currency:
            raise ValueError(f"{ticker}: expected {currency}, provider reports {meta.get('currency')}.")
        h.index = h.index.tz_localize(None).normalize()
        # Only completed calendar days in the exchange's time zone.
        today = pd.Timestamp.now(tz=meta["exchangeTimezoneName"]).date()
        h = h[h.index.date < today]
        if h.empty:
            raise ValueError(f"No completed quotes for {ticker}.")
        data[ticker] = h
    return data


def open_store():
    try:
        config = st.secrets.get("trading_signals", {})
    except FileNotFoundError:
        config = {}
    location = os.getenv("TRADING_SIGNALS_DATABASE_URL") or config.get("database_url")
    local = os.getenv("TRADING_SIGNALS_LOCAL_SQLITE")
    user = getattr(st, "user", None)
    if location and user is not None and user.get("is_logged_in") and user.get("iss") and user.get("sub"):
        return Store(location, f"{user['iss']}|{user['sub']}")
    if local and not location:
        return Store(local, "explicit-single-user-local")
    return None


def historical_activation(store):
    from src.historical_activation import reconstruct

    from src.historical_activation import PortfolioConfiguration

    draft = st.session_state.get("trading_draft") or {}
    configuration = draft.get("configuration")
    if not draft:
        st.info("Run Portfolio Optimization or Factor Models, then click Activate in Trading Signals to import the configuration.")
        return
    if not configuration:
        st.warning("This saved result predates configuration capture. Rerun its source analysis once to enable historical activation.")
        return
    config = PortfolioConfiguration(configuration)
    st.json(configuration, expanded=False)
    if reason := config.replay_blocker():
        st.warning(reason)
        return
    st.warning("The imported candidate universe may omit former listings and include later entrants. Yahoo quotes can contain revisions. This reconstruction is subject to survivorship bias.")
    with st.form("historical_activation"):
        name = st.text_input("Portfolio name", draft["model"])
        purchase = st.date_input("Virtual purchase date", max_value=date.today()-timedelta(days=1))
        capital = st.number_input("Virtual capital (EUR)", min_value=1., value=10000.)
        commission = st.number_input("Commission per order (EUR)", min_value=0., value=0.)
        whole = st.checkbox("Whole shares", value=True)
        submitted = st.form_submit_button("Calculate Historical Portfolio")
    if submitted:
        st.session_state.pop("historical_preview", None)
        import exchange_calendars as xcals

        universe = tuple(configuration["universe"])
        years = int(configuration["period"][:-1])
        quant = configuration["selection_mode"] == "Quant Selection"
        screening_years = int(configuration["screening_period"][:-1]) if quant else None
        calendar = xcals.get_calendar("XMIL")
        start = pd.Timestamp(purchase)-pd.DateOffset(years=max(years, screening_years or 0))-pd.Timedelta(days=14)
        history = {}
        unavailable = []
        for ticker in universe:
            try:
                history.update(quotes((ticker,), start.date(), "EUR"))
            except Exception:
                unavailable.append(ticker)
        if unavailable and not quant:
            raise ValueError("Quotes are unavailable for: " + ", ".join(unavailable))
        snapshot, orders, cash = reconstruct(
            history, calendar, purchase, years, configuration["top_n"] if quant else None,
            configuration["optimizer"], capital, commission, whole, configuration["max_weight"],
            configuration["risk_free_rate"], screening_years, tuple(configuration["active_factors"]),
        )
        snapshot["configuration"] = configuration
        snapshot["unavailable_quotes"] = unavailable
        meta = dict(name=name, mode="Historical Paper", historical=True, currency="EUR",
                    capital=capital, cash=cash, start=orders[0]["date"], model=snapshot["model"],
                    created=pd.Timestamp.now(tz="UTC").isoformat())
        st.session_state["historical_preview"] = draft["id"], uuid4().hex, snapshot, orders, meta
    preview = st.session_state.get("historical_preview")
    if preview is None or preview[0] != draft["id"]:
        return
    _, identifier, snapshot, orders, meta = preview
    st.caption(f"Decision {snapshot['decision_timestamp']} · Simulated execution at next session close {snapshot['execution_timestamp']} · Residual cash €{meta['cash']:,.2f}")
    sharpe = snapshot["sharpe"]
    sharpe_text = f"{sharpe:.2f}" if sharpe is not None else "—"
    st.caption(f"{snapshot['model']} · Expected return {snapshot['expected_return']:.2%} · "
               f"Volatility {snapshot['volatility']:.2%} · Sharpe {sharpe_text}")
    if snapshot["unavailable_quotes"]:
        st.warning("Quotes unavailable for: " + ", ".join(snapshot["unavailable_quotes"]))
    st.dataframe(pd.Series(snapshot["weights"], name="Target weight"))
    st.dataframe(orders)
    if st.button("Confirm Virtual Purchase", disabled=store is None):
        store.put(identifier, [("portfolio", "initial", meta), ("snapshot", "initial", snapshot),
                               *[("transaction", str(i), o) for i, o in enumerate(orders)]])
        del st.session_state["historical_preview"]
        st.session_state["historical_active"] = identifier
        st.session_state["ts_open_monitor"] = True
        st.rerun()


PAPER_BUY = "Activation — Buy Portfolio"
PAPER_ANALYSIS = "Analysis — Evaluate Entry"


def entry_analysis(draft, start, currency):
    st.caption("Retrospective synthetic index using the imported weights, with theoretical daily rebalancing. "
               "The displayed index starts at 1. This analysis uses no orders, commissions or invested capital.")
    if not st.button("Run Pre-Trade Analysis"):
        return
    weights = pd.Series(draft["weights"], dtype=float)
    history = quotes(tuple(weights.index), start - timedelta(days=730), currency)
    prices = pd.DataFrame({ticker: history[ticker]["Adj Close"] for ticker in weights.index}).sort_index()
    returns = prices.pct_change(fill_method=None).mul(weights).sum(axis=1, min_count=len(weights))
    # Restart after a gap rather than connect levels across unknown returns.
    gaps = returns[returns.isna()].index
    if len(gaps):
        returns = returns.loc[gaps[-1]:].iloc[1:]
    index = pd.concat([pd.Series(1., index=gaps[-1:]), (1 + returns).cumprod()])
    visible = index.loc[str(start):]
    if visible.empty:
        st.warning("No common price history is available for the selected analysis dates.")
        return
    if len(gaps) and gaps[-1].date() >= start:
        st.warning("Incomplete quotes shortened the displayed history to the latest continuous segment.")
    data = indicators(index)
    display = data.loc[visible.index]
    base = visible.iloc[0]
    realized = returns.rolling(21).std() * np.sqrt(252)
    initial = draft.get("volatility")
    render_entry_assessment(data, realized, initial)
    render_portfolio_charts(display, base, realized, returns, initial, start,
                            "Imported weights · displayed base 1")


def render_portfolio_charts(display, base, realized, returns, initial, start, subtitle, *, monitor_view=False):
    overlays = go.Figure()
    for column, label in (("Price", "Synthetic Index"), ("SMA50", "SMA50"), ("SMA200", "SMA200"),
                          ("Upper", "Bollinger Upper"), ("Lower", "Bollinger Lower")):
        overlays.add_scatter(x=display.index, y=display[column]/base, name=label, mode="markers" if len(display) == 1 else "lines")
    label_chart(overlays, "Portfolio Index — Trend & Overlays", subtitle, "Index")
    st.plotly_chart(overlays)
    z_chart = go.Figure(go.Scatter(x=display.index, y=display["Z-score"], name="Rolling Z-score"))
    for level, color in ((0, "#ffd54f"), (2, "#a0aab8"), (-2, "#ff5252")):
        z_chart.add_hline(y=level, line_color=color, line_dash="dash")
    label_chart(z_chart, "Portfolio Rolling Z-Score", "20 trading sessions", "Z-score")
    st.plotly_chart(z_chart)
    ewma = np.sqrt(returns.pow(2).ewm(alpha=.06, adjust=False).mean() * 252)
    risk = go.Figure()
    for label, values in (("Realized · 21 sessions", realized), ("EWMA · λ 0.94", ewma)):
        shown = values.loc[str(start):]
        risk.add_scatter(x=shown.index, y=shown, name=label)
    if initial is not None and np.isfinite(initial):
        risk.add_hline(y=initial, annotation_text="Imported ex-ante volatility", line_dash="dash")
    label_chart(risk, "Risk Indicators", "Rolling and conditional volatility", "Annualized Volatility (%)")
    risk.update_yaxes(tickformat=".1%")
    fit = cached_forecast(returns, "normal")
    if monitor_view and fit:
        filtered = fit["filtered"].loc[str(start):] * np.sqrt(252)
        risk.add_scatter(x=filtered.index, y=filtered, name="GARCH filtered")
    st.plotly_chart(risk)
    if fit is not None:
        st.metric("GARCH next-session volatility · annualized", f"{fit['sigma']*np.sqrt(252):.2%}")
    else:
        st.caption("GARCH unavailable. It requires at least 126 returns and a valid fit.")

def render_entry_assessment(data, realized, initial, *, exit_view=False):
    row = data.iloc[-1]
    trend_inputs = row[["SMA50", "SMA200", "Momentum 63"]]
    bullish = row.Price > row.SMA50 > row.SMA200
    bearish = row.Price < row.SMA50 < row.SMA200
    momentum = row["Momentum 63"]
    state = "Mixed Signals"
    if trend_inputs.isna().any():
        state = "Insufficient Data"
    elif bullish and momentum > 0:
        state = "Favorable Trend Conditions"
    elif bearish and momentum < 0:
        state = "Unfavorable Trend Conditions"
    tone = {"Favorable Trend Conditions": "green", "Unfavorable Trend Conditions": "red"}.get(state, "gray")
    heading = "Exit Assessment" if exit_view else "Entry Assessment"
    if exit_view:
        state = "Insufficient Data" if trend_inputs.isna().any() else "Hold"
        if state != "Insufficient Data" and bearish and momentum < 0:
            state = "Evaluate Exit" if row["Momentum 126"] < 0 else "Evaluate Reduction"
        tone = {"Hold": "green", "Evaluate Exit": "red", "Evaluate Reduction": "orange"}.get(state, "gray")
    st.markdown(f"### {heading} · :{tone}-badge[{state}] · {data.index[-1]:%d %b %Y}")
    trend = "Bullish" if bullish else "Bearish" if bearish else "Mixed"
    latest_vol = realized.iloc[-1] if not realized.empty else np.nan
    gap = (latest_vol - initial) * 100 if initial is not None else np.nan
    z = row["Z-score"]
    cards = [("Trend", trend, "Ordering of the index, SMA50 and SMA200.")]
    if exit_view:
        if trend_inputs.isna().any():
            cards[0] = ("Trend", "Insufficient Data", cards[0][2])
        cards.insert(1, ("Exit Signal", state, "Uses trend and medium-term momentum."))
    for months, sessions in ((1, 21), (3, 63), (6, 126)):
        value = row[f"Momentum {sessions}"]
        cards.append((f"Momentum {months}M", f"{value:+.2%}" if pd.notna(value) else "—",
                      f"Index return over {sessions} trading sessions."))
    cards.extend([
        ("Z-score (20d)", f"{z:+.2f}" if pd.notna(z) else "—",
         "Distance from the 20-session mean in standard deviations."),
        ("Volatility Gap", f"{gap:+.2f} pp" if np.isfinite(gap) else "—",
         "Annualized 21-session realized volatility minus imported ex-ante volatility."),
    ])
    for offset in range(0, len(cards), 3):
        for col, (label, value, help_text) in zip(st.columns(3), cards[offset:offset+3]):
            col.metric(label, value, help=help_text, border=True)

    def position(series):
        latest = series.iloc[-1]
        if pd.isna(latest):
            return "—"
        previous = series.iloc[-2] if len(series) >= 2 else np.nan
        if previous <= 0 < latest:
            return "↑ Crossed above"
        if previous >= 0 > latest:
            return "↓ Crossed below"
        return "↑ Above" if latest > 0 else "↓ Below" if latest < 0 else "• At level"

    for col, (label, detail) in zip(st.columns(3), (
        ("Moving averages", f"SMA50 {position(data.Price-data.SMA50)} · SMA200 {position(data.Price-data.SMA200)}"
         f" · SMA50 vs SMA200 {position(data.SMA50-data.SMA200)}"),
        ("Bollinger Upper", position(data.Price-data.Upper)),
        ("Bollinger Lower", position(data.Price-data.Lower)),
    )):
        col.caption(f"**{label}** · {detail}")

    with st.expander(f"How to interpret {heading}"):
        if exit_view:
            st.write("Evaluate Reduction requires index < SMA50 < SMA200 and negative 3-month momentum. "
                     "Negative 6-month momentum as well gives Evaluate Exit. Otherwise the state is Hold "
                     "when the required trend and momentum inputs are available.")
        st.write("Favorable requires index > SMA50 > SMA200 and positive 63-session momentum. "
                 "Unfavorable requires the reverse ordering and negative momentum. Other combinations are mixed. "
                 "Missing moving averages or 63-session momentum produce Insufficient Data.")
        st.write("An absolute Z-score of at least 2 marks an extreme deviation. Band crossings describe price behavior. "
                 "These indicators describe market conditions and have no validated predictive interpretation for future returns.")


def activation(store, mode):
    draft = st.session_state.get("trading_draft")
    if mode == "Historical Activation":
        historical_activation(store)
        return
    if mode == "Paper" and not draft:
        st.info("Activate a snapshot from Portfolio Optimization or Factor Models.")
        return
    name = st.text_input("Portfolio name", value=draft["model"] if draft and mode == "Paper" else "Manual portfolio",
                         key=f"ts_name_{draft['id'] if draft and mode == 'Paper' else 'manual'}")
    currency = "EUR"
    if mode == "Paper":
        st.dataframe(pd.Series(draft["weights"], name="Target weight").to_frame())
        st.caption(f"{draft['model']} · Optimized {draft.get('optimized_at') or 'date unavailable'}")
        if draft.get("configuration"):
            with st.expander("Imported configuration"):
                st.json(draft["configuration"])
        with st.container(key="kangemi_controls_trading_paper"):
            mode_col, date_col = st.columns([2, 1], vertical_alignment="bottom")
            purpose = mode_col.radio("Paper mode", [PAPER_BUY, PAPER_ANALYSIS],
                                     horizontal=True, key="ts_paper_purpose")
            if purpose == PAPER_ANALYSIS:
                start = date_col.date_input("Analysis Start Date", value=date.today()-timedelta(days=90),
                                            max_value=date.today())
            else:
                day = date_col.date_input("Activation Date", value=date.today(), max_value=date.today())
            capital_col, fee_col, shares_col = st.columns([2, 2, 1], vertical_alignment="bottom")
            capital = capital_col.number_input("Virtual capital (EUR)", min_value=1., value=10000.)
            fee = fee_col.number_input("Commission per order (EUR)", min_value=0., value=0.)
            whole = shares_col.checkbox("Whole shares", value=True)
        if purpose == PAPER_ANALYSIS:
            st.button("Proceed to Paper Activation", on_click=st.session_state.update,
                      args=({"ts_paper_purpose": PAPER_BUY},))
            entry_analysis(draft, start, currency)
            return
        st.caption("Simulated at the latest common completed close on or before the requested date. Historical replay uses saved target weights and is labelled separately; it is not a historical forecast.")
        if st.button("Preview paper orders"):
            history = quotes(tuple(draft["weights"]), day-timedelta(days=10), currency)
            closes = pd.DataFrame({t:h.Close for t,h in history.items()}).loc[:str(day)].dropna()
            if closes.empty:
                raise ValueError("No common completed close within ten days.")
            execution = closes.index[-1]
            prices = {t: closes.iloc[-1][t] * h.loc[h.index > execution, "Stock Splits"].replace(0,1).prod()
                      for t,h in history.items()}
            orders, cash = paper_orders(draft["weights"], prices, capital, fee, whole)
            for order in orders:
                order["date"] = str(execution.date())
            st.session_state["ts_preview"] = ((draft["id"], name, capital, fee, whole, day), orders, cash)
        preview = st.session_state.get("ts_preview")
        if not preview or preview[0] != (draft["id"], name, capital, fee, whole, day):
            return
        _, orders, cash = preview
        identifier, snapshot = draft["id"], draft
        st.dataframe(orders)
        st.caption(f"Residual cash: €{cash:,.2f}")
        if draft.get("optimized_at") and orders[0]["date"] < draft["optimized_at"][:10]:
            st.warning("Historical Replay Using Today's Weights. The allocation was formed after the simulated execution date.")
    else:
        currency = st.selectbox("Portfolio currency", ["EUR", "USD", "GBP", "CHF"])
        cash = st.number_input("Available cash", min_value=0., value=0.)
        rows = st.data_editor(pd.DataFrame({"ticker":pd.Series(dtype=str), "quantity":pd.Series(dtype=float),
                                           "price":pd.Series(dtype=float), "date":pd.Series(dtype="datetime64[ns]"),
                                           "fee":pd.Series(dtype=float)}), num_rows="dynamic")
        if rows.empty:
            return
        if rows.isna().any().any():
            st.info("Complete all fields, including commission (use zero if none).")
            return
        orders = [{"ticker":str(r.ticker).strip().upper(), "quantity":float(r.quantity), "price":float(r.price),
                   "fee":float(r.fee), "date":str(pd.Timestamp(r.date).date())} for r in rows.itertuples()]
        capital = cash + sum(o["quantity"]*o["price"]+o["fee"] for o in orders)
        identifier = st.session_state.setdefault("ts_manual_id", uuid4().hex)
        snapshot = {}
    if st.button("Confirm activation", type="primary", disabled=store is None):
        for o in orders:
            if (not np.isfinite([o["quantity"],o["price"],o["fee"]]).all() or
                    min(o["quantity"],o["price"]) <= 0 or o["fee"] < 0 or o["date"] > str(date.today())):
                raise ValueError("Invalid purchase details.")
        start = min(o["date"] for o in orders)
        history = quotes(tuple(sorted({o["ticker"] for o in orders})), pd.Timestamp(start).date()-timedelta(days=10), currency)
        for o in orders:
            if pd.Timestamp(o["date"]) not in history[o["ticker"]].index:
                raise ValueError(f"No completed quote on purchase date for {o['ticker']}.")
        meta = dict(name=name.strip() or mode, mode=mode, currency=currency, cash=cash,
                    capital=capital, start=start, model=snapshot.get("model","Manual"),
                    created=pd.Timestamp.now(tz="UTC").isoformat(),
                    replay=bool(snapshot.get("optimized_at") and start <= snapshot["optimized_at"][:10]))
        store.put(identifier, [("portfolio","initial",meta),("snapshot","initial",snapshot),
                               *[("transaction",str(i),o) for i,o in enumerate(orders)]])
        if mode == "Manual":
            st.session_state.pop("ts_manual_id", None)
        else:
            st.session_state.pop("trading_draft", None)
            st.session_state.pop("ts_preview", None)
        st.success("Portfolio saved.")


@st.cache_data(ttl=900, show_spinner=False)
def cached_forecast(returns, distribution):
    return forecast(returns, distribution)


def monitor_indicator_returns(full_close, quantities, returns, cash):
    """Extend the realized return series with a conditional opening-basket history."""
    opening = quantities.iloc[0]
    opening = opening[opening > 0]
    prices = full_close.loc[full_close.index <= returns.index[0], opening.index]
    proxy = prices.mul(opening).sum(axis=1, min_count=len(opening)) + cash
    warmup = proxy.pct_change(fill_method=None).iloc[1:-1]
    # Use the latest uninterrupted history before the activation.
    if warmup.isna().any():
        warmup = warmup.iloc[np.flatnonzero(warmup.isna())[-1]+1:]
    return pd.concat([warmup, returns])


def dashboard(store, identifier, meta, metric):
    if st.button("Refresh Data"):
        quotes.clear()
    orders = [v for _,_,v in store.read("transaction", identifier)]
    snapshot = store.read("snapshot", identifier)[0][2]
    start = pd.Timestamp(meta["start"])
    history = quotes(tuple(sorted({o["ticker"] for o in orders})), start.date()-timedelta(days=400), meta["currency"])
    def frame(column):
        return pd.DataFrame({t:h[column] for t,h in history.items()}).sort_index()
    close = frame("Close").loc[start:]
    nav, returns, dd, marks, qty, flows = monitor(orders, close, frame("Dividends").reindex(close.index),
                                                 frame("Stock Splits").reindex(close.index), meta["cash"])
    if nav.empty or pd.isna(nav.iloc[-1]):
        raise ValueError("Latest session is missing a quote for an active position; valuation paused.")
    last = nav.index[-1]
    stamp = pd.Timestamp.now(tz="UTC")
    store.put(identifier,[("daily_value",str(d.date()),{"nav":float(v)}) for d,v in nav.dropna().items()])
    st.caption(f"{meta['mode']} · {meta['model']} · Activated {meta['start']} · Latest close {last.date()} · Updated {stamp:%Y-%m-%d %H:%M} UTC")
    if meta.get("replay"):
        st.caption("Historical replay: these target weights were not necessarily available at the simulated entry.")
    if meta.get("historical"):
        st.caption(f"Decision {snapshot['decision_timestamp']} · Execution {snapshot['execution_timestamp']} · Monitoring {stamp.isoformat()}")
        import exchange_calendars as xcals

        calendar = xcals.get_calendar(snapshot["calendar"])
        with st.expander("Historical walk-forward risk"):
            if st.button("Reconstruct daily risk forecasts"):
                from src.historical_activation import historical_forecasts

                prior = {key for _, key, _ in store.read("historical_forecast", identifier)}
                for key, value in historical_forecasts(returns, calendar, last, prior):
                    store.put(identifier, [("historical_forecast", key, value)])
            records = [v for _, _, v in store.read("historical_forecast", identifier) if "tails" in v]
            if not records:
                st.caption("Walk-forward forecasts require at least 126 daily observations and a valid fit.")
            if records:
                confidence = st.selectbox("Historical confidence", [.95, .975, .99])
                chart = go.Figure()
                for model in ("normal", "t"):
                    values = sorted((v for v in records if v["model"] == model), key=lambda v: v["valid"])
                    for i, label in enumerate(("VaR", "ES")):
                        chart.add_scatter(x=[v["valid"] for v in values],
                                          y=[v["tails"][str(confidence)][i] for v in values],
                                          name=f"{model} {label}")
                chart.update_yaxes(tickformat=".1%")
                st.plotly_chart(chart)
                st.caption("Reconstructed forecasts, separate from forecasts archived in real time.")
    ccy = meta["currency"]
    realized = returns.rolling(21).std()*np.sqrt(252)
    initial = snapshot.get("volatility")
    live_fit = cached_forecast(returns, "normal")
    for col, (label, value, color) in zip(st.columns(3), [
        ("Initial ex-ante volatility", initial, "blue"),
        ("Realized volatility · 21 sessions", realized.iloc[-1], "red"),
        ("GARCH next session · annualized", live_fit["sigma"]*np.sqrt(252) if live_fit else None, "purple"),
    ]):
        with col:
            metric(label, f"{value:.2%}" if value is not None and np.isfinite(value) else "—", color)
    for col,(label,value,color) in zip(st.columns(4),[
        ("Initial capital",f"{ccy} {meta['capital']:,.2f}","blue"),
        ("Current value",f"{ccy} {nav.iloc[-1]:,.2f}","blue"),
        ("P&L",f"{ccy} {nav.iloc[-1]-meta['capital']:,.2f}","green"),
        ("Current / max drawdown",f"{dd.iloc[-1]:.2%} / {dd.min():.2%}","red")]):
        with col:
            metric(label,value,color)
    st.caption(f"Cash incl. credited dividends: {ccy} {nav.iloc[-1]-marks.iloc[-1].sum():,.2f} · "
               f"P&L / contributed capital: {(nav.iloc[-1]/meta['capital']-1):.2%}")
    indicator_returns = monitor_indicator_returns(frame("Close"), qty, returns, meta["cash"])
    indicator_index = (1 + indicator_returns).cumprod()
    assessment = indicators(indicator_index)
    displayed = assessment.loc[start:]
    st.caption("Pre-activation warm-up uses opening quantities and cash with historical closing prices. "
               "It is a conditional reconstruction and excludes pre-activation dividends. "
               "The index uses realized portfolio returns from activation onward.")
    render_entry_assessment(assessment, realized, initial, exit_view=True)
    indicator_vol = indicator_returns.rolling(21).std() * np.sqrt(252)
    render_portfolio_charts(displayed, displayed.Price.iloc[0], indicator_vol, indicator_returns,
                            initial, start, "Opening-basket warm-up · displayed base 1", monitor_view=True)
    equity = (1+returns).cumprod()
    target = pd.Series(snapshot.get("weights", {}),dtype=float)
    positions = pd.DataFrame({"Quantity":qty.iloc[-1],"Price":close.iloc[-1],"Market value":marks.iloc[-1],
                              "Target weight":target,"Current weight":marks.iloc[-1]/nav.iloc[-1]})
    basis = pd.Series({t:sum(o["quantity"]*o["price"]+o["fee"] for o in orders if o["ticker"]==t) for t in marks})
    positions["Acquisition cost"] = basis
    positions["Position P&L excl. dividends"] = positions["Market value"]-basis
    positions["Initial price (split-normalized)"] = basis / qty.iloc[-1]
    weights = positions["Current weight"].dropna()
    cov = close.pct_change(fill_method=None).dropna().tail(63).cov().reindex(index=weights.index,columns=weights.index)
    variance = weights @ cov @ weights
    positions["Risk contribution"] = weights * (cov @ weights) / variance if variance > 0 else np.nan
    st.dataframe(positions)
    value_chart = go.Figure()
    for label, values in (("Portfolio NAV", nav), ("Contributed Capital", flows.cumsum())):
        value_chart.add_scatter(x=values.index, y=values, name=label,
                                hovertemplate=dated_hover(label, f"%{{y:,.2f}} {ccy}"))
    label_chart(value_chart, "Portfolio Value & Invested Capital",
                "Evolution of portfolio NAV versus contributed capital.",
                f"Portfolio Value ({ccy})")
    st.plotly_chart(value_chart)
    for dist in ("normal","t"):
        result = cached_forecast(returns,dist)
        if not result or pd.isna(returns.iloc[-1]) or ccy != "EUR":
            continue
        # A forecast first produced after the next exchange session started is
        # displayed live but must not be retroactively scored.
        # Quotes exclude today's bar. Archive only if the latest bar is
        # recent enough that an unobserved session cannot already have closed.
        next_day = (last + pd.offsets.BDay()).date()
        if meta.get("historical"):
            next_day = calendar.next_session(last).date()
            if stamp >= calendar.session_open(pd.Timestamp(next_day)):
                continue
        if stamp.tz_convert("Europe/Rome").date() > next_day:
            continue
        if stamp.tz_convert("Europe/Rome").date() == next_day and stamp.tz_convert("Europe/Rome").hour >= 8:
            continue
        records = []
        for conf in (.95,.975,.99):
            var, es = tail(result["mean"],result["sigma"],conf,result["params"].get("nu"))
            records.append(("forecast",f"{last.date()}:{dist}:{conf}",dict(
                origin=str(last.date()), valid=str(next_day), produced=stamp.isoformat(),
                model=dist, confidence=conf,var=var,es=es, params=result["params"], valid_period="next observed session")))
        store.put(identifier,records)
    section = st.radio("Analysis", ["Volatility & Tail Risk","Trend & Mean Reversion","Signals"],horizontal=True)
    if section == "Volatility & Tail Risk":
        risk_view(store, identifier, nav, returns, initial, ccy, show_volatility=False)
    else:
        z = st.number_input("Z-score threshold",min_value=.1,value=2.)
        window = st.selectbox("Mean reversion window",[20,50,100])
        series = {"Portfolio":equity, **{t:h["Adj Close"] for t,h in history.items()}}
        analysis = {t:indicators(p,window) for t,p in series.items()}
        signals = {t:signal(data,z) for t,data in analysis.items()}
        st.caption(f"Portfolio state: {signals['Portfolio']['state']}")
        store.put(identifier,[("signal",f"{last.date()}:{window}:{z}:{t}",v) for t,v in signals.items()])
        if section == "Signals":
            st.dataframe(pd.DataFrame(signals).T)
            if meta.get("historical"):
                states = [signal(analysis["Portfolio"].iloc[max(0, i-199):i+1], z)["state"]
                          for i in range(len(equity))]
                st.plotly_chart(go.Figure(go.Scatter(x=equity.index, y=states, mode="markers",
                                                   name="Reconstructed portfolio state")))
        else:
            asset = st.selectbox("Series", list(history))
            d = analysis[asset]
            lines = st.multiselect("Overlays",["SMA20","SMA50","SMA100","SMA200","EMA20","Upper","Lower"],["SMA50","SMA200","Upper","Lower"])
            series_label = "Portfolio index" if asset == "Portfolio" else f"{asset} adjusted price"
            trend = go.Figure([
                go.Scatter(x=d.index, y=d[column], name=series_label if column == "Price" else column)
                for column in ["Price", *lines]
            ])
            trend.update_layout(title=f"{series_label} · Trend & Overlays")
            st.plotly_chart(trend)
            fig = go.Figure(go.Scatter(x=d.index, y=d["Z-score"], name="Z-score"))
            fig.update_layout(title=f"{asset} · Rolling Z-score ({window} sessions)")
            fig.add_hline(y=0, line_color="#ffd54f", line_dash="dash", line_width=1.5)
            fig.add_hline(y=-z, line_color="#ff5252", line_dash="dash", line_width=1.5)
            fig.add_hline(y=z)
            st.plotly_chart(fig)
            st.dataframe(d.tail(1))
            st.caption("Bands and Z-score measure distance from the rolling mean; a large deviation alone does not establish mean reversion.")


def dated_hover(label, value, detail=""):
    return f"%{{x|%d %b %Y}}<br>{label}: {value}{detail}<extra></extra>"


def label_chart(fig, title, subtitle, y_title):
    fig.update_layout(title=dict(text=title, subtitle=dict(text=subtitle), x=0.02,
                                 font=dict(family="Menlo, Consolas, monospace", color="#ffbf69")),
                      margin=dict(t=105), xaxis_title="Date", yaxis_title=y_title)


def risk_view(store, identifier, nav, returns, initial, ccy, *, show_volatility=True):
    window = st.selectbox("Realized volatility window",[21,63,126,252])
    decay = st.slider("EWMA lambda",.80,.99,.94)
    realized = returns.rolling(window).std()*np.sqrt(252)
    ewma = np.sqrt(returns.pow(2).ewm(alpha=1-decay,adjust=False).mean()*252)
    fig = go.Figure()
    for name, values, detail in (
        ("Realized Volatility", realized, f"<br>Rolling historical window: {window} sessions"),
        ("EWMA Filtered Volatility", ewma, f"<br>Recursive estimate · λ = {decay:.2f}"),
    ):
        fig.add_scatter(x=values.index, y=values, name=name,
                        hovertemplate=dated_hover(name, "%{y:.2%} annualized", detail))
    if initial is not None:
        name = "Initial Ex-Ante Volatility"
        fig.add_scatter(x=returns.index, y=np.full(len(returns), initial), name=name,
                        mode="lines", line=dict(dash="dash", color="#ffbf69"),
                        hovertemplate=dated_hover(name, "%{y:.2%} annualized",
                                                  "<br>Risk estimate from optimization"))
        if initial > 0 and pd.notna(realized.iloc[-1]):
            st.caption(f"Realized / initial volatility: {realized.iloc[-1]/initial:.2f}×. Divergence alone does not invalidate the initial estimate.")
    confidence = st.selectbox("One-day confidence",[.95,.975,.99],format_func=lambda x:f"{x:.1%}")
    distribution = go.Figure()
    fits = {d:cached_forecast(returns,d) for d in ("normal","t")}
    valid_fits = [r for r in fits.values() if r]
    extent = max((abs(r["mean"])+8*r["sigma"] for r in valid_fits),default=.1)
    x = np.linspace(-extent,extent,1601)
    rows = []
    for dist,r in fits.items():
        if not r:
            st.caption(f"GARCH {dist}: insufficient observations or no valid converged fit. EWMA remains available.")
            continue
        filtered = r["filtered"]*np.sqrt(252)
        fig.add_scatter(x=filtered.index,y=filtered,name=f"GARCH {dist} filtered")
        fig.add_scatter(x=[nav.index[-1]],y=[r["sigma"]*np.sqrt(252)],mode="markers",name=f"GARCH {dist} next-session forecast")
        nu = r["params"].get("nu")
        var, es = tail(r["mean"],r["sigma"],confidence,nu)
        law = stats.norm(loc=r["mean"],scale=r["sigma"]) if nu is None else stats.t(nu,loc=r["mean"],scale=r["sigma"]*np.sqrt((nu-2)/nu))
        distribution.add_scatter(x=x,y=law.pdf(x),name=dist)
        left = np.r_[x[x < -var],-var]
        distribution.add_scatter(x=left,y=law.pdf(left),name=f"{dist} loss tail",fill="tozeroy")
        for label,v in [("VaR",-var),("ES",-es),("mean",r["mean"])]:
            distribution.add_vline(x=v,annotation_text=f"{dist} {label}")
        rows.append({"Model":dist,"Annualized forecast":r["sigma"]*np.sqrt(252),
                     "VaR %":100*var,"ES %":100*es,f"VaR {ccy}":nav.iloc[-1]*var,f"ES {ccy}":nav.iloc[-1]*es})
    fig.update_traces(hovertemplate=dated_hover("%{fullData.name}", "%{y:.2%} annualized"),
                      selector=lambda trace: trace.name.startswith("GARCH"))
    label_chart(fig, "Portfolio Volatility — Realized vs EWMA",
                "Annualized realized volatility, EWMA volatility and initial ex-ante risk estimate.",
                "Annualized Volatility (%)")
    fig.update_yaxes(tickformat=".0%")
    if show_volatility:
        st.plotly_chart(fig)
    st.caption(f"Realized volatility requires {window} daily returns and becomes available only after the initial observation window.")
    if rows:
        st.dataframe(rows)
        distribution.update_xaxes(tickformat=".1%",title="One-day return")
        st.plotly_chart(distribution)
    st.caption("GARCH(1,1) forecasts next-session conditional variance; VaR and ES use loss-positive convention. ES is the expected loss conditional on crossing the model quantile.")
    archived = [v for _,_,v in store.read("forecast",identifier) if v["confidence"]==confidence]
    hits_fig = go.Figure(go.Scatter(x=returns.index,y=returns,name="Observed return"))
    for dist in fits:
        records = sorted((v for v in archived if v["model"]==dist),key=lambda v:v["valid"])
        matched = []
        for v in records:
            subsequent = returns.index[returns.index > pd.Timestamp(v["origin"])]
            if len(subsequent) and pd.notna(returns.loc[subsequent[0]]):
                matched.append((subsequent[0], v))
        h = [returns[d] < -v["var"] for d,v in matched]
        hits_fig.add_scatter(x=[d for d,_ in matched],y=[-v["var"] for _,v in matched],name=f"{dist} VaR",mode="markers")
        hits_fig.add_scatter(x=[d for (d,_),hit in zip(matched,h) if hit],y=[returns[d] for (d,_),hit in zip(matched,h) if hit],name=f"{dist} breach",mode="markers")
        dates = pd.DatetimeIndex([d for d,_ in matched])
        contiguous = len(dates)>0 and dates.equals(returns.dropna().loc[dates[0]:dates[-1]].index)
        p = coverage_tests(h,confidence) if contiguous else None
        es_gap = [(-returns[d])-v["es"] for (d,v),hit in zip(matched,h) if hit]
        st.caption(f"{dist}: {sum(h)}/{len(h)} violations" + (f" ({np.mean(h):.1%})" if h else "") +
                   (f" · Kupiec p={p[0]:.3f} · Christoffersen p={p[1]:.3f}" if p else " · Insufficient observations or gaps for joint diagnostics") +
                   (f" · Mean realized tail loss minus ES: {np.mean(es_gap):.2%} (descriptive)" if len(es_gap)>=5 else " · Insufficient observations for ES diagnostics"))
    label_chart(hits_fig, "Daily Portfolio Returns",
                "Daily percentage returns of the activated portfolio.", "Daily Return (%)")
    hits_fig.update_traces(hovertemplate=dated_hover("%{fullData.name}", "%{y:.2%}"))
    hits_fig.update_yaxes(tickformat=".1%")
    hits_fig.add_hline(y=0, line_color="#ffd54f", line_dash="dash", line_width=1.5)
    st.plotly_chart(hits_fig)


TRADING_SIGNALS_STYLE = """
<style>
.st-key-trading_signals_terminal {
    --terminal-amber: #ffbf69;
    --terminal-panel: #151b25;
    --terminal-border: #303d50;
    --terminal-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.st-key-trading_signals_terminal :is(h2, h3, h4) {
    font-family: var(--terminal-mono);
}
.st-key-trading_signals_terminal [data-testid="stWidgetLabel"] p {
    color: var(--terminal-amber);
    font-family: var(--terminal-mono);
    font-weight: 600;
}
.st-key-trading_signals_terminal [data-testid="stRadio"] [role="radiogroup"] {
    gap: .4rem 1rem;
    flex-wrap: wrap;
}
.st-key-trading_signals_terminal [data-testid="stMetric"] {
    background: var(--terminal-panel);
    border-radius: .45rem;
}
.st-key-trading_signals_terminal [data-testid="stMetricValue"] {
    font-family: var(--terminal-mono);
    font-variant-numeric: tabular-nums;
}
.st-key-trading_signals_terminal button[data-testid="stBaseButton-primary"]:enabled {
    background: #cd9b60;
    border-color: var(--terminal-amber);
    color: #14181f;
}
.st-key-trading_signals_terminal button[data-testid="stBaseButton-primary"]:enabled:hover {
    background: var(--terminal-amber);
}
.st-key-trading_signals_terminal [data-testid="stExpander"] details {
    border-color: var(--terminal-border);
}
.st-key-trading_signals_terminal [data-testid="stExpander"] summary {
    background: var(--terminal-panel);
    font-family: var(--terminal-mono);
}
</style>
"""


def render(metric):
    with st.container(key="trading_signals_terminal"):
        st.markdown(TRADING_SIGNALS_STYLE, unsafe_allow_html=True)
        render_workspace(metric)


def render_workspace(metric):
    st.header("Trading Signals")
    st.caption("Prospective monitoring of paper and externally executed portfolios.")
    try:
        store = open_store()
        if store is None:
            st.info("Preview is available. Saving portfolios requires PostgreSQL with OIDC, or a local SQLite journal configured with TRADING_SIGNALS_LOCAL_SQLITE.")
        views = ["Activate", "Monitor"] if st.session_state.get("trading_draft") else ["Monitor", "Activate"]
        if st.session_state.pop("ts_open_monitor", False):
            st.session_state["ts_workspace"] = "Monitor"
        with st.container(key="kangemi_controls_trading_workspace"):
            workspace_col, type_col = st.columns([1, 2])
            view = workspace_col.radio("Workspace", views if store else ["Activate"],
                                       horizontal=True, key="ts_workspace")
            if view == "Activate":
                mode = type_col.radio("Portfolio type", ["Paper", "Historical Activation", "Manual"],
                                      horizontal=True, key="ts_portfolio_type")
        if view == "Activate":
            activation(store, mode)
            return
        portfolios = {p:v for p,_,v in store.read("portfolio")}
        archived = {p for p,_,_ in store.read("archived")}
        names = {}
        for p,k,v in sorted(store.read("rename"),key=lambda r:r[1]):
            names[p] = v
        options = sorted(portfolios.keys()-archived)
        if not options:
            st.info("No active portfolios. Open Activate to start.")
            return
        identifier = st.selectbox("Portfolio",options,index=options.index(st.session_state["historical_active"]) if st.session_state.get("historical_active") in options else 0,format_func=lambda p:f"{names.get(p,portfolios[p]['name'])} · {p[:6]}")
        with st.expander("Manage portfolio"):
            name = st.text_input("New name",key=f"ts_rename_{identifier}")
            if st.button("Rename") and name.strip():
                store.put(identifier,[("rename",pd.Timestamp.now(tz="UTC").isoformat(),name.strip())])
                st.rerun()
            if st.checkbox("Confirm archiving this portfolio") and st.button("Archive"):
                store.put(identifier,[("archived","once",True)])
                st.rerun()
        dashboard(store,identifier,portfolios[identifier],metric)
    except Exception as exc:
        # Database and provider errors may contain connection details.
        if type(exc) is ValueError:
            st.error(str(exc))
        else:
            st.error(f"Trading Signals could not finish ({type(exc).__name__}). Check configuration and data provider availability.")
