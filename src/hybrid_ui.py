"""Manual additions to Asset Selection."""
import re
from datetime import date

import pandas as pd
import streamlit as st

from src.eur_risk_free import EONIA_ESTR, ESTR_ONLY
from src.hybrid_portfolio import GLOBAL, REGIONAL, REGIONS, effective_universe
from src.time_series_market import daily_bars


def active_universe(market, catalog):
    if not st.session_state.get("hybrid_manual"):
        return None
    screening = st.session_state.get("screening_results") or {}
    ranked = st.session_state.get("quant_selected_tickers", ()) if screening.get("market") == market else ()
    names = catalog.set_index("ticker")["name"].to_dict()
    config = effective_universe([(t, names.get(t, t)) for t in ranked if t in names],
                                st.session_state.get("hybrid_manual", {}), market,
                                "EUR",
                                st.session_state.get("hybrid_factors", REGIONAL),
                                st.session_state.get("hybrid_rf_mode", ESTR_ONLY))
    denomination = "USD" if market == "USA" else "EUR"
    for asset in config["assets"]:
        asset.setdefault("quote_currency", denomination)
    st.session_state["hybrid_universe"] = config
    return config


def _preserve_setting(key, default):
    widget = f"_widget_{key}"
    st.session_state.setdefault(widget, st.session_state.get(key, default))

    def save():
        st.session_state[key] = st.session_state[widget]

    return {"key": widget, "on_change": save}


def render_manual_selection(market, catalog):
    st.subheader("Manual Asset Selection")
    with st.expander("Manual additions", expanded=bool(st.session_state.get("hybrid_manual"))):
        cfg = st.columns(2)
        cfg[0].selectbox("Factor methodology", [REGIONAL, GLOBAL], **_preserve_setting("hybrid_factors", REGIONAL))
        cfg[1].selectbox("EUR factor risk-free series", [ESTR_ONLY, EONIA_ESTR],
                         format_func=lambda mode: "€STR" if mode == ESTR_ONLY else "EONIA → €STR",
                         **_preserve_setting("hybrid_rf_mode", ESTR_ONLY))
        st.caption("The quantitative ranking is unchanged. Manual entries are merged after the Top N. "
                   "Confirm economic region explicitly for ADRs and multiple listings.")
        manual = st.session_state.get("hybrid_manual", {}).copy()
        with st.form("hybrid_manual_add"):
            row = st.columns(3)
            ticker = row[0].text_input("Ticker").strip().upper()
            region = row[1].selectbox("Factor region", REGIONS, index=3)
            minimum = row[2].number_input("Minimum weight if mandatory (%)", 0.01, 100., 5.)
            mode = st.radio("Inclusion", ["Optimizer Discretion", "Mandatory Inclusion"], horizontal=True)
            today = pd.Timestamp.now(tz="Europe/Rome").date()
            effective = st.date_input("Historical eligibility from", value=today, min_value=date(1900, 1, 1), max_value=today,
                                      help="Earlier dates are explicit retrospective assumptions. Default is today.")
            add = st.form_submit_button("Add or update ticker")
        if add:
            try:
                if not re.fullmatch(r"[A-Z0-9.^=/-]{1,30}", ticker):
                    raise ValueError("Invalid ticker format.")
                frame, currency = daily_bars(ticker)
                info = frame.attrs.get("quote", {})
                symbol = str(info.get("symbol", ticker)).upper()
                if not info.get("exchange") or not info.get("type") or info["type"] not in {"EQUITY", "ETF", "MUTUALFUND"}:
                    raise ValueError("Yahoo did not verify a supported listed instrument and its exchange.")
                valid = frame.Close.dropna()
                if len(valid) < 3:
                    raise ValueError("Insufficient native historical close coverage.")
                if (pd.Timestamp(today)-valid.index[-1]).days > 7:
                    raise ValueError(f"{ticker}: latest native close is stale.")
                manual[symbol] = dict(
                    name=info.get("name") or symbol, exchange=info["exchange"], region=region,
                    quote_currency=info.get("original_currency") or currency, currency_unit=currency,
                    first_price=valid.index[0].date().isoformat(), last_native_close=valid.index[-1].date().isoformat(),
                    timezone=info.get("timezone"), calendar=info.get("calendar"), effective_from=effective.isoformat(), inclusion=mode,
                    minimum=minimum/100 if mode == "Mandatory Inclusion" else 0.,
                )
                st.session_state["hybrid_manual"] = manual
                st.rerun()
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        remove = st.multiselect("Manual tickers to remove", list(manual))
        if st.button("Remove selected", disabled=not remove):
            st.session_state["hybrid_manual"] = {t: row for t, row in manual.items() if t not in remove}
            st.rerun()
        universe = active_universe(market, catalog)
        records = universe["assets"] if universe else [{"ticker": t, **row} for t, row in manual.items()]
        if records:
            shown = pd.DataFrame(records)
            if "sources" in shown:
                shown["sources"] = shown["sources"].map(" + ".join)
            st.dataframe(shown.style.format({"minimum": "{:.2%}"}, na_rep="—"), hide_index=True)
