import copy
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.trading_market import (
    Quote,
    confirmed_history,
    current_equity,
    fetch_intraday,
    select_quote,
    snapshot_values,
    temporary_series,
)
from src.trading_signals import Store, indicators, monitor, signal


ZONE = "America/New_York"
META = {"currency": "USD", "exchangeName": "NMS", "exchangeTimezoneName": ZONE}
OPEN = pd.Timestamp("2026-11-23 16:00Z")
CLOSE = pd.Timestamp("2026-11-23 21:25Z")


def history():
    days = pd.DatetimeIndex(["2026-11-19", "2026-11-20", "2026-11-23"], tz=ZONE)
    return pd.DataFrame({"Close": [99., 100., 109.], "Adj Close": [98., 99., 109.],
                         "Stock Splits": 0., "Dividends": 0.}, index=days)


def minutes(prices, times):
    return pd.DataFrame({"Close": prices}, index=pd.to_datetime(times, utc=True, format="mixed"))


class TradingIntradayTests(unittest.TestCase):
    def test_provider_and_validation(self):
        bars = minutes([106., 107., 0., -1., 999.], [
            "2026-11-23 15:58Z", "2026-11-23 15:58Z", "2026-11-23 15:59Z",
            "2026-11-23 15:59:30Z", "2026-11-23 16:01Z",
        ])
        ticker = Mock(history_metadata={**META, "regularMarketTime": int(OPEN.timestamp()),
                                        "regularMarketPrice": 108., "exchangeDataDelayedBy": 15})
        ticker.history.return_value = bars
        with patch("src.trading_market.yf.Ticker", return_value=ticker):
            response = fetch_intraday("TEST")
        ticker.history.assert_called_once_with(period="1d", interval="1m", prepost=False,
                                               auto_adjust=False, actions=True, raise_errors=True)
        daily = confirmed_history(history(), META, OPEN)
        quote = select_quote("TEST", daily, response, OPEN)
        self.assertEqual((quote.status, quote.price, quote.timestamp), ("INTRADAY", 108., OPEN))
        self.assertIn("15 min", quote.detail)
        bar_quote = select_quote("TEST", daily, (bars, META, ""), OPEN)
        self.assertEqual(bar_quote.price, 107.)
        self.assertAlmostEqual(bar_quote.adjusted_price, 107. * .99)
        self.assertEqual(bar_quote.previous_close, 100.)

    def test_weekend_holiday_close_and_missing(self):
        empty = pd.DataFrame(), {}, "Feed down"
        frame = history()
        daily = confirmed_history(frame, META, OPEN)
        weekend = pd.Timestamp("2026-11-21 16:00Z")
        quote = select_quote("TEST", daily, empty, weekend)
        self.assertEqual((quote.status, quote.market, quote.price), ("OFFICIAL_CLOSE", "CLOSED", 100.))
        holiday = pd.Timestamp("2026-11-26 16:00Z")
        closed = confirmed_history(frame, META, CLOSE)
        self.assertEqual(select_quote("TEST", closed, empty, CLOSE).price, 109.)
        holiday_quote = select_quote("TEST", closed, empty, holiday)
        self.assertEqual((holiday_quote.market, holiday_quote.status), ("CLOSED", "STALE"))
        post_close = pd.Timestamp("2026-11-23 21:05Z")
        self.assertEqual(len(confirmed_history(frame, META, post_close)), 2)
        self.assertEqual(select_quote("TEST", daily, empty, OPEN).status, "STALE")
        old = minutes([1.], ["2026-11-23 14:31Z"])
        self.assertEqual(select_quote("TEST", daily, (old, META, ""), OPEN).price, 100.)
        self.assertEqual(select_quote("TEST", daily.iloc[:0], empty, OPEN).status, "UNAVAILABLE")
        invalid = {**META, "regularMarketTime": "invalid", "regularMarketPrice": 500.}
        self.assertEqual(select_quote("TEST", daily, (pd.DataFrame(), invalid, ""), OPEN).status, "STALE")
        self.assertEqual(len(frame), 3)

    def test_split_does_not_revalue_old_shares_in_new_units(self):
        frame = history()
        frame[["Close", "Adj Close"]] /= 2
        frame.loc[frame.index[-1], "Stock Splits"] = 2.
        daily = confirmed_history(frame, META, OPEN)
        quote = select_quote("TEST", daily, (minutes([55.], [OPEN]), META, ""), OPEN)
        self.assertEqual((quote.status, quote.price), ("STALE", 100.))
        closed = confirmed_history(frame, META, CLOSE)
        order = {"ticker": "TEST", "quantity": 2, "price": 99., "fee": 0., "date": "2026-11-19"}
        nav, _, _, _, qty, _ = monitor([order], *(closed[c].to_frame("TEST")
                                                   for c in ("Close", "Dividends", "Stock Splits")), 0.)
        self.assertEqual(qty.iloc[-1, 0], 4.)
        self.assertEqual(nav.iloc[-1], 218.)

    def test_marks_fx_and_provisional_drawdown(self):
        positions = pd.Series({"A": 2., "B": 3.})
        quotes = {"A": Quote("A", "USD", ZONE, "OPEN", "INTRADAY", 20.),
                  "B": Quote("B", "EUR", "Europe/Rome", "CLOSED", "OFFICIAL_CLOSE", 10.)}
        fx = Quote("USDEUR=X", "EUR", "UTC", "OPEN", "INTRADAY", .9)
        value, marks, rates, status = snapshot_values(positions, 10., quotes, "EUR", {"USD": fx})
        self.assertEqual((value, rates, status), (76., {"A": .9, "B": 1.}, "PROVISIONAL"))
        self.assertEqual(marks.to_dict(), {"A": 36., "B": 30.})
        self.assertEqual(snapshot_values(positions, 10., quotes, "EUR", {})[3], "UNAVAILABLE")
        self.assertAlmostEqual(current_equity(pd.Series([1., 1.1]), 110., 100., 1.25)[1], -.20)

    def test_momentum_z_score_and_signal_preserve_history(self):
        idx = pd.bdate_range(end="2026-11-20", periods=220)
        baseline = pd.Series(np.linspace(100., 80., len(idx)), index=idx)
        original = baseline.copy()
        day = pd.Timestamp("2026-11-23")
        live = temporary_series(baseline, day, 65.)
        result = indicators(live)
        row = result.iloc[-1]
        for offset in (21, 63, 126):
            self.assertAlmostEqual(row[f"Momentum {offset}"], 65. / live.iloc[-offset-1] - 1)
        window = live.iloc[-20:]
        self.assertAlmostEqual(row["Z-score"], (65. - window.mean()) / window.std(ddof=0))
        self.assertEqual(signal(result)["state"], "EXIT CANDIDATE")
        self.assertEqual(len(temporary_series(live, day, 70.)), len(live))
        pd.testing.assert_frame_equal(result.iloc[:-1], indicators(original))
        pd.testing.assert_series_equal(baseline, original)

    def test_streamlit_refresh_updates_cards_but_not_journal_or_daily_cache(self):
        from src import trading_signals_ui as ui

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/journal.db"
            store = Store(path, "explicit-single-user-local")
            idx = pd.bdate_range(end="2026-11-20", periods=270)
            closes = np.linspace(130., 100., len(idx))
            daily = pd.DataFrame({"Close": closes, "Adj Close": closes,
                                  "Dividends": 0., "Stock Splits": 0.}, index=idx)
            daily.attrs["market"] = META
            daily_before = daily.copy(deep=True)
            activation = str(idx[100].date())
            meta = dict(name="Sample", mode="Manual", model="Manual", currency="USD",
                        cash=10., capital=250., start=activation)
            order = dict(ticker="TEST", quantity=2., fee=0., price=120., date=activation)
            store.put("p", [("portfolio", "initial", meta), ("snapshot", "initial", {"volatility": .19}),
                            ("transaction", "zero", order)])
            original = copy.deepcopy(store.read("transaction", "p"))
            t1 = Quote("TEST", "USD", ZONE, "OPEN", "INTRADAY", 105., OPEN,
                       pd.Timestamp("2026-11-23"), 100., "Mock", 105.)
            t2 = replace(t1, price=90., adjusted_price=90.)
            response = Mock(side_effect=[(None, {}, ""), (None, {}, "")])
            with (patch.dict("os.environ", {"TRADING_SIGNALS_LOCAL_SQLITE": path}),
                  patch.object(ui, "quotes", return_value={"TEST": daily}) as eod,
                  patch.object(ui, "intraday_response", response),
                  patch.object(ui, "select_quote", side_effect=[t1, t1, t2]),
                  patch.object(ui, "cached_forecast", return_value=None),
                  redirect_stderr(StringIO())):
                app = AppTest.from_string("""
                    from src.trading_signals_ui import render_workspace
                    import streamlit as st
                    render_workspace(lambda label, value, color: st.metric(label, value))
                """).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                cards = {m.label: m.value for m in app.metric}
                self.assertEqual(cards["Current value"], "USD 220.00")
                self.assertTrue(any("Exit Assessment · PROVISIONAL" in m.value for m in app.markdown))
                baseline = store.read("daily_value", "p")
                next(button for button in app.button if button.label == "Refresh Data").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                cards = {m.label: m.value for m in app.metric}
                self.assertEqual(cards["Current value"], "USD 190.00")
                self.assertEqual(cards["P&L"], "USD -60.00")
                self.assertEqual(cards["Initial ex-ante volatility"], "19.00%")
                eod.clear.assert_not_called()
                response.clear.assert_called_once_with("TEST")
            self.assertEqual(store.read("daily_value", "p"), baseline)
            self.assertEqual(store.read("transaction", "p"), original)
            self.assertEqual(store.read("portfolio", "p")[0][2]["cash"], 10.)
            pd.testing.assert_frame_equal(daily, daily_before)
