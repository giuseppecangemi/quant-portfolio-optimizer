"""Prospective portfolio accounting and quantitative monitoring."""
import json
import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


class Store:
    """Owner-scoped storage. Local SQLite; PostgreSQL for hosted deployments."""

    def __init__(self, location, owner):
        if not owner:
            raise ValueError("A trusted portfolio owner is required.")
        self.location, self.owner = location, owner
        self.pg = location.startswith(("postgresql://", "postgres://"))
        with closing(self.connection()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS trading_records (
                owner TEXT NOT NULL, portfolio TEXT NOT NULL, kind TEXT NOT NULL,
                record TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(owner, portfolio, kind, record))""")

    def connection(self):
        if self.pg:
            import psycopg
            return psycopg.connect(self.location)
        return sqlite3.connect(self.location)

    def put(self, portfolio, records):
        sql = ("INSERT INTO trading_records VALUES (?, ?, ?, ?, ?) "
               "ON CONFLICT(owner,portfolio,kind,record) DO NOTHING")
        with closing(self.connection()) as db, db:
            for kind, key, value in records:
                cursor = db.execute(sql.replace("?", "%s") if self.pg else sql,
                                    (self.owner, portfolio, kind, key, json.dumps(value, allow_nan=False)))
                if kind == "portfolio" and cursor.rowcount == 0:
                    return

    def read(self, kind, portfolio=None):
        sql = "SELECT portfolio,record,payload FROM trading_records WHERE owner=? AND kind=?"
        args = [self.owner, kind]
        if portfolio is not None:
            sql += " AND portfolio=?"
            args.append(portfolio)
        with closing(self.connection()) as db, db:
            rows = db.execute(sql.replace("?", "%s") if self.pg else sql, args).fetchall()
        return [(p, k, json.loads(v)) for p, k, v in rows]


def paper_orders(weights, prices, capital, fee, whole):
    w = pd.Series(weights, dtype=float)
    p = pd.Series(prices, dtype=float).reindex(w.index)
    if (w.empty or not np.isfinite(w).all() or (w < 0).any()
            or not np.isclose(w.sum(), 1) or not np.isfinite(p).all()
            or (p <= 0).any() or not np.isfinite([capital, fee]).all()
            or capital <= 0 or fee < 0):
        raise ValueError("Invalid weights, quotes, capital or commission.")
    q = (capital * w - fee).clip(lower=0) / p
    if whole:
        q = np.floor(q)
    orders = [{"ticker": t, "quantity": float(q[t]), "price": float(p[t]), "fee": fee}
              for t in w.index if q[t] > 0]
    if not orders:
        raise ValueError("Capital is insufficient for any purchase.")
    cash = capital - sum(o["quantity"] * o["price"] + fee for o in orders)
    return orders, max(0., cash)


def monitor(orders, close, dividends, splits, cash):
    """Quotes use split-normalized units; convert recorded units on purchase."""
    index = close.index
    marks = pd.DataFrame(index=index)
    quantities = pd.DataFrame(index=index)
    paid = pd.Series(0., index=index)
    flows = paid.copy()
    for order in orders:
        t, date = order["ticker"], pd.Timestamp(order["date"])
        if t not in close or date > index[-1]:
            raise ValueError(f"No usable quotes for {t}.")
        future_splits = splits.loc[splits.index > date, t].replace(0, 1).dropna()
        q = order["quantity"] * future_splits.prod()
        active = index >= date
        mark = (close[t] * q).where(active, 0.)
        marks[t] = marks[t] + mark if t in marks else mark
        quantities[t] = quantities.get(t, 0.) + pd.Series(np.where(active, q, 0.), index=index)
        paid += dividends[t].fillna(0).where(index > date, 0.) * q
        cost = order["quantity"] * order["price"] + order["fee"]
        flows.loc[index[active][0]] += cost
    nav = marks.sum(axis=1, min_count=len(marks.columns)) + cash + paid.cumsum()
    flows.iloc[0] += cash
    returns = ((nav - flows) / nav.shift() - 1).replace([np.inf, -np.inf], np.nan)
    if nav.iloc[0] == nav.iloc[0] and flows.iloc[0] > 0:
        returns.iloc[0] = nav.iloc[0] / flows.iloc[0] - 1
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax().clip(lower=1) - 1
    return nav, returns, drawdown, marks, quantities, flows


def forecast(returns, distribution):
    from arch import arch_model
    if returns.isna().any():
        return None
    r = returns * 100
    if len(r) < 126 or r.std() <= 0:
        return None
    fit = arch_model(r, p=1, q=1, dist=distribution, rescale=False).fit(disp="off")
    var = float(fit.forecast(horizon=1).variance.iloc[-1, 0])
    if fit.convergence_flag or not np.isfinite(var) or var <= 0:
        return None
    params = {k: float(v) for k, v in fit.params.items()}
    if (not np.isfinite(list(params.values())).all()
            or params["alpha[1]"] + params["beta[1]"] >= 1
            or params.get("nu", 3) <= 2):
        return None
    return {"mean": params["mu"] / 100, "sigma": np.sqrt(var) / 100,
            "params": params, "filtered": fit.conditional_volatility / 100}


def tail(mean, sigma, confidence, nu=None):
    p = 1 - confidence
    if nu is None:
        q = stats.norm.ppf(p)
        es = -stats.norm.pdf(q) / p
    else:
        scale = np.sqrt((nu - 2) / nu)
        raw = stats.t.ppf(p, nu)
        q = raw * scale
        es = -scale * stats.t.pdf(raw, nu) * (nu + raw**2) / ((nu - 1) * p)
    return -(mean + sigma*q), -(mean + sigma*es)


def coverage_tests(hits, confidence):
    h = np.asarray(hits, dtype=int)
    if len(h) * (1-confidence) < 5 or len(set(h)) < 2:
        return None
    def ll(k, n, p):
        return xlogy(k, p) + xlogy(n-k, 1-p)
    n, k = len(h), h.sum()
    kupiec = 2*(ll(k,n,k/n)-ll(k,n,1-confidence))
    prev, nxt = h[:-1], h[1:]
    groups = [nxt[prev == x] for x in (0,1)]
    if not all(len(g) for g in groups):
        return None
    independent = 2*(sum(ll(g.sum(),len(g),g.mean()) for g in groups)
                     - ll(nxt.sum(),len(nxt),nxt.mean()))
    return float(stats.chi2.sf(max(0,kupiec),1)), float(stats.chi2.sf(max(0,independent),1))


def indicators(price, window=20):
    result = pd.DataFrame({"Price": price})
    for n in (20,50,100,200):
        result[f"SMA{n}"] = price.rolling(n).mean()
    result["EMA20"] = price.ewm(span=20, min_periods=20, adjust=False).mean()
    for n in (21,63,126,252):
        result[f"Momentum {n}"] = price.pct_change(n, fill_method=None)
    change = price.diff()
    gain = change.clip(lower=0).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    result["RSI14"] = (100*gain/(gain+loss)).mask((gain+loss).eq(0), 50)
    mid, std = price.rolling(window).mean(), price.rolling(window).std(ddof=0)
    result["Z-score"] = (price-mid)/std.where(std>0)
    result["Upper"], result["Lower"] = mid+2*std, mid-2*std
    result["Distance"] = price/mid-1
    result["Drawdown"] = price/price.cummax()-1
    return result


def signal(data, z_threshold=2., reduce=-.15, exit_at=-.25):
    row = data.iloc[-1]
    if row[["Price","SMA200","Momentum 63","Drawdown"]].isna().any():
        return {"state":"INSUFFICIENT DATA","reason":"Requires 200 observed sessions."}
    below = row.Price < row.SMA200
    weak = below and row["Momentum 63"] < 0
    persistent = len(data) >= 5 and bool((data.Price < data.SMA200).tail(5).all())
    dd, z = row.Drawdown, row["Z-score"]
    state = ("EXIT CANDIDATE" if weak and persistent and dd <= exit_at else
             "REDUCE CANDIDATE" if weak and persistent and dd <= reduce else
             "WATCH" if below or abs(z) >= z_threshold else "MAINTAIN")
    return {"state":state,
            "reason":f"Price {row.Price:.2f} vs SMA200 {row.SMA200:.2f}; "
                     f"63-session momentum {row['Momentum 63']:.1%} vs 0%; "
                     f"drawdown {dd:.1%} (reduce {reduce:.0%}, exit {exit_at:.0%}); "
                     f"below SMA200 for last 5 sessions: {persistent}; Z {z:.2f} vs ±{z_threshold:g}.",
            "conflict": "Oversold in a weak trend; no reversal inferred." if weak and z <= -z_threshold else "",
            "limit":"Trend and deviation measures do not predict a reversal.",
            "date":str(data.index[-1].date())}
