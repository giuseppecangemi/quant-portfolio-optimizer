"""Stima della volatilità e valutazione mediante finestre temporali
mobili con origine progressiva (rolling-origin), indipendenti
dal backtesting del portafoglio."""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    arch_model = None
    ARCH_AVAILABLE = False

ANNUALIZATION = 252


def load_volatility_prices(tickers, period):
    raw = yf.download(list(dict.fromkeys(tickers)), period=period, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        raise ValueError("No price data returned by Yahoo Finance")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None) if close.index.tz is not None else pd.to_datetime(close.index)
    return close.sort_index().loc[:, ~close.columns.duplicated()]


def ewma_variance(returns, decay=0.94):
    x = np.asarray(returns, dtype=float)
    var = np.full(len(x), np.nan)
    if len(x) < 2:
        return var
    v = float(np.var(x[:min(len(x), 21)], ddof=1))
    for i, r in enumerate(x):
        var[i] = v
        v = decay * v + (1 - decay) * r*r
    return var


def _predict_variance(returns, horizon, model, window, decay):
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < max(30, window // 2):
        raise ValueError("Not enough observations to estimate volatility")
    if model == "Rolling":
        return float(r.iloc[-window:].var(ddof=1))
    if model == "EWMA":
        v = ewma_variance(r.to_numpy(), decay)
        return float(decay * v[-1] + (1-decay) * r.iloc[-1]**2)
    if not ARCH_AVAILABLE:
        raise ImportError("Install 'arch' to run GARCH forecasts")
    if len(r) < 150:
        raise ValueError("At least 150 return observations are needed for GARCH")
    p, o = (1, 1) if model == "GJR-GARCH" else (1, 0)
    kwargs = {"vol": "EGARCH" if model == "EGARCH" else "GARCH", "p": p, "o": o, "q": 1, "dist": "t", "mean": "Zero", "rescale": False}
    # Eseguiamo la stima utilizzando i rendimenti espressi in percentuale
    # per garantire una maggiore stabilità numerica.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = arch_model(r.to_numpy()*100, **kwargs).fit(disp="off", show_warning=False)
        if model == "EGARCH" and horizon > 1:
            # Il calcolo analitico della varianza su più periodi non è supportato
            # dal modello EGARCH: utilizziamo simulazioni riproducibili.
            f = fit.forecast(horizon=horizon, method="simulation", simulations=1000, random_state=np.random.RandomState(42), reindex=False)
        else:
            f = fit.forecast(horizon=horizon, method="analytic", reindex=False)
    predicted = np.asarray(f.variance.iloc[-1], dtype=float) / 10000
    if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0):
        raise ValueError("Non-finite GARCH variance forecast")
    return float(np.mean(predicted))


def analyze_volatility(prices, window=63, horizon=21, model="GARCH(1,1)", decay=0.94):
    p = pd.Series(prices).dropna().astype(float)
    r = np.log(p / p.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    rolling = r.rolling(window).std(ddof=1) * np.sqrt(ANNUALIZATION) * 100
    ewma = pd.Series(np.sqrt(ewma_variance(r.to_numpy(), decay) * ANNUALIZATION)*100, index=r.index)
    prediction = np.sqrt(_predict_variance(r, horizon, model, window, decay) * ANNUALIZATION)*100
    valid = rolling.dropna()
    percentile = float((valid <= valid.iloc[-1]).mean()*100) if len(valid) else float('nan')
    return {"returns": r, "history": pd.DataFrame({"Rolling": rolling, "EWMA": ewma}),
            "summary": {"Rolling volatility (%)": float(valid.iloc[-1]) if len(valid) else float('nan'),
                        "EWMA volatility (%)": float(ewma.iloc[-1]),
                        "Forecast volatility (%)": float(prediction),
                        "Volatility percentile (%)": percentile,
                        "Regime": "High" if percentile >= 75 else ("Low" if percentile <= 25 else "Normal")}}


def backtest_volatility(returns, model, horizon=21, train=500, stride=21, window=63, decay=0.94):
    r = pd.Series(returns).dropna().astype(float)
    records = []
    for end in range(int(train), len(r)-int(horizon)+1, int(stride)):
        past = r.iloc[:end]
        future = r.iloc[end:end+horizon]
        if len(future) != horizon:
            continue
        realized = float(np.mean(future.to_numpy()**2))
        row = {"Forecast date": str(past.index[-1].date()), "Realized daily variance": realized}
        for method in ["Rolling", "EWMA", model]:
            row[method] = _predict_variance(past, horizon, method, window, decay)
        records.append(row)
    df = pd.DataFrame(records)
    if df.empty:
        return df
    methods = list(dict.fromkeys(["Rolling", "EWMA", model]))
    eps = 1e-12
    df.attrs["qlike"] = {m: float(np.mean(np.log(np.maximum(df[m],eps)) + df["Realized daily variance"] / np.maximum(df[m],eps))) for m in methods}
    df.attrs["mse"] = {m: float(np.mean((df[m]-df["Realized daily variance"])**2)) for m in methods}
    return df

# Analisi aggiuntive del portafoglio; le funzioni esistenti
# relative ai singoli asset rimangono invariate.
def portfolio_log_returns(prices, weights):
    """Approssimazione del portafoglio con pesi fissi e ribilanciamento giornaliero;
    non simula le operazioni di trading né i relativi costi di transazione."""
    w = pd.Series(weights, dtype=float)
    w = w[w > 1e-10]
    available = w.index.intersection(prices.columns)
    if len(available) != len(w):
        raise ValueError(f"Missing prices for {len(w)-len(available)} portfolio constituents")
    px = prices.loc[:, list(w.index)].dropna(how="any")
    if len(px) < 180:
        raise ValueError("Insufficient common adjusted-price history for portfolio analysis (minimum 180 sessions).")
    w = w / w.sum()
    daily = px.pct_change().dropna().dot(w)
    if (daily <= -1).any():
        raise ValueError("Invalid portfolio return")
    return np.log1p(daily).rename("Portfolio log return")


def filtered_garch_volatility(returns, model="GARCH(1,1)"):
    """Volatilità condizionata (sigma) stimata in-sample mediante il modello;
    non rappresenta una previsione storica out-of-sample (OOS)."""
    r = pd.Series(returns).dropna().astype(float)
    if not ARCH_AVAILABLE or len(r) < 150:
        return pd.Series(dtype=float)
    p, o = (1, 1) if model == "GJR-GARCH" else (1, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = arch_model(r.to_numpy()*100, mean="Zero", vol="EGARCH" if model == "EGARCH" else "GARCH", p=p, o=o, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False)
    return pd.Series(np.asarray(fit.conditional_volatility)*np.sqrt(ANNUALIZATION), index=r.index, name=model)


def forecast_volatility_path(returns, horizon=21, model="GARCH(1,1)", window=63, decay=0.94):
    """Volatilità condizionata annualizzata per ciascun passo di previsione;
    non rappresenta una previsione dei prezzi futuri."""
    r = pd.Series(returns).dropna().astype(float)
    horizon = int(horizon)
    if model == "Rolling":
        v = float(r.iloc[-window:].var(ddof=1))
        path = np.repeat(v, horizon)
    elif model == "EWMA":
        v = float(decay*ewma_variance(r.to_numpy(), decay)[-1] + (1-decay)*r.iloc[-1]**2)
        path = np.repeat(v, horizon)
    else:
        if not ARCH_AVAILABLE:
            raise ImportError("Install arch to run GARCH forecasts")
        p, o = (1, 1) if model == "GJR-GARCH" else (1, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = arch_model(r.to_numpy()*100, mean="Zero", vol="EGARCH" if model == "EGARCH" else "GARCH", p=p, o=o, q=1, dist="t", rescale=False).fit(disp="off", show_warning=False)
            if model == "EGARCH" and horizon > 1:
                f = fit.forecast(horizon=horizon, method="simulation", simulations=1000, random_state=np.random.RandomState(42), reindex=False)
            else:
                f = fit.forecast(horizon=horizon, method="analytic", reindex=False)
        path = np.asarray(f.variance.iloc[-1], dtype=float)/10000
    if np.any(~np.isfinite(path)) or np.any(path < 0):
        raise ValueError(f"Invalid forecast for {model}")
    return pd.Series(np.sqrt(path*ANNUALIZATION)*100, index=pd.Index(np.arange(1,horizon+1), name="Forecast session"), name=model)


def compare_volatility_models(returns, horizon=21, train=500, stride=21, window=63, decay=0.94, models=None):
    """Stesse origini delle finestre mobili e stessi periodi di osservazione
    dei valori effettivamente realizzati per tutti i modelli."""
    models = models or ["Rolling", "EWMA", "GARCH(1,1)", "GJR-GARCH", "EGARCH"]
    r = pd.Series(returns).dropna().astype(float)
    records, errors = [], {}
    for end in range(int(train), len(r)-int(horizon)+1, int(stride)):
        past, future = r.iloc[:end], r.iloc[end:end+int(horizon)]
        row = {"Forecast date": str(past.index[-1].date()), "Realized daily variance": float(np.mean(future.to_numpy()**2))}
        for m in models:
            try:
                row[m] = _predict_variance(past, horizon, m, window, decay)
            except Exception as exc:
                row[m] = np.nan
                errors[m] = str(exc)
        records.append(row)
    df = pd.DataFrame(records)
    if df.empty:
        return df, pd.DataFrame(), errors
    # Valutazione appaiata: utilizziamo le stesse date di origine delle previsioni
    # per tutti i modelli che dispongono di previsioni.
    valid_models = [m for m in models if m in df and df[m].notna().all()]
    scores = []
    eps = 1e-12
    for m in valid_models:
        f = df[m].clip(lower=eps)
        realized = df["Realized daily variance"]
        scores.append({"Model":m,"QLIKE":float((np.log(f)+realized/f).mean()),"MSE (variance)":float(((f-realized)**2).mean()),"OOS windows":len(df)})
    return df, pd.DataFrame(scores).sort_values("QLIKE") if scores else pd.DataFrame(), errors


def portfolio_risk_contributions(prices, weights):
    """Calcola la matrice di covarianza annualizzata e i contributi al rischio
    secondo la decomposizione di Eulero, utilizzando pesi target fissi
    del portafoglio."""
    w = pd.Series(weights, dtype=float)
    w = w[w > 1e-10]
    px = prices.loc[:, list(w.index)].dropna()
    daily = px.pct_change().dropna()
    w = w/w.sum()
    cov = daily.cov()*ANNUALIZATION
    sigma = float(np.sqrt(w.to_numpy() @ cov.to_numpy() @ w.to_numpy()))
    contrib = w * cov.dot(w) / sigma if sigma > 0 else w*0
    return pd.DataFrame({"Weight (%)":w*100,"Risk contribution (vol pp)":contrib*100,"Risk share (%)":contrib/sigma*100 if sigma>0 else contrib*0}), sigma*100
