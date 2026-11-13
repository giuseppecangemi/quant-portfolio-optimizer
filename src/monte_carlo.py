"""Simulazione di scenari futuri basata sui rendimenti storici out-of-sample
del portafoglio, opportunamente allineati.

Questo modulo è volutamente separato dai motori di costruzione del
portafoglio e di backtesting.

Le simulazioni sono condizionate alla distribuzione empirica dei rendimenti
e non costituiscono previsioni dei prezzi futuri.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def extract_equity(result: dict, key: str) -> pd.Series | None:
    raw = result.get(key)
    if isinstance(raw, pd.DataFrame):
        if "Portfolio Value" not in raw:
            return None
        raw = raw["Portfolio Value"]
    if not isinstance(raw, pd.Series):
        return None
    equity = pd.to_numeric(raw, errors="coerce").copy()
    equity.index = pd.to_datetime(equity.index, errors="coerce")
    equity = equity.loc[~equity.index.isna()]
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    if len(equity) < 3 or not np.isfinite(equity).all() or (equity <= 0).any():
        return None
    if "initial_capital" in raw.attrs:
        equity.attrs["initial_capital"] = raw.attrs["initial_capital"]
    if isinstance(result.get(key), pd.DataFrame) and "initial_capital" in result[key].attrs:
        equity.attrs["initial_capital"] = result[key].attrs["initial_capital"]
    return equity


def aligned_returns(equities: dict[str, pd.Series]) -> pd.DataFrame:
    if not equities:
        raise ValueError("Nessuna curva storica valida disponibile.")
    # The union of source valuations is the observed session calendar.
    # Reindex levels before differencing so both endpoints must be present.
    levels = pd.concat(equities, axis=1).sort_index()
    if np.isinf(levels.to_numpy(dtype=float)).any() or (levels <= 0).any().any():
        raise ValueError("NAV storico infinito o non positivo.")
    daily = levels.pct_change(fill_method=None).dropna(how="any")
    if len(daily) < 40:
        raise ValueError(f"Solo {len(daily)} intervalli comuni validi: ne servono almeno 40.")
    if not np.isfinite(daily.to_numpy()).all() or (daily <= -1).any().any():
        raise ValueError("Rendimenti storici non finiti o <= -100%.")
    daily.attrs["session_positions"] = dict(zip(levels.index, range(len(levels))))
    entry = {}
    for name, equity in equities.items():
        capital = equity.attrs.get("initial_capital")
        if capital is not None:
            if not _finite_real(capital) or capital <= 0:
                raise ValueError(f"Capitale iniziale storico non valido: {name}.")
            ratio = float(equity.sort_index().iloc[0] / capital)
            if not 0 < ratio <= 1 + 1e-10:
                raise ValueError(f"NAV di ingresso incompatibile con il capitale iniziale: {name}.")
            entry[name] = min(1., ratio)
    daily.attrs["entry_multipliers"] = entry
    return daily


def _finite_real(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


def _integer_between(value, low, high):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and low <= value <= high


def empirical_tail_mean(pnl, probability):
    """Average the lowest probability mass of an equally weighted sample."""
    ordered = np.sort(pnl)
    mass = probability * len(ordered)
    whole = int(mass)
    weights = np.clip(mass - np.arange(whole + 1), 0, 1)[:len(ordered)]
    return float(np.dot(ordered[:len(weights)], weights / mass))


def simulate(
    historical_returns: pd.DataFrame,
    *,
    years: int = 1,
    n_paths: int = 1000,
    initial_capital: float = 10000.0,
    method: str = "Block bootstrap",
    block_size: int = 5,
    seed: int = 42,
    annual_drift_adjustment: float = 0.0,
    return_paths: bool = False,
) -> tuple:
    """Restituisce (metriche, percorsi percentili, valori finali) per tutte
    le strategie selezionate.

    Per tutte le strategie vengono utilizzati gli stessi giorni storici
    campionati e gli stessi shock multivariati.

    La media empirica dei rendimenti viene mantenuta; l'eventuale
    correzione del rendimento atteso (drift) è espressa in unità
    decimali annualizzate.
    """
    if not _integer_between(years, 1, 10) or not _integer_between(n_paths, 100, 10000):
        raise ValueError("Anni (1–10) e scenari (100–10000) devono essere interi.")
    if not _finite_real(initial_capital) or initial_capital <= 0:
        raise ValueError("Il capitale iniziale deve essere positivo e finito.")
    if not _finite_real(annual_drift_adjustment):
        raise ValueError("L'aggiustamento annuo deve essere finito.")
    if not _integer_between(seed, 0, 2**64-1):
        raise ValueError("Il seed deve essere un intero non negativo entro 64 bit.")
    if method not in ("Block bootstrap", "IID bootstrap", "Multivariate Gaussian"):
        raise ValueError(f"Metodo sconosciuto: {method}")
    if not isinstance(historical_returns, pd.DataFrame) or not historical_returns.shape[1]:
        raise ValueError("Seleziona almeno una strategia con rendimenti storici.")
    if not historical_returns.columns.is_unique or not historical_returns.index.is_unique:
        raise ValueError("Strategie o date duplicate nei rendimenti storici.")
    try:
        hist = historical_returns.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("I rendimenti storici devono essere numerici.") from exc
    if len(hist) < 40 or not np.isfinite(hist).all() or (hist <= -1).any():
        raise ValueError("Servono almeno 40 rendimenti finiti per strategia, tutti superiori a -100%.")
    entry = historical_returns.attrs.get("entry_multipliers", {})
    ratios = [entry.get(name, 1.) for name in historical_returns.columns]
    if any(not _finite_real(ratio) or not 0 < ratio <= 1 for ratio in ratios):
        raise ValueError("Coefficienti del costo iniziale non validi.")
    n_days = years * TRADING_DAYS
    rng = np.random.default_rng(seed)
    if method == "Block bootstrap":
        if not _integer_between(block_size, 1, len(hist)):
            raise ValueError("La lunghezza del blocco deve essere un intero positivo entro il campione.")
        valid_starts = np.arange(len(hist) - block_size + 1)
        positions = historical_returns.attrs.get("session_positions")
        if positions is not None:
            session = np.array([positions[date] for date in historical_returns.index])
            valid_starts = valid_starts[
                session[block_size-1:] - session[:len(valid_starts)] == block_size-1
            ]
        if not len(valid_starts):
            raise ValueError("Nessun blocco di sedute consecutive disponibile: riduci la lunghezza.")
        n_blocks = (n_days + block_size - 1) // block_size
        starts = valid_starts[rng.integers(0, len(valid_starts), size=(n_paths, n_blocks))]
        indices = (starts[:, :, None] + np.arange(block_size)).reshape(n_paths, -1)[:, :n_days]
        shocks = hist[indices]
    elif method == "IID bootstrap":
        indices = rng.integers(0, len(hist), size=(n_paths, n_days))
        shocks = hist[indices]
    elif method == "Multivariate Gaussian":
        mu = hist.mean(axis=0)
        cov = np.atleast_2d(np.cov(hist, rowvar=False))
        cov = cov / 2 + cov.T / 2
        if not np.isfinite(mu).all() or not np.isfinite(cov).all():
            raise ValueError("Media o covarianza non finita: controlla i rendimenti estremi.")
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        cov = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        shocks = rng.multivariate_normal(mu, cov, size=(n_paths, n_days), check_valid="ignore")
    else:
        raise ValueError(f"Metodo sconosciuto: {method}")
    # Correzione additiva approssimativa dei rendimenti aritmetici giornalieri attesi.
    shocks = shocks + float(annual_drift_adjustment) / TRADING_DAYS
    if not np.isfinite(shocks).all():
        raise ValueError("Scenari non finiti: controlla i rendimenti e lo stress.")
    if np.any(shocks <= -1):
        raise ValueError("Uno scenario genera una perdita giornaliera >=100%; riduci lo stress sulla media.")
    # Includiamo il capitale iniziale al tempo t=0.
    growth = np.concatenate([np.ones((n_paths, 1, hist.shape[1])), np.cumprod(1 + shocks, axis=1)], axis=1)
    values = growth * initial_capital
    values[:, 1:, :] *= ratios
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Capitalizzazione fuori dai limiti numerici: riduci orizzonte o stress.")
    final = values[:, -1, :]
    running_peak = np.maximum.accumulate(values, axis=1)
    max_dd = np.minimum((values / running_peak - 1).min(axis=1), np.array(ratios) - 1)
    metrics = []
    pct_paths = {}
    final_values = {}
    for j, name in enumerate(historical_returns.columns):
        terminal = final[:, j]
        pnl = terminal - initial_capital
        cutoff = float(np.quantile(pnl, 0.05))
        tail_mean = empirical_tail_mean(pnl, 0.05)
        daily = shocks[:, :, j].ravel()
        daily_var = max(0., -float(np.quantile(daily, .05)))
        daily_es = max(0., -empirical_tail_mean(daily, .05))
        metrics.append({
            "Strategy": name,
            "Median final (€)": float(np.median(terminal)),
            "Mean final (€)": float(np.sum(terminal / len(terminal))),
            "P5 final (€)": float(np.quantile(terminal, 0.05)),
            "P95 final (€)": float(np.quantile(terminal, 0.95)),
            "Probability of loss (%)": float(np.mean(terminal < initial_capital) * 100),
            "VaR 95% (€)": float(max(0, -cutoff)),
            "ES 95% (€)": max(0., -tail_mean),
            "Daily VaR 95% (€)": initial_capital * daily_var,
            "Daily ES 95% (€)": initial_capital * daily_es,
            "Daily VaR 95% (%)": 100 * daily_var,
            "Daily ES 95% (%)": 100 * daily_es,
            "Probability MDD ≤ -20% (%)": float(np.mean(max_dd[:, j] <= -0.20) * 100),
        })
        p = np.percentile(values[:, :, j], [5, 25, 50, 75, 95], axis=0)
        pct_paths[name] = pd.DataFrame(p.T, columns=["P5", "P25", "Median", "P75", "P95"], index=pd.RangeIndex(n_days + 1, name="Trading day"))
        final_values[name] = terminal
    result = (pd.DataFrame(metrics).set_index("Strategy"), pct_paths, pd.DataFrame(final_values))
    if not np.isfinite(result[0].to_numpy()).all() or any(
        not np.isfinite(frame.to_numpy()).all() for frame in pct_paths.values()
    ):
        raise ValueError("Statistiche non finite: controlla capitale e rendimenti estremi.")
    if return_paths:
        if values.max() > np.finfo(np.float32).max:
            raise ValueError("Traiettorie fuori dall’intervallo numerico del grafico.")
        # Struttura: (simulazione, giorno di negoziazione, strategia);
        # memorizziamo i dati in modo compatto per la rappresentazione grafica.
        return (*result, {name: values[:, :, j].astype(np.float32) for j, name in enumerate(historical_returns.columns)})
    return result
