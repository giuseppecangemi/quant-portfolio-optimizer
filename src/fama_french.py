"""Analisi del modello europeo Fama-French a tre fattori (FF3),
con dati giornalieri dei fattori espressi in USD."""
from io import BytesIO, StringIO
from zipfile import ZipFile
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd

EUROPE_FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "Europe_3_Factors_Daily_CSV.zip"
)


def load_europe_ff3() -> pd.DataFrame:
    """Scarica i fattori giornalieri europei di Kenneth French,
    con rendimenti espressi in USD e convertiti in formato decimale."""
    req = Request(EUROPE_FF3_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        raw = response.read()
    with ZipFile(BytesIO(raw)) as archive:
        name = next((n for n in archive.namelist() if n.lower().endswith('.csv')), None)
        if name is None:
            raise ValueError("Europe FF3 archive does not contain a CSV file.")
        lines = archive.read(name).decode('utf-8-sig', errors='replace').splitlines()
    # La libreria include un testo introduttivo e una sezione con dati annuali:
    # conserviamo esclusivamente le osservazioni giornaliere identificate
    # da date a otto cifre nel formato YYYYMMDD.
    records = []
    for line in lines:
        cells = [v.strip() for v in line.split(',')]
        if len(cells) < 5 or len(cells[0]) != 8 or not cells[0].isdigit():
            continue
        try:
            date = pd.to_datetime(cells[0], format='%Y%m%d', errors='raise')
            values = [float(cells[j]) / 100.0 for j in range(1, 5)]
        except (ValueError, IndexError):
            continue
        if all(np.isfinite(values)) and all(abs(v) < 0.5 for v in values):
            records.append((date, *values))
    if not records:
        raise ValueError("No valid daily observations found in Europe FF3 data.")
    return (pd.DataFrame(records, columns=['Date', 'MKT-RF', 'SMB', 'HML', 'RF'])
            .drop_duplicates('Date').set_index('Date').sort_index())


def fit_ff3(asset_usd_returns: pd.DataFrame, factors: pd.DataFrame,
            min_observations: int = 80) -> pd.DataFrame:
    """Regressione OLS con intercetta; i rendimenti attesi in USD
    vengono stimati utilizzando le medie storiche dei fattori."""
    rows = []
    for ticker in asset_usd_returns.columns:
        joined = pd.concat([asset_usd_returns[ticker].rename('Asset'), factors], axis=1,
                           join='inner').dropna()
        if len(joined) < min_observations:
            raise ValueError(f"Insufficient common FF3 observations for {ticker}: {len(joined)}")
        y = (joined['Asset'] - joined['RF']).to_numpy(float)
        x = joined[['MKT-RF', 'SMB', 'HML']].to_numpy(float)
        X = np.column_stack([np.ones(len(x)), x])
        coefficients, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank != 4:
            raise ValueError(f"Singular FF3 regression for {ticker}")
        fitted = X @ coefficients
        residual = y - fitted
        tss = np.sum((y-y.mean())**2)
        r2 = 1 - np.sum(residual**2)/tss if tss > 0 else np.nan
        alpha, market, smb, hml = coefficients
        # Il rendimento atteso ex-ante del modello esclude per costruzione
        # l'alpha stimato tramite regressione.
        usd_expected_daily = float(joined['RF'].mean() + np.dot(
            coefficients[1:], joined[['MKT-RF','SMB','HML']].mean().to_numpy()))
        rows.append({'Ticker': ticker, 'Alpha (annualized)': alpha*252,
                     'Beta MKT': market, 'Beta SMB': smb, 'Beta HML': hml,
                     'R Squared': r2, 'Observations': len(joined),
                     'FF3 Expected Return USD': usd_expected_daily*252,
                     'Historical Return USD': joined['Asset'].mean()*252})
    return pd.DataFrame(rows).set_index('Ticker')


def convert_eur_prices_to_usd(asset_prices_eur: pd.DataFrame,
                              eurusd_prices: pd.DataFrame) -> pd.DataFrame:
    """Il tasso di cambio EURUSD=X esprime il valore di un euro in dollari USD;
    moltiplicare i prezzi in EUR per questo tasso li converte in USD."""
    fx = eurusd_prices.iloc[:, 0].reindex(asset_prices_eur.index).ffill()
    converted = asset_prices_eur.mul(fx, axis=0).dropna()
    if converted.empty or len(converted) < 81:
        raise ValueError("Insufficient aligned EUR/USD exchange-rate history.")
    return converted


def estimate_ff3_eur_returns(historical_prices_eur, factors, eurusd_prices,
                             min_observations=80, return_stats=False):
    """Calcola i rendimenti attesi in EUR secondo il modello Fama-French
    a tre fattori (FF3), utilizzando esclusivamente i dati storici
    disponibili al momento della stima (point-in-time).

    I premi per il rischio dei fattori sono espressi in USD; i titoli
    denominati in EUR vengono convertiti in USD prima della stima.

    La variazione attesa del tasso di cambio viene stimata utilizzando
    la stessa finestra storica, senza ricorrere a dati futuri.
    """
    eur = historical_prices_eur.sort_index().dropna()
    fx = eurusd_prices.iloc[:, 0] if isinstance(eurusd_prices, pd.DataFrame) else eurusd_prices
    fx = fx.sort_index().reindex(eur.index).ffill()
    if fx.isna().any() or (fx <= 0).any():
        raise ValueError("Missing historical EUR/USD exchange rates for FF3 estimation.")
    usd_returns = eur.mul(fx, axis=0).pct_change().dropna()
    available_factors = factors.loc[factors.index < eur.index[-1]]
    stats = fit_ff3(usd_returns, available_factors, min_observations=min_observations)
    common_dates = usd_returns.index.intersection(available_factors.index)
    fx_returns = fx.pct_change().reindex(common_dates).dropna()
    if len(fx_returns) < min_observations:
        raise ValueError("Insufficient FX history for point-in-time FF3.")
    mu_usd_daily = stats['FF3 Expected Return USD'] / 252.0
    mu_eur = ((1.0 + mu_usd_daily) / (1.0 + float(fx_returns.mean())) - 1.0) * 252.0
    expected = mu_eur.reindex(eur.columns)
    return (expected, stats) if return_stats else expected
