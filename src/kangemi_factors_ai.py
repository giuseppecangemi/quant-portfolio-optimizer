from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import zipfile

import pandas as pd
import streamlit as st

from src.kangemi_ai import _request_analysis, _num, _table



FACTOR_PROMPT = """Sei K, l'analista quantitativo e guida educativa di Kangemi Edu – Quantitative Boutique.

LA TUA MISSIONE
Aiuta l'utente a comprendere i modelli fattoriali della finanza, non soltanto a leggerne i risultati.
Spiega cosa sono, perché sono stati sviluppati, quali domande cercano di risolvere e come interpretare le stime ottenute.
L'obiettivo è trasformare l'analisi quantitativa in un percorso di apprendimento chiaro, coinvolgente e rigoroso.

Scrivi in italiano professionale, elegante, naturale e accessibile.
Assumi che il lettore sia interessato alla finanza, ma non necessariamente esperto di econometria.
Introduci i concetti tecnici progressivamente, spiegandoli prima di utilizzarli.

REGOLE DI RIGORE
Usa esclusivamente il JSON per i risultati numerici specifici.
Puoi utilizzare conoscenze teoriche consolidate per spiegare CAPM, Fama-French, regressioni e premi per il rischio.
Distingui le proprietà teoriche dei modelli dalle evidenze realmente presenti nei dati.
Analizza soltanto i modelli effettivamente disponibili nel JSON.
Non inventare coefficienti, p-value, significatività statistica, correlazioni o risultati non forniti.
Le stime in-sample non sono risultati di backtesting e non rappresentano rendimenti futuri garantiti.
Non fornire consigli personalizzati di investimento.

STRUTTURA DEL REPORT

1. INTRODUZIONE — PERCHÉ ESISTONO I MODELLI FATTORIALI
Spiega che il rendimento di un'azione non dipende necessariamente soltanto dalle caratteristiche individuali dell'impresa.
Introduci il concetto di fattore di rischio e di esposizione sistematica.
Spiega la differenza tra rischio sistematico e rischio specifico.
Presenta l'obiettivo dei modelli fattoriali: descrivere e stimare la relazione tra rendimenti e fattori comuni.
Chiarisci che un modello fattoriale non è una macchina capace di prevedere con certezza il mercato.

2. CAPM — IL RISCHIO DI MERCATO

Spiega il Capital Asset Pricing Model e la sua intuizione economica.
Introduci il rendimento privo di rischio, il premio per il rischio di mercato e il Beta.

Spiega la relazione teorica:
Rendimento atteso = Risk-free rate + Beta × Premio per il rischio di mercato.

Interpreta:
- Beta: sensibilità del titolo ai movimenti del mercato.
- Beta superiore a 1: maggiore sensibilità stimata al mercato.
- Beta tra 0 e 1: sensibilità positiva inferiore a quella del mercato.
- Beta negativo: relazione inversa stimata rispetto al mercato.
- Alpha: intercetta della regressione, non prova automatica di abilità o sovraperformance.
- R²: quota della variabilità dei rendimenti spiegata dal modello nel campione.
- Market Correlation: intensità e direzione della relazione lineare con il benchmark.
- Historical Return: rendimento medio storico annualizzato secondo la metodologia adottata.
- CAPM Expected Return: rendimento implicito nel CAPM sulla base degli input stimati.

Spiega perché un Beta elevato non implica automaticamente un rendimento realizzato più alto.
Chiarisci che un R² elevato non garantisce capacità predittiva futura.

Dopo la spiegazione teorica, analizza i risultati CAPM degli asset presenti.
Individua i titoli con Beta maggiori e minori, eventuali Alpha positivi o negativi,
differenze nei rendimenti e capacità esplicativa del modello.

3. FAMA-FRENCH THREE-FACTOR MODEL — OLTRE IL MERCATO

Se FF3 è disponibile, spiega perché il solo fattore di mercato potrebbe non descrivere completamente
le differenze storiche tra i rendimenti azionari.

Introduci i tre fattori:
- MKT-RF: rendimento del mercato in eccesso rispetto al tasso privo di rischio.
- SMB, Small Minus Big: differenziale di rendimento tra portafogli di società piccole e grandi.
- HML, High Minus Low: differenziale di rendimento tra portafogli value e growth,
  secondo le definizioni dei portafogli fattoriali.

Spiega che i coefficienti di regressione rappresentano sensibilità stimate ai fattori.
Un'esposizione positiva a SMB suggerisce una sensibilità positiva al fattore size,
non necessariamente che il titolo sia una small cap.
Un'esposizione positiva a HML suggerisce sensibilità al fattore value,
non necessariamente che il titolo sia classificabile come value in ogni periodo.

Interpreta le esposizioni effettivamente disponibili, Alpha e R² se presenti.
Spiega cosa cambia rispetto al CAPM e quali aspetti il modello cerca di catturare.

4. FAMA-FRENCH FIVE-FACTOR MODEL — PROFITTABILITÀ E INVESTIMENTI

Se FF5 è disponibile, spiega l'estensione del modello a cinque fattori.
Riprendi MKT-RF, SMB e HML, poi introduci:

- RMW, Robust Minus Weak: differenziale associato alla profittabilità operativa.
- CMA, Conservative Minus Aggressive: differenziale associato agli investimenti delle imprese.

Spiega l'intuizione economica dietro profittabilità e politiche di investimento.
Interpreta le esposizioni stimate, senza trasformarle automaticamente in caratteristiche
fondamentali osservate delle singole società.
Chiarisci che un modello più articolato non è necessariamente migliore fuori campione.

5. COME I MODELLI ENTRANO NEL PORTAFOGLIO

Spiega il passaggio dalla stima dei rendimenti attesi all'ottimizzazione Maximum Sharpe.
Distingui i coefficienti fattoriali, gli Alpha e i rendimenti attesi utilizzati dall'ottimizzatore.
Spiega perché modificare le stime dei rendimenti può modificare sensibilmente i pesi ottimali.

Analizza per ogni modello disponibile:
- Rendimento atteso annualizzato.
- Volatilità annualizzata.
- Sharpe Ratio.
- Distribuzione dei pesi.
- Concentrazione e principali posizioni.
- Capitale investito e liquidità residua.

Distingui i pesi target dall'allocazione effettivamente realizzabile.

6. CONFRONTO TRA CAPM, FF3 E FF5

Quando sono disponibili più modelli, confrontali quantitativamente.
Spiega quali differenze emergono nelle esposizioni fattoriali e nei portafogli costruiti.
Identifica eventuali cambiamenti significativi nei rendimenti attesi, nella volatilità,
nello Sharpe e nella concentrazione.

Non dichiarare che un modello sia superiore solo perché presenta uno Sharpe stimato maggiore.
Se periodi di stima, valute, benchmark o campioni non sono omogenei,
segnala chiaramente che il confronto diretto è limitato.

7. LIMITI ECONOMETRICI E METODOLOGICI

Spiega con linguaggio accessibile:
- Dipendenza dalle osservazioni storiche.
- Instabilità temporale dei coefficienti.
- Sensibilità alla scelta del benchmark.
- Differenza tra correlazione, spiegazione statistica e causalità.
- Rischio di overfitting e di stima.
- Differenza tra R² in-sample e capacità predittiva out-of-sample.
- Possibili differenze tra fattori denominati in USD e portafogli in EUR.
- Limiti derivanti dalla copertura dei dati fattoriali.

Non inventare test statistici o risultati di significatività.
Non definire statisticamente significativo un Alpha senza evidenza adeguata.

8. CONCLUSIONI — COSA ABBIAMO IMPARATO

Riassumi le principali evidenze quantitative effettivamente osservate.
Spiega che cosa aggiunge ciascun modello alla comprensione dei rendimenti.
Indica quali risultati meritano particolare attenzione e perché.
Suggerisci verifiche successive, come backtesting out-of-sample,
stabilità delle esposizioni o confronto dei modelli su finestre temporali omogenee.

STILE EDITORIALE
Usa titoli Markdown chiari e paragrafi fluidi.
Non limitarti a ripetere le tabelle numeriche.
Per ogni concetto importante segui idealmente questa sequenza:
spiegazione intuitiva, significato quantitativo, applicazione ai risultati.
Usa esempi semplici quando aiutano la comprensione, specificando che sono illustrativi.
Alterna teoria e dati per mantenere il lettore coinvolto.
Non usare linguaggio promozionale, frasi ridondanti o tecnicismi non spiegati.
Il report deve essere approfondito ma piacevole da leggere.
"""



def _model_payload(name: str, result: dict) -> dict:
    table = result.get("table")
    stats = result.get("stats")
    weights = result.get("weights")
    expected = result.get("expected_returns_eur", result.get("expected_returns"))
    invested = _num(result.get("total_invested"))
    capital = _num(result.get("capital"))
    cash = _num(result.get("cash"))
    if cash is None and invested is not None and capital is not None:
        cash = capital - invested
    data = {
        "model": name,
        "estimation_period": str(result.get("period", "unknown")),
        "expected_portfolio_return_annual": _num(result.get("return")),
        "portfolio_volatility_annual": _num(result.get("volatility")),
        "portfolio_sharpe": _num(result.get("sharpe")),
        "total_invested_eur": invested,
        "residual_cash_eur": cash,
        "fractional_shares": result.get("use_fractional_shares"),
        "weights": {str(k): _num(v) for k, v in weights.items()} if isinstance(weights, pd.Series) else None,
        "allocation": _table(table) if isinstance(table, pd.DataFrame) else None,
        "asset_statistics": _table(stats.reset_index().rename(columns={stats.index.name or 'index': 'Ticker'})) if isinstance(stats, pd.DataFrame) else None,
        "asset_expected_returns": {str(k): _num(v) for k, v in expected.items()} if isinstance(expected, pd.Series) else None,
    }
    if name == "CAPM":
        data["market_proxy"] = "FTSE MIB"
        data["market_expected_return_annual"] = _num(result.get("market_expected_return"))
        data["covariance_annual"] = _table(result["covariance"].reset_index()) if isinstance(result.get("covariance"), pd.DataFrame) else None
        data["efficient_frontier"] = _table(result["frontier"]) if isinstance(result.get("frontier"), pd.DataFrame) else None
        asset_ret = result.get("asset_returns")
        market_ret = result.get("market_returns")
        data["asset_return_observations"] = len(asset_ret) if isinstance(asset_ret, pd.DataFrame) else None
        data["market_return_observations"] = len(market_ret) if isinstance(market_ret, pd.Series) else None
    else:
        data["factor_observations"] = result.get("factor_observations")
        data["annualized_usd_eur_fx_drift"] = _num(result.get("fx_drift_annualized"))
        data["methodology_note"] = "Factor means historically estimated in USD; EUR conversion uses historical FX drift approximation; regression alpha excluded from expected returns."
    return data


def factors_payload(capm: dict | None, ff3: dict | None, ff5: dict | None, *, capital: float | None = None, risk_free_rate: float | None = None) -> dict:
    models = {}
    for name, result in (("CAPM", capm), ("FF3", ff3), ("FF5", ff5)):
        if isinstance(result, dict):
            models[name] = _model_payload(name, result)
            if models[name]["residual_cash_eur"] is None and capital is not None and models[name]["total_invested_eur"] is not None:
                models[name]["residual_cash_eur"] = float(capital) - models[name]["total_invested_eur"]
    return {
        "section": "Factor Models",
        "risk_free_rate_annual": _num(risk_free_rate),
        "capital_eur": _num(capital),
        "models": models,
        "warning": "In-sample factor-based estimates, not out-of-sample backtests.",
    }


def factors_zip(capm: dict | None, ff3: dict | None, ff5: dict | None, *, capital=None, risk_free_rate=None) -> bytes:
    payload = factors_payload(capm, ff3, ff5, capital=capital, risk_free_rate=risk_free_rate)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps({"section": "Factor Models", "capital_eur": _num(capital), "risk_free_rate_annual": _num(risk_free_rate), "periods": {k: v["estimation_period"] for k, v in payload["models"].items()}}, ensure_ascii=False, indent=2))
        summary = []
        for name, result in (("CAPM", capm), ("FF3", ff3), ("FF5", ff5)):
            if not isinstance(result, dict):
                continue
            p = payload["models"][name]
            summary.append({k: v for k, v in p.items() if k in ("model", "estimation_period", "expected_portfolio_return_annual", "portfolio_volatility_annual", "portfolio_sharpe", "total_invested_eur", "residual_cash_eur", "factor_observations", "annualized_usd_eur_fx_drift")})
            prefix = name.lower()
            for key, suffix in (("stats", "asset_statistics"), ("table", "allocation"), ("covariance", "covariance"), ("frontier", "efficient_frontier")):
                df = result.get(key)
                if isinstance(df, pd.DataFrame):
                    archive.writestr(f"{prefix}_{suffix}.csv", df.to_csv(index=(key != "table"), index_label="Ticker" if key in ("stats", "covariance") else None))
            for key, suffix in (("weights", "weights"), ("expected_returns_eur", "expected_returns_eur"), ("expected_returns", "expected_returns")):
                series = result.get(key)
                if isinstance(series, pd.Series):
                    archive.writestr(f"{prefix}_{suffix}.csv", series.rename(suffix).to_csv(index=True, index_label="Ticker"))
        archive.writestr("factor_models_summary.csv", pd.DataFrame(summary).to_csv(index=False))
    return output.getvalue()


def render_factors_k(capm=None, ff3=None, ff5=None, *, capital=None, risk_free_rate=None) -> None:
    if not any(isinstance(r, dict) for r in (capm, ff3, ff5)):
        return
    st.divider()
    st.subheader("K · L'AI di Kangemi Edu")
    st.caption("Analisi fattoriale CAPM / FF3 / FF5 dei risultati calcolati; nessun dato di mercato in tempo reale.")
    payload = factors_payload(capm, ff3, ff5, capital=capital, risk_free_rate=risk_free_rate)
    fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if st.session_state.get("k_factors_fingerprint") != fingerprint:
        st.session_state.pop("k_factors_report", None)
        st.session_state["k_factors_fingerprint"] = fingerprint
    st.download_button("Scarica i risultati Factor Models (CSV)", data=factors_zip(capm, ff3, ff5, capital=capital, risk_free_rate=risk_free_rate), file_name="kangemi_factor_models_csv.zip", mime="application/zip", key="k_factors_csv")
    if st.button("Chiedi un'analisi a K · L'AI di Kangemi Edu", key="k_factors_analyze", type="primary"):
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
            model = st.secrets.get("K_AI_MODEL", "gpt-4o-mini")
        except FileNotFoundError:
            api_key, model = "", "gpt-4o-mini"
        if not api_key:
            st.warning("Per attivare K, configura OPENAI_API_KEY nei Secrets di Streamlit.")
        else:
            try:
                with st.spinner("K sta analizzando CAPM, FF3 e FF5..."):
                    # Existing API helper reads the global prompt; pass the factor-specific
                    # instructions in the input to avoid modifying the working optimization integration.
                    report = _request_analysis({"task_instructions": FACTOR_PROMPT, "data": payload}, api_key, model)
                st.session_state["k_factors_report"] = report
            except urllib.error.HTTPError as exc:
                try:
                    body = json.loads(exc.read().decode("utf-8"))
                    message = body.get("error", {}).get("message", "Errore API")
                except (ValueError, UnicodeError):
                    message = "Errore API senza dettagli leggibili"
                st.error(f"OpenAI HTTP {exc.code} — {message}")
            except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
                st.error(f"Non è stato possibile generare l'analisi: {exc}")
    report = st.session_state.get("k_factors_report")
    if report:
        st.markdown(report)
        st.download_button("Scarica l'analisi di K (.md)", report.encode("utf-8"), "kangemi_k_factor_models_analysis.md", "text/markdown", key="k_factors_report_download")
