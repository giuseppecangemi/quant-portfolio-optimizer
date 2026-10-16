from __future__ import annotations

import io
import json
import math
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone

import pandas as pd
import streamlit as st


MODELS = (
    ("Minimum Volatility", "min"),
    ("Maximum Sharpe", "max"),
    ("Risk Parity", "risk_parity"),
    ("Hierarchical Risk Parity", "hrp"),
)


def _num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _table(df: pd.DataFrame) -> list[dict]:
    clean = df.copy().reset_index(drop=True)
    clean = clean.where(pd.notna(clean), None)
    return json.loads(clean.to_json(orient="records", date_format="iso"))


def optimization_payload(r: dict) -> dict:
    """Use the same session-state results that feed the visible UI."""
    summary = []
    allocations = {}
    for name, prefix in MODELS:
        weights = r[f"{prefix}_weights"]
        allocation = r[f"{prefix}_table"].copy()
        allocation.insert(0, "Ticker", [str(x) for x in allocation.index]) if "Ticker" not in allocation.columns else None
        allocations[name] = _table(allocation)
        summary.append({
            "model": name,
            "expected_return_annual": _num(r[f"{prefix}_return"]),
            "volatility_annual": _num(r[f"{prefix}_volatility"]),
            "sharpe_ratio": _num(r[f"{prefix}_sharpe"]),
            "total_invested": _num(r[f"{prefix}_total_invested"]),
            "residual_cash": _num(r[f"{prefix}_cash"]),
            "weights": {str(k): _num(v) for k, v in weights.items()},
        })
    return {
        "section": "Portfolio Optimization",
        "estimation_period": str(r["period"]),
        "capital": _num(r["capital"]),
        "risk_free_rate_annual": _num(r["risk_free_rate"]),
        "fractional_shares": bool(r["use_fractional_shares"]),
        "portfolio_summary": summary,
        "allocations": allocations,
        "risk_parity_risk_contributions": {
            str(k): _num(v) for k, v in r["risk_parity_contributions"].items()
        },
        "hrp_cluster_order": [str(x) for x in r["hrp_order"]],
        "asset_expected_returns_annual": {
            str(k): _num(v) for k, v in r["expected_returns"].items()
        },
        "asset_volatilities_annual": {
            str(k): _num(v) for k, v in r["volatility"].items()
        },
        "covariance_annual": _table(r["covariance"].rename_axis("Ticker")),
        "efficient_frontier": _table(r["frontier"]),
    }


def optimization_zip(r: dict) -> bytes:
    """Download separate CSVs; preserve numeric precision, not display formatting."""
    payload = optimization_payload(r)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("portfolio_summary.csv", pd.DataFrame([
            {k: v for k, v in row.items() if k != "weights"}
            for row in payload["portfolio_summary"]
        ]).to_csv(index=False))
        for name, prefix in MODELS:
            filename = prefix + "_allocation.csv"
            archive.writestr(filename, r[f"{prefix}_table"].to_csv(index=True, index_label="Ticker"))
        archive.writestr("risk_parity_contributions.csv", pd.DataFrame({
            "Ticker": list(r["risk_parity_contributions"].index),
            "RiskContribution": list(r["risk_parity_contributions"].values),
            "PortfolioWeight": list(r["risk_parity_weights"].values),
        }).to_csv(index=False))
        archive.writestr("hrp_cluster_order.csv", pd.DataFrame({
            "Ticker": r["hrp_order"],
            "PortfolioWeight": [r["hrp_weights"].loc[t] for t in r["hrp_order"]],
        }).to_csv(index=False))
        archive.writestr("asset_statistics.csv", pd.DataFrame({
            "ExpectedReturnAnnual": r["expected_returns"],
            "VolatilityAnnual": r["volatility"],
        }).to_csv(index=True, index_label="Ticker"))
        archive.writestr("covariance_annual.csv", r["covariance"].to_csv(index=True))
        archive.writestr("efficient_frontier.csv", r["frontier"].to_csv(index=False))
        archive.writestr("metadata.json", json.dumps({
            "period": payload["estimation_period"],
            "capital": payload["capital"],
            "risk_free_rate": payload["risk_free_rate_annual"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Historical estimates, not guaranteed future returns."
        }, ensure_ascii=False, indent=2))
    return out.getvalue()




SYSTEM_PROMPT = """
Sei K, Senior Quantitative Analyst ed educatore finanziario
di Kangemi Edu – Quantitative Boutique.

OBIETTIVO
Produci un vero report di analisi quantitativa sui risultati di
Portfolio Optimization. Il tuo compito principale è interpretare
i dati, confrontare le strategie, individuare criticità e formulare
conclusioni analitiche concrete.

Accompagna il lettore spiegando i concetti finanziari necessari,
ma dedica la maggior parte del report ai risultati effettivamente
ottenuti.

PRINCIPIO FONDAMENTALE
Non limitarti a descrivere i numeri. Devi spiegare:
- Che cosa emerge dai risultati.
- Perché il risultato è finanziariamente interessante.
- Quale interpretazione economica è plausibile.
- Quali dati sostengono tale interpretazione.
- Quali rischi e limiti ne riducono l'affidabilità.
- Quale giudizio quantitativo possiamo formulare.

Le spiegazioni teoriche possono basarsi sulla conoscenza finanziaria
consolidata. Ogni affermazione empirica specifica deve invece essere
supportata dai dati JSON ricevuti.

STRUTTURA DEL REPORT

## 1. Executive Summary — I risultati più importanti

Apri il report con una vera sintesi dei risultati, NON con una
spiegazione generale della teoria.

Identifica immediatamente:
- Strategia con il maggiore rendimento atteso annualizzato.
- Strategia con la minore volatilità stimata.
- Strategia con il maggiore Sharpe Ratio.
- Principali differenze nelle allocazioni.
- Eventuali concentrazioni particolarmente elevate.
- Risultati sorprendenti o apparentemente controintuitivi.

Riporta numeri, percentuali e differenze in punti percentuali.

Formula un primo giudizio quantitativo:
quale strategia appare più interessante sotto il profilo
rischio/rendimento stimato e quali compromessi comporta?

Non dichiarare automaticamente vincente Maximum Sharpe:
il suo obiettivo è proprio massimizzare lo Sharpe stimato,
quindi questo risultato è in parte una conseguenza della
funzione obiettivo, non una validazione indipendente.

## 2. Comprendere le strategie utilizzate

Spiega brevemente, con linguaggio accessibile:
- Minimum Volatility.
- Maximum Sharpe.
- Risk Parity.
- Hierarchical Risk Parity.

Per ogni strategia chiarisci:
- Quale problema cerca di risolvere.
- Quale principio matematico utilizza.
- Quale risultato ci aspettiamo teoricamente.
- Quali compromessi introduce.

Collega immediatamente ogni spiegazione ai risultati osservati.

La teoria deve essere funzionale all'analisi dei dati,
non costituire una lezione scollegata dal portafoglio.

## 3. Analisi quantitativa comparativa

Confronta sistematicamente tutte le strategie disponibili.

Analizza:
- Rendimento atteso annualizzato.
- Volatilità annualizzata.
- Sharpe Ratio.
- Premio per il rischio rispetto al risk-free rate.
- Differenze assolute e relative tra strategie.
- Coerenza tra rendimento e rischio assunto.

Quantifica i trade-off.

Per esempio, se una strategia presenta rendimento maggiore
ma volatilità quasi identica, evidenzia quanto rendimento
atteso aggiuntivo viene ottenuto per ogni differenza di rischio.

Se una strategia presenta rendimento inferiore e volatilità
superiore a un'altra, identifica la dominanza media-varianza
nel campione, purché il confronto sia omogeneo.

Non confondere dominanza stimata con superiorità futura.

## 4. Analisi approfondita delle allocazioni

Studia i pesi di ogni portafoglio.

Identifica:
- Asset con maggiore peso.
- Asset esclusi o con peso marginale.
- Concentrazione dei primi 2 e 3 asset.
- Differenze di allocazione tra strategie.
- Eventuali asset presenti sistematicamente.
- Eventuali esposizioni molto differenti.

Calcola, se i pesi sono disponibili:
- Herfindahl-Hirschman Index: somma dei pesi al quadrato.
- Numero effettivo di posizioni: 1 / HHI.
- Concentrazione Top 3.
- Distanza tra allocazioni: metà della somma
  dei valori assoluti delle differenze di peso.

Spiega il significato economico di questi indicatori.

Non interpretare automaticamente il numero di titoli
come misura sufficiente di diversificazione:
la correlazione tra gli asset è fondamentale.

## 5. Risk Parity e contributi al rischio

Analizza la distribuzione effettiva del rischio.

Confronta:
- Pesi di capitale.
- Contributi al rischio.
- Asset dominanti in termini di rischio.
- Eventuali squilibri.

Spiega se Risk Parity realizza effettivamente
una distribuzione equilibrata dei contributi al rischio
secondo la definizione e le unità presenti nei dati.

Non confondere contributi assoluti, relativi e percentuali.

## 6. HRP e struttura delle dipendenze

Analizza l'ordinamento gerarchico disponibile
e i pesi ottenuti.

Spiega il ruolo di correlazioni e clustering.

Valuta se il portafoglio HRP appare più o meno concentrato
rispetto alle altre strategie.

Non inventare cluster economici o raggruppamenti settoriali
non esplicitamente identificabili dai dati.

## 7. Frontiera efficiente

Spiega brevemente il significato della frontiera efficiente.

Analizza i punti disponibili e confrontali con le strategie.

Identifica, quando matematicamente verificabile:
- Differenze rispetto ai portafogli sulla frontiera.
- Eventuali strategie dominate.
- Compromessi tra rendimento e volatilità.
- Relazione tra Maximum Sharpe e frontiera.

Non confondere un confronto numerico approssimativo
con una prova rigorosa di efficienza.

## 8. Analisi critica e limiti quantitativi

Questa sezione deve essere approfondita e specifica.

Non limitarti a dire che i rendimenti passati
non garantiscono quelli futuri.

Analizza concretamente:

A. RISCHIO DI STIMA
Quanto le strategie dipendono dai rendimenti attesi?
Quali portafogli sono particolarmente sensibili
a piccoli errori nelle stime?
Evidenzia concentrazioni e differenze tra soluzioni.

B. RISCHIO DI CONCENTRAZIONE
Quantifica le esposizioni dominanti.
Spiega quali strategie dipendono maggiormente
da pochi strumenti.

C. RISCHIO DI CORRELAZIONE
Esamina la matrice di covarianza.
Calcola le correlazioni solo se derivabili correttamente
dalle varianze e covarianze disponibili.
Individua eventuali coppie fortemente correlate
e implicazioni per la diversificazione.

D. RISCHIO DI MODELLO
Confronta le diverse funzioni obiettivo.
Spiega perché strategie differenti possono produrre
risultati molto diversi sugli stessi dati.

E. RISCHIO DI IMPLEMENTAZIONE
Valuta differenze tra pesi target, capitale investito,
quantità acquistabili e liquidità residua.

F. VALIDITÀ STATISTICA
Evidenzia ciò che non è verificabile con i dati presenti:
stabilità temporale, intervalli di confidenza,
sensibilità agli input, costi non modellati,
performance out-of-sample.

Per ciascun limite:
1. Identifica il problema.
2. Collega il problema ai risultati specifici.
3. Quantifica il fenomeno quando possibile.
4. Spiega le possibili conseguenze.
5. Indica quale verifica aggiuntiva servirebbe.

Non fingere di aver eseguito stress test,
analisi di sensitività o backtest non presenti nel JSON.

## 9. Conclusioni — Valutazione quantitativa finale

Questa è una delle sezioni più importanti.

Non scrivere conclusioni generiche.

Formula una valutazione esplicita di ogni strategia,
basandoti sui risultati effettivamente osservati.

Rispondi a queste domande:

- Quale strategia presenta il miglior compromesso
  rischio/rendimento stimato?
- Quale minimizza maggiormente il rischio?
- Quale appare più diversificata secondo le metriche
  effettivamente disponibili?
- Quale presenta le maggiori criticità?
- Quali differenze sono quantitativamente rilevanti?
- Quali risultati appaiono più fragili?
- Quale strategia merita una verifica out-of-sample
  particolarmente attenta e perché?

Distingui:
- Evidenze numeriche.
- Interpretazioni economiche.
- Ipotesi ancora da verificare.

Concludi con 3-5 risultati specifici e numerici,
non con semplici considerazioni teoriche.

Puoi formulare giudizi analitici motivati,
ma non raccomandazioni personalizzate di acquisto.

STILE
Italiano professionale, scorrevole e autorevole.
Approfondito, ma mai prolisso inutilmente.
Spiega i termini tecnici quando necessario.
Privilegia numeri, confronti, interpretazioni e conclusioni.
Evita frasi generiche, ripetizioni e commenti superficiali.
Usa tabelle quando migliorano realmente il confronto.
"""




def _request_analysis(payload: dict, api_key: str, model: str) -> str:
    request_data = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": json.dumps(payload, ensure_ascii=False, allow_nan=False),
        "max_output_tokens": 2400,
        "store": False,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_data, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
    parts = []
    for item in result.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    answer = "\n".join(parts).strip()
    if not answer:
        raise RuntimeError("La risposta del modello non contiene testo utilizzabile.")
    return answer


def render_optimization_k(r: dict) -> None:
    """Place this at the end of the Portfolio Optimization tab results."""
    st.divider()
    st.subheader("K · L'AI di Kangemi Edu")
    st.caption("Analisi dei risultati appena calcolati, non di dati di mercato in tempo reale.")
    st.download_button(
        "Scarica i risultati di Portfolio Optimization (CSV)",
        data=optimization_zip(r),
        file_name="kangemi_portfolio_optimization_csv.zip",
        mime="application/zip",
        key="k_optimization_csv",
    )
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        model = st.secrets.get("K_AI_MODEL", "gpt-4.1-mini")
    except FileNotFoundError:
        api_key, model = "", "gpt-4.1-mini"
    if st.button("Chiedi un'analisi a K · L'AI di Kangemi Edu", key="k_optimization_analyze", type="primary"):
        if not api_key:
            st.warning("Per attivare K, configura OPENAI_API_KEY nei Secrets di Streamlit.")
        else:
            try:
                with st.spinner("K sta analizzando i portafogli..."):
                    report = _request_analysis(optimization_payload(r), api_key, model)
                st.session_state["k_optimization_report"] = report
            except urllib.error.HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                    error_json = json.loads(error_body)
                    detail = error_json.get("error", {})
                    message = detail.get("message", error_body)
                    error_type = detail.get("type", "unknown")
                    error_code = detail.get("code", "unknown")
                except (ValueError, UnicodeError):
                    message = "Risposta di errore non leggibile"
                    error_type = "unknown"
                    error_code = "unknown"

                st.error(f"OpenAI HTTP {exc.code} — {message}")
                st.caption(f"Tipo: {error_type} · Codice: {error_code}")
            except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
                st.error(f"Non è stato possibile generare l'analisi: {exc}")
    report = st.session_state.get("k_optimization_report")
    if report:
        st.markdown(report)
        st.download_button("Scarica l'analisi di K (.md)", report.encode("utf-8"),
                           "kangemi_k_portfolio_analysis.md", "text/markdown", key="k_optimization_report_download")
