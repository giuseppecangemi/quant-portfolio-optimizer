# Monte Carlo Simulation — audit quantitativo

## A. Executive Summary

**REQUIRES CORRECTIONS.** Capitalizzazione, campionamento congiunto, simulazione gaussiana e ripetibilità nello stesso ambiente sono supportati dai test sintetici. L'allineamento non garantisce intervalli omogenei, l'ES può includere una massa maggiore del 5% e la conversione NAV→rendimenti esclude il costo iniziale di ingresso. Questi problemi riguardano l'input e l'interpretazione del rischio, oltre alla presentazione.

Ambito: checkout locale su HEAD `fc9cef6`, incluso il contenuto effettivo dei file locali. Nessuna modifica al codice applicativo. Il report distingue l'evidenza sintetica dai controlli che richiedono dati storici riproducibili.

## B. Architettura

| Passaggio | Sorgente effettiva |
|---|---|
| Quotazioni | Yahoo Finance con `auto_adjust=True`, `src/data.py:33,43`. Le serie adjusted incorporano azioni societarie secondo il fornitore. |
| Formazione storica | Fixed Horizon in `app_core.py:2025–2279` e Standard Walk-Forward in `src/backtest.py:187–397`. Si usano stime storiche per formazione e ribilanciamento. |
| Esecuzione e NAV | Walk-forward usa quantità frazionarie e costi a ogni esecuzione, con quantità fisse tra ribilanciamenti e prezzi mancanti valorizzati tramite ultimo prezzo noto. `src/backtest.py:438–515`. Fixed Horizon mantiene le quantità dopo l'acquisto, `app_core.py:2254–2279`. |
| Sorgente Monte Carlo | `app_core.py:8234–8271` legge `standard_backtest_results` o `fixed_backtest_results`. Mappa Maximum Sharpe, CAPM, FF3, FF5, FF3/FF5 Factor Cov, Minimum Volatility, Risk Parity, HRP, Equal Weight ai rispettivi risultati. Nessun peso corrente o rendimento atteso viene passato al simulatore. Il benchmark è escluso. |
| Preparazione | `extract_equity` e `aligned_returns`, `src/monte_carlo.py:18–46`, estraggono Portfolio Value, ordinano le date, tengono l'ultima riga duplicata, escludono curve non valide, calcolano rendimenti semplici e intersecano le date finali. |
| Scenari e statistiche | `simulate`, `src/monte_carlo.py:49–133`. Output: tabella per strategia, percentili per seduta, terminali per scenario e traiettorie opzionali float32. |
| UI e confronto | `app_core.py:8274–8594`. Parametri, anteprima periodo comune, risultati in session state, ventaglio, istogramma e CSV. Robustness richiama lo stesso simulatore sui sottocampioni. |

Il NAV incorpora il drift dei pesi e i costi intermedi del walk-forward quando configurati. Si simulano rendimenti aggregati, senza modellare nuove operazioni o liquidità separata. Nei prezzi adjusted i dividendi non sono contabilizzati come pagamenti separati da Monte Carlo. La denominazione EUR viene ereditata dall'app. La conversione effettiva dei NAV USD in EUR non è stata validata, e il motore non riceve metadati di valuta. Il buffer EUR/USD dei fattori riguarda il modello di stima, non costituisce verifica della valuta del NAV.

## C. Metodologia

- **Moving block bootstrap** non circolare. Ogni inizio è uniforme con rimpiazzo tra 0 e n−L inclusi. L è convertito a intero e limitato a [1,n]. Il blocco finale viene troncato a T. Tutte le strategie condividono gli indici. Ai bordi, le osservazioni entrano in meno blocchi: la media marginale attesa non coincide in generale con la media empirica dichiarata. Il valore predefinito 5 è euristico, senza stima della lunghezza ottimale.
- **IID bootstrap** uniforme con rimpiazzo su righe intere. Conserva il supporto empirico e l'associazione contemporanea prima dello stress. Elimina la dipendenza temporale per costruzione.
- **Normale multivariata** con media giornaliera e covarianza campionaria n−1. Simmetrizza e tronca gli autovalori negativi a zero prima del campionamento con RNG locale. La regolarizzazione non è riportata. La correlazione è preservata in distribuzione, non esattamente in ogni campione. Il supporto normale ammette rendimenti sotto −100% e una sola estrazione del genere interrompe il run.
- **Robustness** usa intero periodo, prima metà [0,⌊n/2⌋), seconda metà [⌊n/2⌋,n), ultimo 75% [⌊n/4⌋,n). Con n=252 le dimensioni sono 252,126,126,189. Il filtro è max(40, minimo UI), la numerosità è 300/500/1000. Lo stesso campione alimenta i tre metodi e il seed è base+indice del metodo, riutilizzato fra sottocampioni. Finestre sovrapposte e seed riutilizzati non producono repliche indipendenti. Il confronto è una sensibilità storica, non una validazione walk-forward.

## D. Verifica delle formule

| Calcolo | Implementazione e riferimento indipendente | Esito |
|---|---|---|
| Rendimento giornaliero | NAV(t)/NAV(t−1)−1. Corretto su intervalli coincidenti. Il primo rendimento rispetto al capitale pre-ordini va ricostruito separatamente. | FAIL ai gap e ai costi iniziali |
| Orizzonte e capitale | T=252×anni. V(0)=C e V(T)=C∏(1+r). Nessun log-return. | PASS |
| Stress annuo | r′=r+δ/252, δ in decimali. Esatto come shift aritmetico annualizzato, approssimato per rendimento composto. Da zero, ±10 pp danno rispettivamente circa +10,515% e −9,518%. | PASS per la convenzione |
| Media/mediana | Somma dei terminali divisa per numero scenari e quantile 0,50. Rendimento cumulativo medio derivabile come media terminale/C−1. | PASS |
| P5/P95 e ventaglio | Quantili lineari cross-scenario dei valori a data fissa, anche P25/P50/P75 per i percorsi. Sono percentili condizionali, non intervalli di confidenza per parametri o bande simultanee. | PASS |
| Probabilità perdita | Conteggio V(T)<C diviso per N, visualizzato ×100. | PASS |
| VaR terminale 95% | max(0,−Q05(V(T)−C)), in euro. Misura di orizzonte, non VaR giornaliero. | PASS con la definizione dichiarata |
| ES terminale 95% | max(0,−media(P&L≤Q05)). Riferimento: perdita media sulla massa del 5%, con peso frazionario al bordo. | FAIL con legami o masse discrete |
| Probabilità drawdown ≤−20% | min_t[V(t)/max_{s≤t}V(s)−1], include capitale iniziale. Conta percorsi sotto soglia. | PASS |
| Altri KPI | Target, CAGR, volatilità annualizzata, Sharpe, skew e kurtosis non sono output Monte Carlo di produzione. Nessun risk-free applicato nel simulatore. | N/A |

## E. Bias e integrità temporale

- **Nessun problema identificato nei test di separazione**: Fixed Horizon taglia il training prima dell'esecuzione (`app_core.py:2053–2065`), Standard usa lo slice fino a rebalance_start escluso (`src/backtest.py:187–199`). Il test esistente sugli shock OOS verifica invariabilità dei KPI iniziali e coerenza dei NAV nelle fixture. Non è prova per tutte le configurazioni live.
- **Potenziale look-ahead nei dati vintage**: il cutoff Fama–French di due mesi è una regola conservativa (`src/factor_currency.py:8–10`, `src/backtest.py:302,325`, `app_core.py:2157–2159`). Non prova il timestamp di pubblicazione né l'assenza di revisioni. Yahoo adjusted è retrospettivo. La disponibilità effettiva dei dati fondamentali, dei fattori, dei cambi e il trattamento di delisting reali non sono stati verificati su archivi vintage.
- **Survivorship e selection bias potenziali**: un universo attuale applicato all'indietro può escludere uscite e includere entranti. La scelta manuale di fonte e strategie dopo aver visto il backtest può produrre data snooping. Il codice permette questa scelta, ma non consente di stabilire se un utente l'abbia praticata.
- **Disallineamento confermato**: allineare solo la data finale accoppia r(t−2,t) e r(t−1,t). Dopo il drop, blocchi consecutivi per posizione possono attraversare sedute mancanti. Il forward fill upstream evita prezzi futuri ma può appiattire volatilità e generare salti al ritorno della quotazione.
- **Interpretazione OOS**: i dati di calibrazione derivano dall'OOS del backtest. Le traiettorie riusano quel campione e non aggiungono un test indipendente di capacità predittiva. La UI parla di simulazioni condizionate e sconsiglia la lettura come previsione certa, ma non esplicita questo punto nel confronto.
- **ML**: nessuna chiave ML nel mapping della simulazione. Nessuna attestazione di leakage ML è applicabile al percorso ispezionato. Overfitting dei modelli e della scelta ex post rimane potenziale.

## F. Diagnostica e adeguatezza statistica

Fixture a 252 osservazioni con innovazioni Student-t e cambio di regime: skew −0,678, eccesso di kurtosis 3,327, autocorrelazione a lag 1 pari a 0,426 sui rendimenti e 0,220 sui quadrati. Volatilità giornaliera delle metà 0,87% e 2,51%. Questi numeri descrivono il test, non dati di mercato.

| Lunghezza blocco | Autocorr. lag 1 | Autocorr. quadrati | ES terminale € |
|---:|---:|---:|---:|
| 1 | −0,002 | −0,002 | 6815,50 |
| 3 | 0,286 | 0,149 | 7554,43 |
| 5 | 0,341 | 0,179 | 7624,68 |
| 10 | 0,375 | 0,194 | 7476,63 |
| 20 | 0,388 | 0,205 | 7425,34 |

I blocchi conservano parte della dipendenza. Non riproducono integralmente i regimi o le transizioni. L'IID e la gaussiana eliminano la dipendenza temporale. La gaussiana non riproduce asimmetria e code pesanti per costruzione. Con 252 rendimenti, il 5% corrisponde a circa 13 sedute e l'1% a circa 2–3, prima di considerare dipendenza. Aumentare il numero delle simulazioni riduce l'errore numerico, senza aggiungere informazioni sul mercato. Per esempio, con N=500 e p=5%, l'errore standard binomiale condizionale è circa 0,97 pp. Per differenze fra strategie usare la varianza delle differenze appaiate per scenario. Comparare finestre o metodi come se fossero repliche indipendenti sottostima l'incertezza.

## G. Test eseguiti

| Test | Riferimento ed evidenza | Esito |
|---|---|---|
| A–C | 252 sedute, C=10000, r=0/+0,001/−0,002. Formula scalare, t=0 incluso, MDD oltre 20% soltanto nel caso negativo. Tutti e tre i metodi. | PASS |
| D–F, I, L–M | Indici ricostruiti con loop scalari, lunghezza 252 con blocco finale parziale, serie identiche perfettamente coincidenti nei bootstrap, seconda serie con shock doppi. Seed identico ripete, seed diverso cambia. | PASS |
| G–H | NAV con NaN escluso, 39 osservazioni rifiutate. Il 2,01% a due sedute viene abbinato all'1,00% a una seduta. | FAIL allineamento riprodotto |
| J–N | V(T)=C(1+δ/252)^252 a stress zero e ±10 pp. | PASS |
| K | Quantili, P(loss), MDD, media, mediana, VaR ricalcolati indipendentemente. ES effettivo €9.590,847120 su 8/101 scenari contro €9.598,557123 per la massa esatta 5%. | FAIL ES riprodotto |
| O | Input singolare con due colonne identiche. Correlazione teorica 0,88791, simulata 0,88913. Confronto covarianza e terminali superato entro tolleranza. | PASS |
| Edge | Stress NaN restituisce 100 terminali NaN senza rifiuto. | FAIL validazione riprodotto |
| Sensibilità | Blocchi 1/3/5/10/20, 500 percorsi a seed 42. Tabella in F. | PASS comportamento atteso |
| Regressione repository | `python -m unittest discover -s _test -q` prima dell'aggiunta: 6 test unittest più controlli sintetici eseguiti all'import, tutti riusciti. | PASS |

La suite indipendente ha 8 metodi di test. I test che riproducono un difetto passano come test di caratterizzazione. Le tolleranze di 1e−12/1e−13 sono per calcoli deterministici float64. Covarianza gaussiana con tolleranza relativa 2,5% per campionamento e valori di percorso float32, medie con tolleranza assoluta 1,2e−4 e terminali di colonne identiche gaussiane entro 1e−7 relativo.

**Non eseguiti:** confronto su NAV live con provenienza riproducibile, diagnostica storica reale di code e correlazioni, stabilità empirica delle classifiche tra sottocampioni, verifica del rendering browser, stress test al limite di memoria e verifica del timestamp di pubblicazione storico. Le metriche aggregate o screenshot non ricostruiscono le serie giornaliere.

## H. Registro dei rilievi

| ID | Gravità e categoria | Fonte ed evidenza | Impatto | Correzione e test di verifica |
|---|---|---|---|---|
| MC-01 | High · allineamento | `src/monte_carlo.py:35–45`, rendimento su due intervalli scambiato per daily | Covarianze e blocchi temporalmente incoerenti | Calendario e intervalli comuni, segmentare gap. Fixture con seduta mancante |
| MC-02 | High · consistenza costi | `src/monte_carlo.py:18–41`, `app_core.py:2276–2278`, `src/backtest.py:551–567`. Il costo a t0 sta nel primo livello NAV e sparisce in pct_change | «Netto» incompleto per acquisto iniziale | Decidere se simulare portafoglio già investito o addebitare ingresso una volta separatamente. Fixture con NAV piatto e costo noto |
| MC-03 | Medium · statistica | `src/monte_carlo.py:112–122`, soglia con 8/101 osservazioni | Coda di massa errata rispetto alla descrizione | Integrare quantile empirico con massa frazionale. Fixture discreta |
| MC-04 | High · validazione | `src/monte_carlo.py:71–99`, drift NaN attraversa il motore. Capitale +inf, input con zero colonne, anni frazionari, blocchi non positivi e storia ≤−100% richiedono controlli espliciti | Output non finiti, eccezioni poco informative, shock invalidi non sempre estratti | Validare input e output prima di pubblicare risultati. Test parametrico |
| MC-05 | Medium · riproducibilità UI | `app_core.py:8350–8373,8540–8561`. Run salvato per etichetta e selezione. Main non segnala modifiche di tutti i parametri, entrambi ignorano il fingerprint del NAV | Risultati precedenti esposti con controlli o sorgente modificati | Salvare fingerprint, versione, date e parametri. Simulare sostituzione del backtest a nome invariato |
| MC-06 | Medium · bias del bootstrap | `src/monte_carlo.py:67–83`. Righe di bordo con peso inferiore | Media e code del campione di shock possono spostarsi | Dichiarare pesatura. Valutare variante circolare previa approvazione. Fixture con outlier all'estremo |
| MC-07 | Medium · dipendenze e memoria | `src/monte_carlo.py:82–105,132`. 3.000×1.260×10 produce 302 MB per singolo cubo float64, diversi cubi contemporanei | OOM e rerun pesanti | Preflight memoria e test equivalenza seeded se si procede a batching |
| MC-08 | Medium · rischio di modello | `src/monte_carlo.py:88–99`. Clipping silenzioso, normale non limitata | Approssimazione su code e fallimento scenario | Registrare condizionamento e correzione. Test PSD e rendimenti estremi |
| MC-09 | Medium · presentazione | `app_core.py:8229–8233,8484–8518`. Nessuna esplicita avvertenza sulle sovrapposizioni e su OOS di calibrazione | Interpretazione eccessiva della robustezza | Indicare natura condizionale e incertezza Monte Carlo. Test didascalie |
| MC-10 | Medium · provenienza | `src/monte_carlo.py:26–31`, `app_core.py:8268–8271`. Timestamp scartati, duplicate tenuta ultima, strategie invalide omesse senza motivi individuali | Copertura e qualità non tracciate | Rendere visibili esclusioni, valuta, calendario e date. Fixture duplicate/NaT |

## I. Piano correttivo

1. **Obbligatorio:** definire calendario e intervalli omogenei, esplicitare l'ingresso e rendere tracciabile il campione netto. Congelare una fixture integrata backtest→MC con gap, costi e più valute.
2. **Obbligatorio:** correggere ES discreto, rifiutare input non validi e impedire pubblicazione di output non finiti o obsoleti.
3. **Obbligatorio per l'interpretazione:** indicare carattere condizionale, sovrapposizione, orizzonte dei rischi, numerosità e provenienza dei dati.
4. **Estensioni da valutare:** selezione della lunghezza blocco, bootstrap circolare, modelli a code pesanti, incertezza appaiata e ottimizzazione memoria. Validare su campioni con identità e valuta documentate prima di ampliare le affermazioni di robustezza.

## J. Final Assessment

**REQUIRES CORRECTIONS.** La riproducibilità sintetica e il meccanismo di simulazione sono verificati entro le convenzioni testate. L'allineamento, la definizione di ES, l'ingresso e la validazione input impediscono una validazione complessiva. La validità predittiva e la disponibilità storica effettiva dei dati restano non validate.

## K. Remediation — verifica successiva all'audit

Le sezioni A–J documentano la baseline prima della correzione. Le conclusioni aggiornate sono qui.

- **MC-01:** `src/monte_carlo.py`, `aligned_returns` e `simulate`. Il calendario osservato è l'unione delle date di valorizzazione delle strategie selezionate. Prima si allineano i livelli, poi si calcola NAV(t)/NAV(t−1)−1 solo quando entrambe le estremità sono presenti per tutte le strategie. Nessuna imputazione. I blocchi attraversano soltanto posizioni consecutive nel calendario osservato, anche nei sottocampioni. Nella fixture con 45 date si ottengono 42 intervalli validi. L'abbinamento 1,00%/2,01% scompare. Le sedute mancanti simultaneamente da tutte le fonti non sono identificabili senza un calendario indipendente. I prezzi già imputati dal backtest restano parte dei NAV originali.
- **MC-02:** `extract_equity`, `aligned_returns` e `simulate`. La calibrazione usa variazioni nette del NAV dopo l'ingresso. Si applica una sola volta il rapporto k=NAV iniziale netto/capitale storico, per cui V(T)=C×k×∏(1+r). Il fan parte dal capitale pre-ingresso e il drawdown include anche la perdita immediata d'ingresso. Costi successivi già nel NAV e nessun nuovo addebito su quelle variazioni. Rapporto non disponibile: portafoglio già investito, k=1, con avviso UI. Con k=0,98 e 252 sedute a +0,1%, il terminale passa da circa €12.864,34 a €12.607,05 e coincide con il NAV storico. I sottocampioni riusano lo stesso costo d'ingresso della fonte, separato dalle estrazioni.
- **MC-03:** `empirical_tail_mean` e `simulate`. Ordinato il P&L crescente, per m=0,05N si pesano ⌊m⌋ osservazioni intere più la frazione residua della successiva. Si divide per m e si mantiene il segno di perdita e il limite inferiore zero. La fixture da 101 scenari passa da €9.590,847120 a €9.598,557123. Il VaR conserva il quantile lineare precedente.
- **MC-04:** `simulate` rifiuta capitale e stress non finiti, parametri interi fuori dominio, storia senza strategie, dati non finiti o ≤−100%, e scenari non rappresentabili. `app_core.py` mostra l'errore e rimuove il risultato precedente quando si lancia un nuovo run. La fixture con drift NaN ora restituisce un errore invece di 100 terminali NaN.

`_test/test_monte_carlo_audit.py` verifica i regressi e i casi a massa frazionaria. Restano aperti il calendario di mercato indipendente, le verifiche su NAV live e valute, i dati vintage, il fingerprint delle sorgenti già salvate e i limiti di memoria. Nessuna modifica alle metodologie di backtest e ottimizzazione.

**Classificazione aggiornata: NOT VALIDATED complessivamente.** Le quattro correzioni sono coperte da test sintetici. La validazione end-to-end su dati reali e calendario indipendente resta da eseguire.

## L. Validazione empirica — 13 novembre 2026

Fonti effettive: `~/Downloads/rolling_robustness_results.csv` e
`~/Downloads/rolling_robustness_diagnostics.csv`. Sono riepiloghi di un run
Rolling Robustness, non le curve Standard Walk-Forward o Fixed Horizon
lette dal tab Monte Carlo. Le directory locali `saved_experiments` e
`results/ff_comparison` non contenevano dati di backtest. Nessuna serie
giornaliera è stata dedotta da Sharpe, drawdown o rendimenti aggregati.

| Verifica | Metodo e risultato | Esito |
|---|---|---|
| Copertura dei riepiloghi | Conteggio per anno OOS e modello. 19 righe completate nelle sei finestre 2020–2025, tutte etichettate EUR. Benchmark 6, Equal Weight 6, CAPM 3 e una finestra per ciascuna delle quattro varianti FF. Il file diagnostico riporta 23 combinazioni escluse. | PASS limitatamente ai conteggi |
| Cronologia riportata | Nelle 13 righe di strategie con metadati, 13/13 rispettano fine stima < prima esecuzione ≤ ultima esecuzione. Le sei righe Benchmark non forniscono questi campi. Nella finestra 2025, fine stima 30/12/2024, esecuzione dal 02/01/2025 al 30/12/2025, tre ribilanciamenti riportati per ciascuna strategia. | PASS per l'ordine delle date aggregate |
| Costi aggregati | 0/19 righe con rendimento netto superiore al lordo. FF3 nel 2025 riporta 55 ordini, €189,59 di costi complessivi e rendimento lordo 56,6959% contro netto 54,2260%. Il divario di 2,4699 pp non basta a riconciliare il costo iniziale, i singoli addebiti e il loro effetto composto. | PASS per il controllo direzionale, NOT TESTED per la riconciliazione |
| Calendario indipendente | Identificati i calendari ufficiali di negoziazione Borsa Italiana [2025](https://www.borsaitaliana.it/borsaitaliana/calendario-e-orari-di-negoziazione/tradingcalendar-2025.en.pdf) e [2026](https://www.borsaitaliana.it/borsaitaliana/calendario-e-orari-di-negoziazione/tradingcalendar-2026.en.pdf). Per un run Milano si confrontano le valorizzazioni con le sedute diurne, tenendo distinte chiusure del solo After Hours, sedute prima dell'esecuzione o dopo l'uscita e sospensioni di singoli titoli. Sedute NAV osservabili nei due export: nessuna. | NOT TESTED per il confronto con NAV |
| Qualità NAV e riconciliazione OOS→MC | Servono date giornaliere, livelli NAV e capitale pre-ingresso. Duplicati, sedute mancanti comuni o individuali, NAV invalidi, intervalli cumulati, flussi e riconciliazione numerica non sono ricavabili dai riepiloghi. | NOT TESTED |
| Diagnostica statistica | Nessun rendimento giornaliero disponibile per media, volatilità, autocorrelazioni, skew, kurtosis, estremi, correlazioni, stabilità della covarianza e sensibilità alle finestre. L'adeguatezza empirica dei tre metodi resta indeterminata. | NOT TESTED |
| Simulazioni reali | Nessuna simulazione eseguita su questi aggregati. Riproducibilità su NAV reali, allineamento degli shock, capitale terminale, ES, drawdown, probabilità di perdita e stabilità dei ranghi restano da misurare. | NOT TESTED |
| Integrità informativa | L'ordine delle date aggregate non stabilisce disponibilità vintage, formazione dei pesi, esecuzione T+1 a ciascun segnale o disponibilità storica dell'universo. Un universo corrente applicato retrospettivamente e la scelta post hoc delle strategie restano fonti potenziali di bias. | NOT TESTED oltre la cronologia aggregata |

Limite da controllare appena disponibili i NAV: il Monte Carlo oggi usa l'unione
delle date osservate. Se tutte le strategie omettono una seduta aperta,
l'intervallo successivo può essere considerato giornaliero. Per riprodurre il
caso, eliminare la stessa data interna da due curve allineate e confrontare
i predecessori prodotti da `aligned_returns` con quelli del calendario
ufficiale. La correzione, se necessaria per il mercato del run, consiste nel
verificare gli estremi sul calendario indipendente prima del campionamento,
escludendo entrambi gli intervalli che toccano la seduta mancante.

Per completare il confronto serve il JSON di **Saved Experiments** salvato
dopo il backtest usato in Monte Carlo. Il salvataggio acquisisce i risultati
Standard e Fixed presenti in sessione (`app_core.py:276–277,338–341`).
In alternativa servono NAV giornalieri per strategia, mercato e valuta,
capitale iniziale e registri datati di costi, pesi e ribilanciamenti.
Con quei dati, confrontare prima la stessa finestra a seed fisso e con
più numerosità per isolare l'errore Monte Carlo, poi cambiare campione o
lunghezza del blocco per misurare la sensibilità di calibrazione.

**Classificazione empirica: NOT VALIDATED.** Il controllo sui riepiloghi
verifica la sequenza delle date dichiarate e il segno dell'impatto dei
costi. La calibrazione Monte Carlo e l'integrità dei NAV giornalieri
restano da verificare. Nessuna modifica alla logica finanziaria.

## M. Daily risk and audit export

Daily VaR uses the linear 5% quantile of all simulated daily simple shocks
for each strategy. Daily ES averages exactly 5% of that same distribution
with fractional boundary mass. Both measures report nonnegative losses
against the fixed initial notional. Entry costs affect terminal wealth once
and are absent from pooled shocks. Daily risk is unconditional and can be
positive even when the terminal fifth percentile exceeds initial capital.

The backtest keeps daily net NAV and Daily Return in its result frames, with
gross NAV when costs are enabled. Fixed Horizon builds these frames in
app_core.py near the assignment of Portfolio Value; Walk-Forward assembles
them in src/backtest.py before adding initial_capital and transaction_log.
Monte Carlo extracts Portfolio Value from standard_backtest_results and
fixed_backtest_results and recomputes the common daily intervals. Saved
Experiments already serializes the result frame columns, dates and weights
history. Older JSON drops pandas attrs, including starting capital and
per-curve trade logs. New snapshots retain those attrs and older files
remain readable. Missing attrs remain unknown.

The optional audit ZIP captures the existing session curves, exact
calibration returns, valuation dates, available trade/weight records,
entry multipliers and configuration at export time. A live run may lack
an experiment identifier. An observed date union is not an exchange calendar.

Independent synthetic tests cover pooled daily risk, exact tail, capital,
horizon, identical strategies, NAV/return CSV roundtrip, gross/net relationship
and an omitted date shared by all strategies. That test confirms the
remaining limitation: the next interval still enters calibration when all
curves miss the same open session. For a real run, compare the exported dates
with an independent market calendar before accepting the sample.

**NOT VALIDATED.** No Saved Experiment or daily real NAV was available in
the inspected files. The aggregate evidence in section L still stands.
Actual-date coverage, realized cost/cash-flow reconciliation, real-sample
diagnostics and method/seed/path-count/block-length sensitivity remain
NOT TESTED. Provide the audit ZIP from the source session or its Saved
Experiment JSON and the corresponding market calendar.

## N. Validazione empirica del run Standard · Fixed

Fonte: monte_carlo_audit.zip, SHA-256
`45bab740d16429cccfe92ced85e119555db97a1795b85bea288932d03e07ca2b`.
Il mercato non è valorizzato nei metadati, ma i ticker dei pesi sono .MI.
Il calendario di riferimento è quindi quello diurno di Milano
([Borsa Italiana 2025](https://www.borsaitaliana.it/borsaitaliana/calendario-e-orari-di-negoziazione/tradingcalendar-2025.en.pdf),
[Euronext 2026](https://live.euronext.com/en/resources/trading-hours-holidays)).
Il file dichiara un anno, 1.000 percorsi, capitale €10.000, blocco 5,
seed 42, stress zero.

| Verifica | Risultato | Esito |
|---|---|---|
| Date, qualità, calendario | 10 strategie, ciascuna 253 NAV dal 14/11/2025 al 13/11/2026 e 252 rendimenti di calibrazione. Zero duplicati, NAV non positivi o non finiti, sedute assenti, date fuori calendario o disallineamenti. | PASS sul campione |
| NAV e rendimenti | NAV(t)/NAV(t−1)−1 e primo NAV/capitale−1 riconciliati con Daily Return. Errore massimo 5,43e−16, anche rispetto alla calibrazione esportata. | PASS |
| Pesi e costi | Quattro date di esecuzione comuni: 14/11/2025, 18/02/2026, 21/05/2026, 18/08/2026. Pesi e log coincidono sulle date. Scarto massimo della somma pesi 2,78e−15. Costi totali per strategia €193,43–€615,42, identici alla colonna Transaction Cost per data. NAV di esecuzione riconciliato con traded notional / executed turnover meno costi entro €1,82e−11. | PASS |
| Prezzi, flussi e OOS | Mancano quotazioni e quantità per singolo ordine, registro indipendente dei flussi, timestamp effettivi di chiusura e finestre di stima per ribilanciamento. Dal solo export non si certificano lordo per titolo, assenza di flussi, disponibilità vintage, esecuzione T+1 o selezione storica dell'universo. | NOT TESTED |
| Monte Carlo | Ripetizione identica a seed 42. Ricostruiti gli indici dei blocchi congiunti e verificati tutti i 10.000 terminali (rtol 1e−13), ES di orizzonte e giornaliero, probabilità di perdita e probabilità MDD ≤ −20%. | PASS |

Statistiche giornaliere sulle 252 righe: media 0,052–0,139%, volatilità
0,557–1,047%, skew −0,710/+0,131, eccesso di curtosi 0,947–3,804,
autocorrelazione lag 1 da −0,109 a −0,019, autocorrelazione dei quadrati
0,083–0,211. Estremi fra tutte le strategie −4,085% e +3,384%.
Correlazione minima 0,564. Numero di condizionamento della covarianza 9.144.
La distanza di Frobenius fra le covarianze delle due metà vale il 35,0%
della norma della covarianza intera. Il bootstrap a blocchi conserva parte
della dipendenza nei quadrati. IID e gaussiana non la modellano e la
gaussiana non cattura l'eccesso di curtosi osservato. Con 252 osservazioni
la coda storica al 5% si fonda su circa 13 sedute.

Baseline: Daily VaR €77,74–€160,05, Daily ES €113,89–€222,50.
Per FF5 Factor Cov la VaR di orizzonte è €0, l'ES di orizzonte €218,39
e la Daily VaR €160,05, coerenti con differenti distribuzioni e orizzonti.

| Sensibilità rispetto alla baseline | Spearman mediana terminale | Max variazione mediana | Max variazione P(perdita) |
|---|---:|---:|---:|
| IID | 1,0000 | €62,11 | 3,10 pp |
| Gaussiana | 0,9879 | €67,73 | 3,00 pp |
| Seed 43 | 0,9879 | €226,86 | 1,70 pp |
| 5.000 percorsi | 0,9758 | €169,37 | 0,58 pp |
| Blocco 3 | 0,9879 | €93,26 | 1,70 pp |
| Blocco 10 | 0,9879 | €120,47 | 4,30 pp |
| Blocco 20 | 0,9636 | €416,38 | 10,40 pp |
| Prima metà | 0,9152 | €1.807,40 | 9,70 pp |
| Seconda metà | 0,9515 | €1.409,68 | 27,60 pp |
| Ultimo 75% | 0,9879 | €1.213,52 | 25,00 pp |

Le variazioni di seed e numerosità misurano variabilità numerica
condizionata al campione. Le finestre cambiano la calibrazione.
I ranghi qui sono diagnostici di stabilità, non evidenza di superiorità
o capacità predittiva. 10.000 percorsi e intervalli appaiati non eseguiti.

**Stato:** calcoli validati per questo export. Audit complessivo
**NOT VALIDATED** per la provenienza point-in-time degli input e il
limite già documentato del calendario osservato in presenza di gap comuni.
