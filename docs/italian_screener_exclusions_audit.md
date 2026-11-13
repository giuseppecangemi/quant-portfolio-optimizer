# Audit delle esclusioni italiane — 13 novembre 2026

Fonte: export delle esclusioni dello screener. 28 righe riportano mancato recupero prezzi, due riportano una sola osservazione. La ragione esportata è l'esito del ramo software, non un log HTTP o una prova di revoca. Il probe Yahoo ECNL.MI ha restituito DNSError nell'ambiente di esecuzione.

Fonti per i riscontri di mercato: [archivio revoche Borsa Italiana](https://www.borsaitaliana.it/borsaitaliana/statistiche/statistiche-storiche/revoche.en.pdf), [Piovan](https://www.piovan.com/wp-content/uploads/Press-release-issued-at-the-request-of-Automation-Systems-S.p.A.-Delisting.pdf), [Datalogic](https://www.emarketstorage.it/sites/default/files/comunicati/2026-09/20260909_188898.pdf), [Tinexta](https://www.emarketstorage.it/sites/default/files/comunicati/2026-09/20260909_188916.pdf), [listino regolamentato](https://live.euronext.com/it/markets/milan/equities/euronext/list?page=0), [STAR](https://live.euronext.com/nl/markets/milan/equities/star/list), [Moltiply](https://www.borsaitaliana.it/borsa/azioni/scheda/IT0004195308-MTAA.html?lang=it), [Tessellis e Trevi](https://live.euronext.com/it/markets/milan/equities/list?page=21), [Landi Renzo](https://www.borsaitaliana.it/borsa/azioni/listino-ufficiale.html?isin=IT0005619843).

| Instrument | Original ticker | Listing status | Data availability | Root cause | Recommended action | Final status |
|---|---|---|---|---|---|---|
| AS Roma | ASR.MI | Revocato | No price data after individual retry | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Alkemy | ALK.MI | Da verificare | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| Almawave | AWE.MI | Da verificare | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| Aquafil | AQF.MI | Quotato come ECNL | No price data after individual retry | Simbolo nel CSV diverso dal codice di mercato | Verificare identità e copertura del simbolo Yahoo prima di rimappare | CODE DEFECT |
| Banca Popolare di Sondrio | BPSO.MI | Da verificare | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| Be Think Solve Execute | BE.MI | Da verificare | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| Bialetti Industrie | BIA.MI | Da verificare | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| Carraro | CARR.MI | Da verificare | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| Civitanavi Systems | CVNS.MI | Revocato | No price data after individual retry | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Datalogic | DAL.MI | Revocato | Insufficient price history (1 observations) | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Digital Value | DGV.MI | Da verificare | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| Exprivia | XPR.MI | Revocato | No price data after individual retry | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Gefran | GEF.MI | Quotato come GE | No price data after individual retry | Simbolo nel CSV diverso dal codice di mercato | Verificare identità e copertura del simbolo Yahoo prima di rimappare | CODE DEFECT |
| Giglio Group | GG.MI | Da verificare | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| Gruppo MutuiOnline | MUT.MI | Quotato come MOL | No price data after individual retry | Simbolo nel CSV diverso dal codice di mercato | Verificare identità e copertura del simbolo Yahoo prima di rimappare | CODE DEFECT |
| Landi Renzo | LR.MI | Quotato | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| Monrif | MON.MI | Da verificare | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| Openjobmetis | OJM.MI | Revocato | No price data after individual retry | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Piovan | PVN.MI | Revocato | No price data after individual retry | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Saes Getters | SG.MI | Revocato | No price data after individual retry | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Salcef Group | SCF.MI | Revocato | No price data after individual retry | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Seco | SECO.MI | Quotato come IOT | No price data after individual retry | Simbolo nel CSV diverso dal codice di mercato | Verificare identità e copertura del simbolo Yahoo prima di rimappare | CODE DEFECT |
| Servizi Italia | SRI.MI | Revocato | No price data after individual retry | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Sogefi | SO.MI | Quotato come SGF | No price data after individual retry | Simbolo nel CSV diverso dal codice di mercato | Verificare identità e copertura del simbolo Yahoo prima di rimappare | CODE DEFECT |
| Tinexta | TNXT.MI | Revocato | Insufficient price history (1 observations) | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Tiscali | TIS.MI | Quotato come TSL | No price data after individual retry | Simbolo nel CSV diverso dal codice di mercato | Verificare identità e copertura del simbolo Yahoo prima di rimappare | CODE DEFECT |
| Tod's | TOD.MI | Revocato | No price data after individual retry | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |
| Trevi Finanziaria Industriale | TFI.MI | Quotato come TFIN | No price data after individual retry | Simbolo nel CSV diverso dal codice di mercato | Verificare identità e copertura del simbolo Yahoo prima di rimappare | CODE DEFECT |
| Unieuro | UNIR.MI | Da verificare | No price data after individual retry | Non determinata | Verificare avvisi per ISIN e richiesta Yahoo | UNRESOLVED |
| UnipolSai | US.MI | Revocato | No price data after individual retry | Esclusione corrente giustificata dalla revoca | Conservare identità storica | CORRECTLY EXCLUDED |

## Risultati e limiti

12 esclusioni giustificate dalla revoca, 7 discrepanze di codice, 11 casi irrisolti, 0 recuperi e 0 guasti Yahoo attribuibili al run originale. Una sola osservazione nei due casi non dimostra uno storico genuinamente corto. Per le righe senza prezzi, il messaggio indica che il ramo di retry ha restituito una risposta vuota o non estraibile. Il CSV non conserva righe grezze, errori provider, periodo richiesto o timestamp delle richieste. Non è possibile determinare a posteriori la fase esatta del guasto.

## Percorso e correzione

`app_core.py` carica `data/instruments.csv` e passa il mercato selezionato a `screen_universe`. `src/screening.py` pulisce il catalogo, scarica per batch Yahoo con period relativo, prezzi adjusted e thread disabilitati, estrae Close nei due orientamenti MultiIndex e ritenta individualmente i ticker insufficienti. Lo scoring usa almeno 80 prezzi, momentum 12–1 da 252 osservazioni, 6–1 da 127, e storico disponibile sotto 127. Risk usa deviazione standard annualizzata e drawdown. Value usa earnings yield e book-to-market, Quality ROE e leva. Le lacune fondamentali sono gestite nella fase di scoring, distinta dalle esclusioni per prezzo.

Corretto il conteggio che accettava duplicati, prezzi infiniti e non positivi. Il retry usa history con errori espliciti e mostra richiesta, stato del batch, esito individuale, conteggio e date quando disponibili. I test mock verificano pulizia, fallback, estrazione MultiIndex e tre esiti di retry. Lo stesso catalogo alimenta i backtest, quindi le sostituzioni richiedono verifica Yahoo e mapping per identità e data. Una revoca corrente non deve cancellare un titolo da un universo storico.

Prima: 30 esclusi. Dopo: NOT TESTED su Yahoo. Senza un rerun completo e la configurazione originale, non è possibile confrontare il Top N o dichiarare recuperi.

## Verifica live successiva

Con periodo 1y, Momentum + Risk e Top N 10, due esecuzioni Yahoo sullo
stesso universo di 163 titoli hanno prodotto 30 esclusi prima e 23 dopo.
I sette simboli corretti sono ECNL.MI, GE.MI, MOL.MI, IOT.MI, SGF.MI,
TFIN.MI e TSL.MI. Tutti hanno 254 osservazioni valide e stato Candidate
(RECOVERED). Il Top 10 è identico anche nell'ordine: CE.MI, G.MI,
ANIM.MI, CED.MI, ENI.MI, ENV.MI, OVS.MI, UNI.MI, IVG.MI, PIRC.MI.
Le 12 revoche documentate e gli 11 casi irrisolti restano distinti.
Le rimappature in app_core.py sono applicate allo screening corrente,
alla selezione quant risultante e alla sua diagnostica. Il catalogo
storico in data/instruments.csv rimane disponibile alle analisi storiche.

## Riesame degli 11 casi residui

| Ticker | Esito per lo screening corrente | Fonte |
|---|---|---|
| ALK.MI | CORRECTLY EXCLUDED · revoca dopo OPA Retex | [Alkemy](https://alkemy.imgix.net/wp-content/uploads/2025/07/14101236/Retex-Comunicato-esecuzione-procedura-congiunta-e-revoca-dalla-quotazione_ALK.pdf) |
| AWE.MI | CORRECTLY EXCLUDED · revoca dopo OPA Almaviva | [Almaviva](https://www.almaviva.it/dam/pdf/investor_relations/Comunicato-diffuso-da-Almawave-S.p.A.-su-richiesta-e-per-conto-di-Almaviva-S.p.A--Delisting-delle-azioni-Almawave-S.p.A.pdf) |
| BPSO.MI | CORRECTLY EXCLUDED · incorporazione in BPER | [BPER](https://group.bper.it/it/governance/operazioni-con-banca-popolare-di-sondrio) |
| BE.MI | CORRECTLY EXCLUDED · revoca dopo OPA Overlord | [Emittente](https://www.be-tse.it/wp-content/uploads/2022/12/PR%20Delisting.pdf) |
| BIA.MI | CORRECTLY EXCLUDED · revoca dopo OPA Octagon | [Comunicato CONSOB](https://www.consob.it/documents/11973/9797550/opa_bialetti_20250707_cs_20250728.pdf/759ac40a-a926-6154-af02-95a8e3052c96) |
| CARR.MI | CORRECTLY EXCLUDED · revoca dal 6 agosto 2021 | [Carraro](https://www.carraro.com/en/financial-data/investor-relations-before-delisting/files/investor-relations) |
| DGV.MI | CORRECTLY EXCLUDED · revoca dopo OPA Danzig BidCo | [Digital Value](https://www.digitalvalue.it/wp-content/uploads/2026/05/OEP-Danzig-BidCo-S.p.A.-Comunicato-esecuzione-della-Procedura-e-delisting.pdf) |
| GG.MI | CORRECTLY EXCLUDED alla verifica · ex Giglio Group ora Ops Ecom, codice OEC, ISIN IT0005122400. Fase SUS. Rivalutare alla riapertura e solo dopo verifica dello storico Yahoo. | [Borsa Italiana](https://www.borsaitaliana.it/borsa/azioni/dati-completi.html?isin=IT0005122400) |
| LR.MI | UNRESOLVED per Yahoo · Landi Renzo è quotata, ISIN IT0005619843. Il run live precedente non ha ottenuto prezzi con il simbolo originale. Nessun alias Yahoo verificato con almeno 80 osservazioni valide. | [Borsa Italiana](https://www.borsaitaliana.it/borsa/azioni/listino-ufficiale.html?isin=IT0005619843&lang=it) |
| MON.MI | CORRECTLY EXCLUDED · revoca dall’8 maggio 2025 | [Comunicato Monrif](https://www.emarketstorage.it/sites/default/files/comunicati/2025-05/20250509_166670.pdf) |
| UNIR.MI | CORRECTLY EXCLUDED · revoca dopo offerta Fnac Darty/Ruby | [Euronext](https://live.euronext.com/en/products/equities/company-news/2025-01-08-settlement-joint-procedure-and-delisting-unieuro-shares) |

Questo riesame aggiorna le classificazioni precedenti. Dei 30 casi iniziali:
7 recuperati, 21 cessati come titoli autonomi, 1 sospeso e 1 quotato con
copertura Yahoo irrisolta. La consultazione delle fonti di mercato non
ricostruisce la risposta Yahoo del run originale. Nessuna modifica applicativa.
