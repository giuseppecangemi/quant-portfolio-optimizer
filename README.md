# Monte Carlo — Robustness & Method Comparison

## Installazione
Sostituisci soltanto `app.py` con quello contenuto nello ZIP. Mantieni il tuo attuale `src/monte_carlo.py` (deve essere la versione con supporto alle traiettorie, già installata per la sezione Monte Carlo).

## Funzionalità
Nella scheda Monte Carlo Simulation, sotto i risultati principali, compare **Monte Carlo — Robustness & Method Comparison**. Il pulsante dedicato esegue un confronto tra tre metodi (Block bootstrap, IID bootstrap, Multivariate Gaussian) su intero periodo, prima metà, seconda metà e ultimo 75% del campione, se sufficientemente lunghi. Confronta tutte le strategie selezionate, produce un grafico interattivo e consente l'esportazione CSV.

Il confronto usa lo stesso numero di scenari e gli stessi parametri economici in tutte le combinazioni; il seed è riproducibile. Le finestre sono in parte sovrapposte e non rappresentano campioni indipendenti. Un singolo anno di dati non permette di validare statisticamente eventi estremi o generalizzare le probabilità al futuro.

## Modifiche
Nessuna modifica ai modelli finanziari, al backtesting, ai calcoli di simulazione o ai colori sfumati. È stata aggiunta soltanto la nuova sezione di confronto a `app.py`.

## Verifiche
Sintassi Python verificata. Test sintetico: 12 combinazioni di campione/metodo, 8 strategie ciascuna. Non è stato eseguito il rendering end-to-end in Streamlit con i dati reali dell'utente.
