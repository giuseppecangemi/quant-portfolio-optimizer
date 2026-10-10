# Saved Experiments: transaction costs

Replace only `app.py` in the project with the one in this archive. This patch is based on the previously delivered `quant_transaction_costs_backtesting.zip` version.

After running the backtests, use the existing **Save Experiment** sidebar button. Open the **Saved Experiments** tab to see the Gross vs Net table, saved cost parameters and (when available) per-rebalance order-cost logs for each standard fixed/dynamic and fixed-horizon backtest.

The full backtest results and curves were already saved in experiment snapshots; this patch adds an explicit transaction-cost snapshot and displays it in the saved experiment view. Existing experiments remain readable, though older saves may not contain the trade logs or explicit cost assumptions. No changes to screening, optimization, pricing or backtest execution.

On Streamlit Community Cloud, files saved only to local `saved_experiments/` may not survive app restarts or redeploys. Durable storage needs a persistent backing store.
