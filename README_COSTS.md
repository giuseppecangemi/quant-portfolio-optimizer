# Transaction costs in Backtesting

Replace `app.py`, `src/backtest.py`, `src/robustness.py` and `src/transaction_costs.py` in your project. Keep your other `src` modules unchanged.

The Backtesting tab now has an `Execution cost scenario` control for Standard Walk-Forward (Fixed/Dynamic selection) and Fixed Horizon. The selected costs are deducted from available capital at every executed rebalance (and initial entry). All eight strategies share the same cost assumptions. A gross/net summary appears above the existing backtest results. Select `None (legacy / gross)` to recover previous no-cost behavior.

The Robustness Analysis retains its previous independent cost-scenario implementation; this patch does not rework its methodology. No cost is applied to the FTSE MIB reference series. Fixed Horizon charges only the initial entry because it is buy-and-hold.

Presets are configurable in `src/transaction_costs.py`. Base: 0.05% of order notional, minimum €3 per order; 5 bps half-spread plus 5 bps slippage. Commission minimum and proportional rate are applied with `max` for each trade. Conservative/Optimistic are sensitivity scenarios, not historical security-level execution quotes. No Italian financial transaction tax, capital gains tax, FX or delisting liquidation costs are modeled.

Important limitations: fills are modeled at the adjusted closing price plus explicit costs, with fractional shares. No live executable bid/ask quotes, order-book market impact, or broker integration. The gross counterfactual uses the same selected target weights as the net run; it is not a separate optimization run.

Sanity check performed: syntactic compilation of all modified Python files and synthetic 60-day equal-weight walk-forward test: zero-cost first value 10000, Base first value ~9984.02, net final value below gross final value. Full Streamlit/cloud execution and real-data factor tests not performed.
