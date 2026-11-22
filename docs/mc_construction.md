# Monte Carlo Portfolio Construction

Source: Anton Vorobets, *Portfolio Construction and Risk Management*, 22 October
2026. [Book and edition](https://antonvorobets.substack.com/p/pcrm-book).
[Official examples](https://github.com/fortitudo-tech/pcrm-book), reference
commit `6e54462b3d10398dfc77431fb6ef9aa59e277154`. Application code implements
the published equations independently. The reference code is GPL-3.0-or-later.
No reference Python source or example datasets are shipped with the application.

## Formula map

| Source | Implementation |
|---|---|
| §3.1, Appendix C.1.1 | `equity_logs`: daily adjusted-price log differences |
| §3.2.1, eqs. (3.2.1)–(3.2.2), §5 | `entropy_posterior`, `calibrate_ffr`: exponential prior and KL projection with state-bin mean equality and second-moment upper bound |
| §3.2.1.1, §5.4 | Cartesian states, linear pools using entropy effective sample size and Hellinger diversity |
| §3.2.1, §3.3 | `draw_paths`: resample entire historical rows via the current posterior, then move to the sampled row's state |
| §3.4 | UI validation on historical and simulated log changes, including shared historical PCA basis and within-path autocorrelation |
| §1.1 and §§6.1–6.2.1 | `scenario_allocation`: sparse linear programs for absolute terminal CVaR, CML and maximum terminal mean under a CVaR budget |

For path s, asset i and step h, cumulative simple return is
`expm1(sum(log_return[s,i,1:h]))`. Tensor order is paths × assets × horizon.
The fixed-share equity portfolio earns the weighted sum of asset cumulative
simple returns, in units of initial NAV. Daily simple returns for comparison
with existing annualized ex-ante cards are recovered from adjacent gross values.
Horizon analytics remain in horizon units. The horizon Sharpe converts the
annual risk-free rate to the same horizon by compounding.

Loss is negative terminal P&L for CVaR. CML uses maximum cumulative loss
relative to initial NAV over the path, including zero. Maximum drawdown uses
the running peak and is reported separately. The linear programs use the
threshold plus average excess loss divided by the tail probability. CVaR
constrained maximizes mean terminal return at a specified risk budget.
The loss tail takes partial mass at the boundary and independently verifies the
solver's objective. The cash-instrument constraints are long only, sum to one,
and have the common cap. Costs are charged by the existing execution engines.

## Data adaptations and extensions

The operational universe is adjusted-price equities on the selected market.
The state observables are trailing 21-session volatility and momentum of the
cross-sectional mean daily log change. These are explicit equity adaptations.
There is no VIX fallback. No exchange calendar is synthesized and missing or
invalid cells in supplied price panels are rejected. Sessions absent from the
entire feed can still go undetected. Other risk-factor transformations,
risk budgeting, and Resampled Portfolio Stacking are not implemented.

Quantile cuts default to 25% and 75%, as in the single-state case study.
Other defaults are implementation choices: 504-session history, 252-session
prior half-life, 500 paths, 21-session horizon and seed 42. ADF (unit-root null)
and KPSS (level-stationarity null) are laboratory extensions. Ljung–Box tests
serial correlation, ARCH-LM tests conditional heteroskedasticity and omnibus
normality tests assess distribution shape. None proves stationarity.
Validation displayed in the lab is in-sample. It does not establish predictive
accuracy. The published tables and figures are not fully replicated.

## Backtesting and reproducibility

Register a strategy in Portfolio Analytics & Export and select it in Backtesting.
Only the configuration is transferred. Each engine calls the estimator on its
historically available slice. Fixed Horizon fits once; Walk-Forward refits at
each decision and uses realized prices afterwards. Seed and simulation horizon
are taken from the registered configuration. The training window follows the
backtest settings. Scenario horizon and rebalance interval can differ.
Results use the existing cost model, benchmark and strategy registry.
Point-in-time universes still depend on the application's selection settings.

Run `.venv/bin/python -m unittest discover -s _test -q` from the project root.
For the single-state reference comparison, use the Chapter 3 `st_df.csv` and
the Fortitudo 1.2.5 `time_series.csv`, extracting column `1m100` and dropping
its first row. Calibrate both implementations with 5,039 observations,
half-life T/2 and the 25%/75% cuts. The observed maximum absolute posterior
difference was 0.00002480; the implied transition difference was 0.002215.
The implementations use different dual scaling and convergence criteria.

## Real-data validation, November 2026

The two-year Italian equity walk-forward exposed a numerical stop in Entropy
Pooling on the decision preceding 12 May 2025. L-BFGS-B returned an abnormal
line-search status with standardized moment residuals of approximately
1.26e-9 and -1.40e-10. The exponential tilt is now accepted only after
independent checks of dual sign, normalized probabilities, moment feasibility
and second-moment complementarity (1e-6 for the moment and complementarity
tests). This certificate applies to every candidate, regardless of termination
status. The KL objective and state specification are unchanged.

The same experiment then completed all three FFR strategies over 505 aligned
out-of-sample valuations and eight rebalances, with the existing Base cost
model. The validation harness recorded all 24 strategy-specific training
slices and verified them against exactly the preceding 504 observations.
This run validates the execution path on the chosen sample. Statistical
adequacy is sample-dependent, including weaker volatility clustering in
the simulated paths and allocation sensitivity to simulations and states.

## Asset coverage

The lab downloads each equity independently using the existing Yahoo
download and retry logic. It aligns each close to observed sessions of the
selected benchmark, excluding the current local date. A missing benchmark
stops calibration with a specific message. Benchmark dates are an observed
calendar and cannot identify sessions absent from the full provider feed.

A calibration of N returns requires N+1 consecutive reference closes.
Strict is the default and preserves this length. It excludes incomplete,
duplicated, off-calendar or invalid series independently. Adaptive uses
the longest contiguous shared suffix of successfully downloaded and
calendar-aligned assets, subject to the configured minimum. The effective
period and lost sessions are displayed explicitly. Manual validates the
selected subset on the strict window. Blank, nonpositive and nonfinite
prices are not filled. The eligibility CSV lists requested-window coverage
and effective calibration dates by asset.

Registered settings carry the policy and manual selection into backtesting.
At each formation, the coverage rule is applied to the candidate universe
and dates passed by the existing backtest engine. No later prices are used
to determine historical coverage. Eligibility is distinct from historical
index membership and remains conditional on the engine's investable
universe, screening and execution checks. Adaptive is the only policy that
can shorten a historical estimation slice. Coverage rules can bias selection
toward seasoned listings and do not address survivorship or delisted stocks.
