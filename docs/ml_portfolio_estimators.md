# ML portfolio estimators

The research workspace defines four input combinations: historical mean and sample
covariance, ML returns and sample covariance, historical mean and ML covariance,
or ML returns and ML covariance. All four pass annualized inputs to the existing
optimizers, with the application's weight cap and risk-free rate.

The pooled return regressors use trailing price features and predict cumulative
returns over the selected session horizon. Training includes labels whose end date
is at or before the forecast origin. Horizon forecasts are converted to an
equivalent daily rate and multiplied by 252 for optimization. A fixed chronological
development/validation/test split is evaluated with rolling fits. Models have fixed
hyperparameters. The training mean of the same horizon is the forecast baseline.

The ML covariance estimator uses a pooled random forest to predict the next daily
squared return from trailing second moments. It combines these variance predictions
with Ledoit–Wolf correlations and checks positive semidefiniteness. Covariance
validation uses the subsequent 21-day realized second moment as a noisy proxy.
This forecasts marginal risk and uses shrinkage for dependence.

Backtesting registers configurations for the current market in the Streamlit
session. Fixed Horizon uses the existing buy-and-hold engine at the first decision.
Walk-Forward calls the estimator on the existing pre-execution slice at each
rebalance. The selected cost model is handled by each existing engine.

Price inputs must have complete positive observations on their supplied common
session index. The research panel uses the selected current universe, which can
introduce survivorship and selection bias. Historical validation does not establish
predictive power on live data. Overlapping forward labels create dependent errors.
