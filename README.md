# Quant Portfolio Optimizer

**An interactive Python and Streamlit framework for quantitative asset selection, portfolio construction, risk management, and strategy evaluation.**

Quant Portfolio Optimizer is a research-oriented application that brings together established methods from financial economics and quantitative investing. Its purpose is not to identify guaranteed winning stocks or predict market movements with certainty. Instead, it provides a transparent environment in which investors and researchers can examine **how different assumptions lead to different portfolios**, and how those portfolios behave under historical and simulated market conditions.

The project follows a complete analytical workflow: from selecting assets and estimating expected returns to constructing portfolios, evaluating risk, and testing the robustness of investment decisions.

## Why this project?

Portfolio construction involves more than choosing promising assets. Even with the same investment universe, different models can recommend very different allocations because they make different assumptions about returns, volatility, correlations, and diversification.

For example, Maximum Sharpe relies on estimates of expected returns; Minimum Volatility focuses on the covariance structure; Risk Parity balances contributions to portfolio risk; and Hierarchical Risk Parity uses clusters of correlated assets. None is universally superior. The objective is to make these trade-offs visible and comparable rather than present a single allocation as the definitive answer.

## Core capabilities

### 1. Asset selection and quantitative screening

Users can select assets manually or apply a multi-signal ranking approach. The quantitative screener evaluates four complementary dimensions:

- **Momentum:** relative price strength, primarily using a 12–1 month signal.
- **Risk:** historical volatility and maximum drawdown.
- **Value:** valuation proxies such as earnings yield and book-to-market.
- **Quality:** profitability and financial leverage, including ROE and debt-to-equity.

Signals are converted into cross-sectional percentile rankings and combined into a **Quant Score**. Missing fundamental data are not automatically treated as poor fundamentals. Screening and portfolio allocation remain separate stages, making it easier to distinguish the effects of asset choice from those of portfolio weights.

### 2. Portfolio optimization and allocation

The application explores Markowitz's mean–variance framework, the efficient frontier, and eight allocation strategies:

| Strategy | Main idea |
| --- | --- |
| Historical Maximum Sharpe | Maximize estimated excess return per unit of volatility using historical mean returns. |
| CAPM Maximum Sharpe | Derive expected returns from market beta and the market risk premium. |
| Fama–French 3-Factor | Incorporate market, size, and value factors into the expected-return model. |
| Fama–French 5-Factor | Extend the factor framework with profitability and investment. |
| Minimum Volatility | Minimize estimated portfolio volatility without forecasting returns. |
| Risk Parity | Seek balanced contributions to overall portfolio risk. |
| Hierarchical Risk Parity (HRP) | Allocate capital using hierarchical clustering of asset correlations. |
| Equal Weight | Use a simple 1/N allocation as a baseline for comparison. |

Portfolio construction uses long-only, fully invested allocations, with configurable concentration limits where applicable. Comparing complex optimizers with simpler benchmarks helps reveal whether additional model complexity provides meaningful value.

### 3. Factor analysis

The framework investigates systematic market exposure through **CAPM beta, alpha, correlation, and R²**, alongside factor-based approaches inspired by Fama–French models. This helps separate historical performance from the assumptions used to estimate future expected returns. A positive historical alpha is not interpreted as a guaranteed source of future outperformance.

### 4. Backtesting and performance evaluation

Historical evaluation includes **walk-forward** and **fixed-horizon** backtesting. The distinction between in-sample estimation and out-of-sample evaluation is central: portfolio weights should be determined using information available at the relevant decision date, rather than future observations.

Strategies can be assessed through total return, CAGR, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, and turnover. These measures describe different aspects of performance and should be interpreted together, not as a single universal ranking.

### 5. Volatility analytics and dynamic risk management

Risk analysis extends beyond a single historical volatility estimate. The methodological framework covers rolling realized volatility, **EWMA, GARCH, GJR-GARCH, and EGARCH**, as well as out-of-sample forecast assessment using metrics such as MSE and QLIKE.

Dynamic risk management explores volatility targeting and exposure reviews between scheduled portfolio rebalancing dates. Such rules are treated as risk-control mechanisms, not automatic guarantees against losses.

### 6. Monte Carlo simulations and robustness

Monte Carlo analysis examines a range of potential portfolio paths rather than one projected outcome. The framework considers **block bootstrap, IID bootstrap, and multivariate Gaussian** simulations, alongside indicators such as terminal wealth, probability of loss, Value at Risk, and drawdown risk.

Robustness comparisons vary simulation methods and historical subsamples to investigate how sensitive conclusions are to modeling choices. Simulated frequencies are conditional on the input data and assumptions; they are **not objective probabilities of future market outcomes**.

## Analytical workflow

```text
Asset universe
    ↓
Manual selection / Momentum–Risk–Value–Quality screening
    ↓
Expected-return and factor models
    ↓
Covariance and volatility estimation
    ↓
Portfolio construction and efficient frontier
    ↓
Backtesting, risk monitoring, and Monte Carlo analysis
    ↓
Comparison of performance, assumptions, and robustness
```

## Technology

The application is built with **Python and Streamlit**, using a modular architecture for data acquisition, screening, return and risk calculations, optimization, factor analysis, backtesting, simulations, and visualization. Key technologies include **pandas, NumPy, SciPy, yfinance, and Plotly**.

## Research methodology

The accompanying paper, *Quantitative Portfolio Optimization: Metodologie quantitative per un investimento consapevole* (**Version 5.0, October 2026**), documents the theoretical foundations, mathematical formulations, implementation choices, exploratory empirical results, and methodological limitations.

The project is guided by five principles: **transparency, comparability, prudence, discipline, and consistency**. Its main research question is not merely *“Which strategy has the highest return?”* but also *“How reliable is that result, what assumptions produced it, and how sensitive is it to estimation error and changing market conditions?”*

## Limitations and ongoing research

Historical estimates are noisy, correlations and factor premia can change, and backtests are vulnerable to look-ahead bias, survivorship bias, data snooping, and imperfect point-in-time fundamentals. Transaction costs, liquidity constraints, and slippage can further affect real-world performance. The paper's empirical findings are exploratory, and prospective validation remains an important next step.

Future methodological directions include covariance shrinkage, Black–Litterman allocation, improved transaction-cost modeling, immutable data snapshots, and stronger rolling out-of-sample validation.

> **Disclaimer:** Quant Portfolio Optimizer is intended for educational and research purposes. It is not personalized investment advice, and neither historical results nor simulated outcomes guarantee future performance.
