# Quant Portfolio Optimizer

**A Python & Streamlit framework for quantitative portfolio construction, risk analysis, and strategy validation.**

Quant Portfolio Optimizer is an interactive research tool designed to support informed investment decisions through transparent, reproducible quantitative methods. Rather than predicting winning stocks, it compares different assumptions about expected returns, diversification, and risk.

## Features

- **Quantitative asset screening:** Rank assets using Momentum, Risk, Value, and Quality signals, or select assets manually.
- **Portfolio optimization:** Explore Markowitz's efficient frontier and compare eight allocation strategies:
  - Historical Maximum Sharpe
  - CAPM Maximum Sharpe
  - Fama–French 3-Factor and 5-Factor Maximum Sharpe
  - Minimum Volatility
  - Risk Parity
  - Hierarchical Risk Parity (HRP)
  - Equal Weight
- **Factor analysis:** Examine market beta, alpha, systematic risk, and factor-based expected returns.
- **Backtesting:** Evaluate strategies with walk-forward and fixed-horizon tests, distinguishing in-sample estimation from out-of-sample performance.
- **Volatility analytics:** Study realized and forecast volatility using rolling estimates, EWMA, GARCH, GJR-GARCH, and EGARCH.
- **Dynamic risk management:** Explore volatility targeting and exposure adjustments between portfolio rebalancing dates.
- **Monte Carlo simulations:** Analyze potential outcomes using historical bootstrap and multivariate Gaussian simulations, with sensitivity comparisons across methods and samples.
- **Performance evaluation:** Compare returns, CAGR, Sharpe and Sortino ratios, volatility, maximum drawdown, and turnover.

## Research approach

The framework follows a structured analytical pipeline:

**Asset Universe → Screening → Expected Return Models → Risk Estimation → Portfolio Construction → Backtesting & Scenario Analysis**

Its central question is not simply *which portfolio performs best?* but **how sensitive is that result to model assumptions, estimation errors, and changing market conditions?**

## Technology

Python · Streamlit · pandas · NumPy · SciPy · yfinance · Plotly

## Methodology and limitations

The accompanying methodological paper, *Quantitative Portfolio Optimization — Metodologie quantitative per un investimento consapevole* (v5.0, October 2026), explains the financial theory, statistical assumptions, implementation choices, and exploratory empirical findings.

Historical backtests and simulated outcomes are not evidence of future performance. The project remains a research and educational framework; further prospective validation, transaction-cost modeling, and reproducibility improvements are areas for development.

**Disclaimer:** This project is for educational and research purposes only and does not constitute personalized financial advice.
