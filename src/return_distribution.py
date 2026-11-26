"""Ex-post return samples and signed one-session loss statistics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import jarque_bera, norm, t

from src.monte_carlo import empirical_tail_mean


HISTORICAL = "Historical Simulation"
GAUSSIAN = "Parametric Gaussian"
STUDENT = "Student-t"
LOGNORMAL = "Shifted Lognormal"
MODELS = (HISTORICAL, GAUSSIAN, STUDENT, LOGNORMAL)


def backtest_return_samples(curve, prices, weights):
    """Use the selected strategy's recorded returns and its evaluation dates."""
    dates = curve.index.sort_values()
    portfolio = curve.get("Daily Return", pd.Series(dtype=float)).sort_index()
    if dates.empty or prices is None:
        return portfolio, pd.DataFrame(index=dates)
    assets = prices.columns.intersection(weights.columns) if isinstance(weights, pd.DataFrame) else prices.columns
    evaluation = prices.sort_index().loc[dates[0]:dates[-1], assets]
    # Exclude the entry date's pre-entry price and preserve gaps in price history.
    evaluation = evaluation.reindex(evaluation.index.union(dates).sort_values())
    valid = evaluation.where(np.isfinite(evaluation) & evaluation.gt(0))
    return portfolio, valid.pct_change(fill_method=None).reindex(dates)


@dataclass(frozen=True)
class FittedDistribution:
    family: str
    location: float
    scale: float
    df: float | None = None

    def log_density(self, values):
        if self.family == GAUSSIAN:
            return norm.logpdf(values, loc=self.location, scale=self.scale)
        if self.family == STUDENT:
            return t.logpdf(values, self.df, loc=self.location, scale=self.scale)
        values = np.asarray(values, dtype=float)
        log_density = np.full(values.shape, -np.inf)
        valid = values > -1
        gross_log = np.log1p(values[valid])
        log_density[valid] = norm.logpdf(gross_log, loc=self.location, scale=self.scale) - gross_log
        return log_density

    def risk(self, confidence):
        """Return signed loss VaR and ES for the lower return tail."""
        p = 1 - confidence
        if self.family == STUDENT:
            q = t.ppf(p, self.df)
            var = -self.location - self.scale * q
            if self.df <= 1:
                return var, np.inf
            tail = (self.df + q * q) / (self.df - 1) * t.pdf(q, self.df) / p
            return var, -self.location + self.scale * tail
        z = norm.ppf(p)
        if self.family == LOGNORMAL:
            log_centile = self.location + self.scale * z
            log_tail = self.location + self.scale**2 / 2 + norm.logcdf(z - self.scale) - np.log(p)
            return -np.expm1(log_centile), -np.expm1(log_tail)
        return -self.location - self.scale * z, -self.location + self.scale * norm.pdf(z) / p


def fit_student(values, location, scale):
    """Multistart location-scale MLE on standardized returns with checked convergence."""
    if len(values) < 5 or np.unique(values).size < 3:
        raise ValueError("Student-t needs at least five observations and three distinct returns.")
    x = (values - location) / scale
    lower_scale, df_limit = 1e-8, 1e5

    def negative_log_likelihood(params):
        center, log_scale, log_df = params
        with np.errstate(over="ignore", invalid="ignore"):
            return -float(np.mean(t.logpdf(x, np.exp(log_df), loc=center, scale=np.exp(log_scale))))

    bounds = [(None, None), (np.log(lower_scale), np.log(1e3)), (np.log(1e-3), np.log(df_limit))]
    candidates = [minimize(negative_log_likelihood, (0., 0., np.log(df)),
                           method="L-BFGS-B", bounds=bounds) for df in (2., 12., 1000.)]
    converged = [result for result in candidates if result.success and np.isfinite(result.fun)]
    if not converged:
        raise ValueError("Student-t likelihood optimization did not converge.")
    best = min(converged, key=lambda result: result.fun)
    center, log_scale, log_df = best.x
    relative_scale, df = np.exp(log_scale), np.exp(log_df)
    if relative_scale <= 10 * lower_scale or df <= .00101:
        raise ValueError("Student-t fit reached a numerical boundary. Tail parameters are unstable.")
    if best.fun > -float(np.mean(norm.logpdf(x))) + 1e-4:
        raise ValueError("Student-t fit did not recover the Gaussian-limit likelihood.")
    return FittedDistribution(STUDENT, location + scale * center, scale * relative_scale, df)


def fit_return_models(values):
    """Fit each model to the identical finite sample and compare return-scale likelihoods."""
    models, unavailable = {}, {}
    if np.ptp(values) == 0:
        return models, dict.fromkeys(MODELS[1:], "A constant sample has no positive-scale MLE.")

    mean, scale = (float(v) for v in norm.fit(values))
    for family in MODELS[1:]:
        try:
            if family == GAUSSIAN:
                fitted = FittedDistribution(family, mean, scale)
            elif family == STUDENT:
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("Sample scale is numerically invalid.")
                fitted = fit_student(values, mean, scale)
            else:
                if (values <= -1).any():
                    raise ValueError("Common sample contains returns at or below −100%.")
                fitted = FittedDistribution(family, *norm.fit(np.log1p(values)))
            if not np.isfinite([fitted.location, fitted.scale]).all() or fitted.scale <= 0:
                raise ValueError("Fitted location or scale is numerically invalid.")
            likelihood = float(fitted.log_density(values).sum())
            if not np.isfinite(likelihood):
                raise ValueError("Simple-return log-likelihood is nonfinite.")
            k = 3 if family == STUDENT else 2
            risks = {confidence: fitted.risk(confidence) for confidence in (.95, .99)}
            if any(not np.isfinite(var) or (
                not np.isfinite(es) and not (family == STUDENT and fitted.df <= 1)
            ) for var, es in risks.values()):
                raise ValueError("Fitted tail metrics are numerically invalid.")
            models[family] = {
                "fit": fitted, "Log-likelihood": likelihood, "AIC": 2 * k - 2 * likelihood,
                "BIC": np.log(len(values)) * k - 2 * likelihood, "risk": risks,
            }
        except (ValueError, FloatingPointError, RuntimeError, OverflowError) as exc:
            unavailable[family] = str(exc)
    return models, unavailable


def distribution_statistics(returns):
    """Summarize one common finite sample and fit all four distribution models."""
    values = returns.to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values)
    sample = returns.iloc[np.flatnonzero(finite)]
    if len(sample) < 2:
        return sample, None
    values = values[finite]
    constant = sample.nunique() == 1
    sigma = 0. if constant else float(sample.std(ddof=1))
    loss = -values
    jb = jarque_bera(values) if 0 < sigma < np.inf and len(values) > 2000 else None
    models, unavailable = fit_return_models(values)
    models[HISTORICAL] = {"risk": {
        confidence: (float(np.quantile(loss, confidence, method="inverted_cdf")),
                     -empirical_tail_mean(values, 1 - confidence))
        for confidence in (.95, .99)
    }}
    return sample, {
        "Mean return": float(sample.mean()), "Volatility": sigma,
        "Skewness": sample.skew() if not constant and len(sample) >= 3 else np.nan,
        "Excess kurtosis": sample.kurt() if not constant and len(sample) >= 4 else np.nan,
        "Observations": len(sample), "Excluded": len(returns) - len(sample),
        "JB statistic": float(jb.statistic) if jb is not None else np.nan,
        "JB p-value": float(jb.pvalue) if jb is not None else np.nan,
        "models": models, "unavailable": unavailable,
    }
