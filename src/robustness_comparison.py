"""Benchmark-independent summaries and revaluation on shared OOS marks."""
import numpy as np
import pandas as pd

from src.benchmarks import matched_portfolio_metrics

FIELDS = ("Total Return", "Sharpe Ratio", "Max Drawdown")


def performance_summary(rows):
    return rows.assign(positive=rows["Total Return"].gt(0)).groupby("Model", sort=False).agg(
        **{
            "Windows": ("Total Return", "count"),
            "Mean Return": ("Total Return", "mean"),
            "Median Return": ("Total Return", "median"),
            "Positive Rate": ("positive", "mean"),
            "Mean Sharpe": ("Sharpe Ratio", "mean"),
            "Median Sharpe": ("Sharpe Ratio", "median"),
            "Mean Drawdown": ("Max Drawdown", "mean"),
            "Worst Drawdown": ("Max Drawdown", "min"),
        }
    )


def absolute_rows(results, models, basis):
    rows = results.loc[results["Model"].isin(models), ["OOS Start", "Model", *FIELDS]].copy()
    if basis == "Net":
        rows.loc[:, list(FIELDS)] = results.loc[rows.index, [f"Net {field}" for field in FIELDS]].to_numpy()
    return rows


def active_risk(portfolio, benchmark, dates, *, gross):
    """Use only return legs that span one portfolio trading interval."""
    column = "Gross Portfolio Value" if gross else "Portfolio Value"
    consecutive = np.diff(portfolio.index.get_indexer(dates)) == 1
    active = (portfolio.loc[dates, column].pct_change() - benchmark.loc[dates].pct_change())
    active = active.iloc[1:].loc[consecutive].dropna()
    tracking = float(active.std(ddof=1) * np.sqrt(252)) if len(active) >= 2 else np.nan
    return tracking, float(active.mean() * 252 / tracking) if tracking > 0 else np.nan, len(active)


def matched_sample(results, info, models, benchmarks, basis, *, observed_only=False):
    """A single benchmark gives pairwise coverage, multiple give their joint sample."""
    original = absolute_rows(results, models, basis).set_index(["OOS Start", "Model"])
    reports = {(row["Window"], row["Benchmark"]): row for row in info["coverage_reports"]}
    diagnostics = []
    for row in info.get("benchmark_diagnostics", []):
        if row["Type"] == "IMPUTED_ADJ_CLOSE":
            diagnostics.append((row["OOS Start"], row["Ticker"], row["Date"]))

    rows, comparisons, exclusions, session_rows = [], [], [], []
    rate = info["risk_free_rate"]
    gross = basis == "Gross"
    for window in (item["OOS Start"] for item in info["window_status"]):
        portfolios = info["portfolio_nav"].get(window, {})
        references = info["benchmark_nav"].get(window, {})
        missing = [model for model in models if (window, model) not in original.index or model not in portfolios]
        reasons = [f"Models incomplete: {', '.join(missing)}"] if missing else []
        for name in benchmarks:
            report = reports.get((window, name), {})
            if name not in references:
                reasons.append(f"{name}: {report.get('Reason') or report.get('Status') or 'No comparable daily coverage'}")
            elif observed_only and report.get("Status") == "COMPLETE_WITH_IMPUTATION":
                reasons.append(f"{name}: window contains interpolated quotes")
        if reasons:
            exclusions.append({"OOS Start": window, "Reason": " | ".join(reasons)})
            continue
        dates = portfolios[models[0]].index
        for nav in [*(portfolios[model] for model in models[1:]), *(references[name] for name in benchmarks)]:
            dates = dates.intersection(nav.index)
        if len(dates) < 2:
            exclusions.append({"OOS Start": window, "Reason": "Fewer than two shared NAV marks"})
            continue
        imputed = {}
        for name in benchmarks:
            report = reports.get((window, name), {})
            imputed[name] = {day for start, ticker, day in diagnostics
                             if start == report.get("OOS Start") and ticker == report.get("Ticker")}
        for date in dates:
            day = str(date.date())
            session_rows.append({
                "OOS Start": window, "Session": day,
                "Interpolated Benchmarks": ", ".join(name for name in benchmarks if day in imputed[name]),
            })

        # Rebase each benchmark to its first shared mark. Portfolio entry cost
        # remains in the result when the first shared mark is the OOS entry.
        baseline = {}
        for name in benchmarks:
            frame = references[name].loc[dates].to_frame("Portfolio Value")
            metrics = matched_portfolio_metrics(frame, dates, rate)
            baseline[name] = metrics["Total Return"]
            rows.append({"OOS Start": window, "Model": name, **{field: metrics[field] for field in FIELDS}})
        for model in models:
            nav = portfolios[model]
            metrics = matched_portfolio_metrics(nav, dates, rate, gross=gross)
            matched = metrics["Total Return"]
            full = original.loc[(window, model), "Total Return"]
            rows.append({"OOS Start": window, "Model": model, **{field: metrics[field] for field in FIELDS}})
            for name in benchmarks:
                tracking, information, count = active_risk(nav, references[name], dates, gross=gross)
                comparisons.append({
                    "OOS Start": window, "Model": model, "Benchmark": name,
                    "Original Return": full, "Matched Return": matched, "Matched minus Original": matched - full,
                    "Benchmark Return": baseline[name], "Excess Return": matched - baseline[name],
                    "Benchmark Win": matched > baseline[name], "Tracking Error": tracking,
                    "Information Ratio": information, "Active Return Legs": count,
                })
    return tuple(pd.DataFrame(items) for items in (rows, comparisons, exclusions, session_rows))


def relative_summary(rows):
    return rows.groupby(["Model", "Benchmark"], sort=False).agg(
        **{
            "Windows": ("Benchmark Win", "size"),
            "Benchmark Win Rate": ("Benchmark Win", "mean"),
            "Mean Excess Return": ("Excess Return", "mean"),
            "Mean Matched minus Original": ("Matched minus Original", "mean"),
        }
    )
