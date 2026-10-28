"""FF3 vs FF5 walk-forward audit. Run from repository root.

python tests/test_walk_forward_ff3_ff5.py --prices data/prices_eur.csv --fx data/eurusd.csv \
    --lookback 756 --rebalance 63 --max-weight 0.25 --output results/ff_comparison

By default prices and EURUSD are downloaded using yfinance; optional CSV overrides.
The existing project backtest is reused unchanged. FF factor files are downloaded
through the project's existing loaders (network required).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'src').is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src.backtest import backtest_portfolio
from src.fama_french import load_europe_ff3
from src.fama_french_5 import load_europe_ff5


def load_csv(path):
    data = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    data.index = pd.DatetimeIndex(data.index).tz_localize(None)
    data = data.loc[~data.index.duplicated(keep='last')]
    return data.apply(pd.to_numeric, errors='coerce')



def download_market_data(tickers, start, end):
    """Download adjusted EUR equity prices and EUR/USD, retrying failed symbols.

    Drop only symbols with insufficient history. No synthetic price imputation.
    """
    import yfinance as yf
    symbols = list(dict.fromkeys(tickers))
    if len(symbols) < 2:
        raise ValueError('Specify at least two EUR-denominated tickers')
    print(f'Downloading {len(symbols)} assets and EURUSD=X ...', flush=True)

    def close_series(symbol):
        try:
            data = yf.download(symbol, start=start, end=end, auto_adjust=True,
                               progress=False, threads=False)
            if data.empty:
                return pd.Series(dtype=float, name=symbol)
            close = data['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors='coerce')
            close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
            return close.loc[~close.index.duplicated(keep='last')].rename(symbol)
        except Exception as exc:
            print(f'WARNING: download failed for {symbol}: {exc}', flush=True)
            return pd.Series(dtype=float, name=symbol)

    # Individual downloads also isolate failures and make retry behavior explicit.
    columns = {}
    rejected = []
    for symbol in symbols:
        series = close_series(symbol)
        if series.notna().sum() < 100:
            print(f'WARNING: retrying {symbol} ...', flush=True)
            series = close_series(symbol)
        if series.notna().sum() < 100:
            rejected.append(symbol)
            continue
        columns[symbol] = series
    if rejected:
        print(f'WARNING: excluded unavailable/short-history tickers: {rejected}', flush=True)
    if len(columns) < 2:
        raise ValueError('Fewer than two valid equity tickers remain')
    fx_series = close_series('EURUSD=X')
    if fx_series.notna().sum() < 100:
        fx_series = close_series('EURUSD=X')
    if fx_series.notna().sum() < 100:
        raise ValueError('EURUSD=X unavailable; cannot estimate FF3/FF5 in EUR')
    prices = pd.concat(columns.values(), axis=1).sort_index()
    fx = fx_series.to_frame().sort_index()
    print(f'Using {len(columns)} assets: {list(columns)}', flush=True)
    return prices, fx


def performance(bt, rf_annual):
    r = bt['Daily Return'].astype(float)
    equity = (1 + r).cumprod()
    n = len(r) - 1  # first date is portfolio formation, no preceding return
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (252 / n) - 1) if n > 0 else np.nan
    vol = float(r.iloc[1:].std(ddof=1) * np.sqrt(252))
    sharpe = float((r.iloc[1:].mean() * 252 - rf_annual) / vol) if vol > 0 else np.nan
    dd = equity / equity.cummax() - 1
    return {'Cumulative Return': total, 'CAGR': cagr, 'Annual Volatility': vol,
            'Sharpe': sharpe, 'Maximum Drawdown': float(dd.min())}


def audit_weights(prices, target_weights, bt):
    """Calculate actual one-way turnover against drifted pre-trade weights.

    The existing engine trades at the rebalance-date price. Therefore the prior
    holdings are marked to that SAME date, and only then compared with new targets.
    Initial deployment is separately flagged and excluded from recurring turnover.
    """
    records, pre_rows, drift_rows = [], [], []
    cols = list(prices.columns)
    previous_target = None
    previous_date = None
    for date, target in target_weights.iterrows():
        target = target.reindex(cols).fillna(0).astype(float)
        if previous_target is None:
            pre = pd.Series(0., index=cols)
            trade = float(target.abs().sum() / 2)  # initial funding convention
            drift = 0.
            initial = True
        else:
            relatives = prices.loc[date, cols] / prices.loc[previous_date, cols]
            held = previous_target * relatives
            if (held[previous_target > 1e-10].isna().any() or
                    (held[previous_target > 1e-10] < 0).any()):
                raise ValueError(f'Missing/invalid held-asset prices on {date}')
            held = held.fillna(0.)
            pre = held / held.sum()
            trade = float((target - pre).abs().sum() / 2)
            drift = float((pre - previous_target).abs().sum() / 2)
            initial = False
        pre.name = date
        pre_rows.append(pre)
        drift_rows.append((pre - previous_target).rename(date) if previous_target is not None
                          else pd.Series(0., index=cols, name=date))
        records.append({'Date': date, 'Initial Investment': initial,
                        'Turnover One Way': trade, 'Market Drift L1 Half': drift,
                        'Target Change L1 Half': float((target - previous_target).abs().sum()/2)
                        if previous_target is not None else np.nan,
                        'Assets Held': int((target > 1e-10).sum())})
        previous_target, previous_date = target, date
    audit = pd.DataFrame(records).set_index('Date')
    return audit, pd.DataFrame(pre_rows), pd.DataFrame(drift_rows)


def plot_all(results, output):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, res in results.items():
        series = res['bt']['Portfolio Value']
        ax.plot(series.index, series / series.iloc[0], label=name)
    ax.set(title='FF3 vs FF5 | wealth index (gross, initial=1)', ylabel='Index')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output / '01_equity.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, res in results.items():
        s = res['bt']['Portfolio Value']
        ax.plot(s.index, s / s.cummax() - 1, label=name)
    ax.set(title='Drawdown', ylabel='Drawdown'); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / '02_drawdown.png', dpi=170); plt.close(fig)

    for name, res in results.items():
        w = res['weights']
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.stackplot(w.index, *[w[c].to_numpy() for c in w.columns], labels=w.columns)
        ax.set(title=f'{name} | target allocation at each rebalance', ylim=(0, 1), ylabel='Weight')
        if len(w.columns) <= 20:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        fig.tight_layout(); fig.savefig(output / f'03_weights_{name}.png', dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, res in results.items():
        a = res['audit'].loc[~res['audit']['Initial Investment']]
        ax.plot(a.index, a['Turnover One Way'], marker='o', label=name)
    ax.set(title='Quarterly one-way turnover (excluding initial purchase)', ylabel='Turnover')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output / '04_turnover.png', dpi=170); plt.close(fig)

    comparison = pd.DataFrame({name: performance(res['bt'], res['rf']) for name, res in results.items()}).T
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    comparison[['Cumulative Return', 'CAGR']].plot.bar(ax=axes[0], rot=0)
    comparison[['Annual Volatility', 'Maximum Drawdown']].plot.bar(ax=axes[1], rot=0)
    fig.suptitle('FF3 vs FF5 | performance comparison')
    fig.tight_layout(); fig.savefig(output / '05_comparison.png', dpi=170); plt.close(fig)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', help='Optional CSV of EUR adjusted close prices')
    parser.add_argument('--fx', help='Optional CSV of EURUSD close prices')
    parser.add_argument('--tickers', nargs='+', default=['ENI.MI', 'ENEL.MI', 'ISP.MI', 'UCG.MI', 'STM.MI', 'G.MI', 'TIT.MI', 'LDO.MI'], help='EUR-denominated tickers (illustrative fixed universe)')
    parser.add_argument('--start', default='2017-01-01', help='Historical data start, inclusive')
    parser.add_argument('--end', default=None, help='Historical data end, exclusive')
    parser.add_argument('--lookback', type=int, default=756)
    parser.add_argument('--rebalance', type=int, default=63)
    parser.add_argument('--max-weight', type=float, default=.25)
    parser.add_argument('--rf', type=float, default=.02, help='Annual EUR risk-free, decimal')
    parser.add_argument('--capital', type=float, default=10000)
    parser.add_argument('--output', default='results/ff_comparison')
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if bool(args.prices) != bool(args.fx):
        parser.error('Provide both --prices and --fx, or neither for automatic download')
    if args.prices:
        prices, fx = load_csv(args.prices), load_csv(args.fx)
    else:
        prices, fx = download_market_data(args.tickers, args.start, args.end)
    if (prices.isna().any().any()):
        print('WARNING: removing rows with missing asset prices to keep a fixed common universe')
        prices = prices.dropna()
    fx = fx.reindex(prices.index).ffill()
    if fx.isna().any().any() or (fx <= 0).any().any():
        raise ValueError('Missing EURUSD exchange rates on asset trading dates')
    if prices.shape[1] * args.max_weight < 1 - 1e-10:
        raise ValueError(f'Infeasible max-weight constraint: {prices.shape[1]} assets x {args.max_weight:.1%} < 100%. Add tickers or increase --max-weight')
    if prices.shape[1] < 2 or len(prices) <= args.lookback + 1:
        raise ValueError('At least two assets and lookback + 2 observations are required')
    if prices.index.has_duplicates or (prices.dropna(how='all') <= 0).any().any():
        raise ValueError('Invalid prices')
    if args.rebalance <= 0 or args.lookback < 100:
        raise ValueError('Invalid lookback or rebalance frequency')
    factors = {'FF3': load_europe_ff3(), 'FF5': load_europe_ff5()}
    results = {}
    for name in ('FF3', 'FF5'):
        print(f'Running {name} ...', flush=True)
        kwargs = {'ff3_factors': factors['FF3']} if name == 'FF3' else {'ff5_factors': factors['FF5']}
        bt, weights = backtest_portfolio(
            prices=prices, strategy=f'{name.lower()}_max_sharpe',
            initial_capital=args.capital, lookback_days=args.lookback,
            rebalance_days=args.rebalance, risk_free_rate=args.rf,
            max_weight=args.max_weight, eurusd_prices=fx,
            **kwargs)
        audit, pre, drift = audit_weights(prices, weights, bt)
        results[name] = {'bt': bt, 'weights': weights, 'audit': audit,
                         'pre': pre, 'drift': drift, 'rf': args.rf}
        for label, data in [('backtest', bt), ('target_weights', weights),
                            ('pretrade_weights', pre), ('market_drift', drift),
                            ('turnover', audit)]:
            data.to_csv(output / f'{name.lower()}_{label}.csv')
    comparison = plot_all(results, output)
    for name, res in results.items():
        a = res['audit'].loc[~res['audit']['Initial Investment'], 'Turnover One Way']
        comparison.loc[name, 'Mean Quarterly Turnover'] = a.mean()
        comparison.loc[name, 'Total One-way Turnover'] = a.sum()
    comparison.to_csv(output / 'metrics_comparison.csv')
    print('\n', comparison.round(4).to_string(), '\n')
    print(f'Files written to {output.resolve()}')
    print('IMPORTANT: gross returns; transaction costs are NOT included.')
    print('Fixed illustrative ticker universe: results may have survivorship bias.')


if __name__ == '__main__':
    main()
