import streamlit as st
import pandas as pd
import plotly.express as px
from src.data import get_prices
from src.ml_forecasting import MODELS, fit_forecasts, backtest_ml
from src.ml_comparison import compare_at_date, walk_forward_comparison, backtest_metrics


def render_ml_forecasting(tickers, risk_free_rate, max_weight, capital):
    st.header('ML Return Forecasting')
    st.caption('Train predictive models on historical features, then optimize using ML expected returns rather than historical mean returns. All forecasts are out-of-sample only when evaluated on a held-out period.')
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        selection=st.selectbox('Machine Learning Model',['Compare All',*MODELS],key='ml_model')
    with c2:
        years=st.selectbox('Training history (years)',[1,2,3,5,10],index=2,key='ml_years')
    with c3:
        horizon=st.selectbox('Forecast horizon (trading days)',[21,63,126,252],index=1,key='ml_horizon')
    with c4:
        rebalance=st.selectbox('Rebalancing (trading days)',[21,63,126,252],index=1,key='ml_rebalance')
    test_sessions=st.selectbox('Backtest length (trading days)',[126,252,504],index=1,key='ml_test_sessions')
    fee_bps=st.number_input('Transaction cost per traded notional (bps)',min_value=0.0,max_value=100.0,value=10.0,step=1.0,key='ml_fee_bps')
    st.caption('The backtest keeps the initial asset universe fixed, retrains on each rebalance date and adjusts only weights. This does not simulate dynamic screening.')
    names=list(MODELS) if selection=='Compare All' else [selection]
    if not tickers:
        st.info('Select assets in the sidebar or run Quant Selection first.')
        return
    if len(tickers)*max_weight < 1-1e-8:
        st.warning('Maximum asset weight is infeasible for the selected number of assets.')
        return
    st.caption(f'{len(tickers)} assets · {years}y training · {horizon}-session forecast · {rebalance}-session rebalance · annual risk-free {risk_free_rate:.2%}')
    if st.button('Run ML Forecasting',type='primary',key='run_ml'):
        try:
            with st.spinner('Downloading history, training models and optimizing...'):
                # Training window plus a separate test window and feature warmup.
                requested_years=max(5,years+int(test_sessions/252)+3)
                prices=get_prices(list(tickers),period=f'{min(10,requested_years)}y')
                result=fit_forecasts(prices,years,horizon,names,risk_free_rate,max_weight)
                st.session_state['ml_results']=(result,prices,names,years,horizon,rebalance,test_sessions,fee_bps)
        except Exception as exc:
            st.error(f'ML forecasting failed: {exc}')
    stored=st.session_state.get('ml_results')
    if stored is None:
        return
    result,prices,stored_names,sy,sh,sr,stests,sfee=stored
    st.subheader('Prediction quality — chronological validation')
    st.dataframe(result.metrics,hide_index=True,use_container_width=True)
    st.caption(f'Training rows: {result.train_rows:,} · Validation rows: {result.validation_rows:,} · Forecast as of: {result.as_of.date()}. Validation uses a purged time split; metrics are not a walk-forward estimate.')
    st.subheader('ML Expected Returns')
    st.dataframe(result.forecasts.style.format({'Forecast horizon return':'{:.2%}','Expected return annualized':'{:.2%}'}),hide_index=True,use_container_width=True)
    st.subheader('Optimized Portfolio Allocation')
    st.dataframe(result.weights.style.format({'Weight':'{:.2%}'}),hide_index=True,use_container_width=True)
    for name in stored_names:
        weights=result.weights[result.weights.Model==name].set_index('Ticker').Weight
        st.caption(f'{name}: expected allocation on €{capital:,.2f} capital (fractional theoretical weights)')
        st.dataframe(pd.DataFrame({'Weight':weights,'Target amount (€)':weights*capital}).style.format({'Weight':'{:.2%}','Target amount (€)':'€{:,.2f}'}),use_container_width=True)
    if st.button('Run ML Walk-Forward Backtest',key='run_ml_bt'):
        try:
            with st.spinner('Running point-in-time retraining and walk-forward backtest...'):
                equity=backtest_ml(prices,sy,sh,sr,stored_names,risk_free_rate,max_weight,stests,sfee)
                st.session_state['ml_equity']=equity
        except Exception as exc:
            st.error(f'ML backtest failed: {exc}')
    equity=st.session_state.get('ml_equity')
    if equity is not None:
        st.subheader('ML Walk-Forward Portfolio Growth')
        st.plotly_chart(px.line(equity,x='Date',y='Equity',color='Model',title='Growth of €1 invested (net of configured trading fees)'),use_container_width=True)
        summary=equity.sort_values('Date').groupby('Model').tail(1).copy()
        summary['Total Return']=summary.Equity-1
        st.dataframe(summary[['Model','Total Return']].style.format({'Total Return':'{:.2%}'}),hide_index=True,use_container_width=True)
        st.caption('Research backtest: no taxes, slippage or dynamic universe; verify point-in-time constituents and delistings before interpreting as investable performance.')


def render_factor_ml_comparison(tickers, risk_free_rate, max_weight, capital):
    st.header('Traditional vs ML — Controlled Comparison')
    st.caption('Historical Mean, CAPM, FF3 and FF5 versus ML-enhanced estimators. All return estimates are daily internally; optimization uses annualized arithmetic returns and covariance.')
    c1,c2,c3,c4=st.columns(4)
    with c1: years=st.selectbox('Training years',[2,3,5],index=1,key='cmp_years')
    with c2: algorithm=st.selectbox('ML model',['Compare All',*MODELS],key='cmp_model')
    with c3: rebalance=st.selectbox('Rebalance (sessions)',[21,63,126,252],index=1,key='cmp_rebal')
    with c4: test=st.selectbox('Backtest sessions',[126,252,504],index=1,key='cmp_test')
    alpha=st.slider('Hybrid weight assigned to ML',0.,1.,.5,.05,key='cmp_alpha',help='Fixed before backtesting; not optimized on test performance.')
    fee=st.number_input('Trading cost (bps)',min_value=0.,max_value=100.,value=3.,key='cmp_fee')
    st.caption('Factor datasets use USD and are converted to EUR using the existing FF3/FF5 methods. The benchmark is the FTSE MIB price index, not a dividend-adjusted total-return index.')
    if not tickers: st.info('Select assets first.'); return
    names=list(MODELS) if algorithm=='Compare All' else [algorithm]
    if st.button('Estimate all portfolios',type='primary',key='cmp_run'):
        try:
            with st.spinner('Loading point-in-time factors and fitting estimators...'):
                from src.fama_french import load_europe_ff3
                from src.fama_french_5 import load_europe_ff5
                requested=min(10,years+int(test/252)+4)
                prices=get_prices(list(tickers),period=f'{requested}y')
                market=get_prices(['^FTSEMIB'],period=f'{requested}y').iloc[:,0]
                fx=get_prices(['EURUSD=X'],period=f'{requested}y')
                ff3=load_europe_ff3(); ff5=load_europe_ff5()
                result=compare_at_date(prices,market,ff3,ff5,fx,risk_free_rate,max_weight,years,names,alpha)
                st.session_state['factor_ml_compare']=(result,prices,market,ff3,ff5,fx,years,names,alpha,rebalance,test,fee)
                st.session_state.pop('factor_ml_equity',None)
        except Exception as exc: st.error(f'Comparison failed: {exc}')
    stored=st.session_state.get('factor_ml_compare')
    if stored is None:return
    result,prices,market,ff3,ff5,fx,sy,names,alpha,sr,stests,sfee=stored
    st.subheader('Daily expected returns — historical/factor vs ML')
    st.caption('Each cell is a forecast of the next trading day, not a realized return. Displayed in daily percentage terms.')
    st.dataframe(result['daily'].style.format({c:'{:.3%}' for c in result['daily'].columns if c!='Ticker'}),hide_index=True,use_container_width=True)
    st.subheader('Portfolio optimization — identical covariance and constraints')
    stats=result['stats']
    st.dataframe(stats.style.format({'Expected Return':'{:.2%}','Volatility':'{:.2%}','Sharpe Ratio':'{:.3f}'})
        .background_gradient(subset=['Expected Return','Sharpe Ratio'],cmap='Purples')
        .background_gradient(subset=['Volatility'],cmap='Reds'),hide_index=True,use_container_width=True)
    st.subheader('Portfolio allocation')
    allocations=result['weights'].copy()
    allocations['Target Amount (€)']=allocations['Weight']*capital
    st.dataframe(allocations.style.format({'Weight':'{:.2%}','Target Amount (€)':'€{:,.2f}'})
        .background_gradient(subset=['Weight'],cmap='Purples'),hide_index=True,use_container_width=True)
    st.caption('Target amounts assume fractional shares. The existing Portfolio Optimization execution table should be used for integer quantities and residual cash.')
    st.subheader('ML prediction quality — purged chronological validation')
    st.dataframe(result['quality'],hide_index=True,use_container_width=True)
    if st.button('Run controlled walk-forward backtest',key='cmp_backtest'):
        try:
            with st.spinner('Retraining and rebalancing all estimators at the same dates; this may take several minutes...'):
                equity=walk_forward_comparison(prices,market,ff3,ff5,fx,sy,names,stests,sr,risk_free_rate,max_weight,alpha,sfee,benchmark=market)
                st.session_state['factor_ml_equity']=equity
        except Exception as exc: st.error(f'Controlled backtest failed: {exc}')
    equity=st.session_state.get('factor_ml_equity')
    if equity is not None:
        st.subheader('Out-of-sample performance — identical rebalance dates')
        st.plotly_chart(px.line(equity,x='Date',y='Equity',color='Strategy'),use_container_width=True)
        metrics=backtest_metrics(equity, risk_free_rate)
        st.dataframe(metrics.style.format({'Total Return':'{:.2%}','CAGR':'{:.2%}','Volatility':'{:.2%}','Sharpe':'{:.2f}','Max Drawdown':'{:.2%}'})
            .background_gradient(subset=['Total Return','Sharpe'],cmap='Greens')
            .background_gradient(subset=['Max Drawdown'],cmap='Reds'),hide_index=True,use_container_width=True)
        st.caption('Research-only: fixed initial universe, no delisting adjustment, no tax or slippage. Benchmark is a price index; its dividend treatment differs from adjusted stock prices. Sharpe uses the selected annual risk-free rate.')
