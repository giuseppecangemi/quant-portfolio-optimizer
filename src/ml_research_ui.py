"""Integrated return/covariance research workspace."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.forward_mc_ui import load_prices
from src.ml_estimators import (
    ANNUAL, CONFIGS, MODELS, OPTIMIZERS, checked_prices, efficient_frontier,
    estimate_covariance, fit_portfolio, forecast_metrics, horizon_to_daily,
    return_fit, return_panel, validate_covariances, validate_returns,
)


TABS = ["Model Configuration", "Expected Returns", "Covariance Forecasting",
        "Model Validation", "Portfolio Optimization", "Portfolio Comparison"]
COVARIANCES = ["Sample", "Ledoit–Wolf", "EWMA", "ML"]


def chart(fig, title):
    fig.update_layout(template="plotly_dark",title=title,title_font_color="#ffbf69")
    st.plotly_chart(fig)


def render_ml_research(tickers, market, rf, cap, capital, theme_tabs, allocation, display):
    theme_tabs()
    with st.container(key="backtest_analysis_tabs_ml"):
        tabs=st.tabs(TABS,key="ml_research_view",on_change="rerun")
    if tabs[0].open:
        with tabs[0], st.container(key="kangemi_controls_ml_research"):
            a,b,c,d=st.columns(4)
            history=a.selectbox("Training history",["2y","3y","5y","10y"],index=2)
            horizon=b.selectbox("Forecast horizon (sessions)",[1,5,21,63],index=2)
            estimator=c.selectbox("Return estimator",[*MODELS,"Compare All"])
            stride=d.selectbox("Validation step (sessions)",[5,21,63],index=1)
            st.caption("Features: 21–252 session momentum, rolling volatility, 63-session drawdown and "
                       "equal-weight universe momentum and relative momentum. Parameters are fixed; preprocessing "
                       "is fitted within each training sample.")
            if st.button("Estimate ML models",type="primary",disabled=not tickers):
                st.session_state.pop("ml_research",None)
                try:
                    with st.spinner("Preparing causal features and model forecasts…"):
                        prices=load_prices(tuple(tickers))
                        zone="America/New_York" if market=="USA" else "Europe/Rome"
                        prices=prices.loc[prices.index.date<pd.Timestamp.now(tz=zone).date(),list(tickers)]
                        prices=checked_prices(prices.tail(int(history[:-1])*ANNUAL+1))
                        returns=prices.pct_change(fill_method=None).iloc[1:]
                        panel=return_panel(prices,horizon)
                        forecasts, importance, errors = [],{},{}
                        models=MODELS if estimator=="Compare All" else (estimator,)
                        for model in models:
                            try:
                                predicted,_,importance[model],_=return_fit(panel,prices.index[-1],model)
                                daily=horizon_to_daily(predicted,horizon)
                                forecasts.append(pd.DataFrame({"Ticker":predicted.index,"Model":model,
                                    "Horizon Return":predicted.to_numpy(),"Daily Equivalent":daily,
                                    "Annualized Arithmetic Equivalent":daily*ANNUAL}))
                            except (ImportError, ValueError, RuntimeError) as exc:
                                errors[model]=str(exc)
                        covs={}
                        for method in COVARIANCES:
                            try:
                                covs[method]=estimate_covariance(returns,method)
                            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                                errors[f"{method} covariance"]=str(exc)
                        st.session_state["ml_research"]=dict(prices=prices,returns=returns,horizon=horizon,
                            history=history,market=market,stride=stride,models=[f.Model.iloc[0] for f in forecasts],
                            forecasts=pd.concat(forecasts,ignore_index=True) if forecasts else pd.DataFrame(),
                            importance=importance,covs=covs,errors=errors,portfolios={})
                except Exception as exc:
                    st.error(str(exc))
    result=st.session_state.get("ml_research")
    if result is None or result["market"]!=market or list(result["prices"].columns)!=list(tickers):
        st.info("Estimate the models for the current market and asset universe in Model Configuration.")
        return
    st.caption(f"Saved analysis · {result['prices'].index[-1].date()} close · {result['history']} · "
               f"{result['horizon']}-session return horizon · {len(result['prices'].columns)} assets")
    for model,error in result["errors"].items():
        st.warning(f"{model} · {error}")
    if tabs[1].open:
        with tabs[1]:
            render_returns(result)
    if tabs[2].open:
        with tabs[2]:
            render_covariance(result)
    if tabs[3].open:
        with tabs[3]:
            render_validation(result)
    if tabs[4].open:
        with tabs[4]:
            render_optimization(result,rf,cap,capital,allocation,display)
    if tabs[5].open:
        with tabs[5]:
            render_comparison(result)


def render_returns(result):
    forecasts=result["forecasts"]
    if forecasts.empty:
        st.info("No return model is available for this sample.")
        return
    st.dataframe(forecasts.style.format({c:"{:.2%}" for c in forecasts.columns[2:]}))
    comparison=forecasts.pivot(index="Ticker",columns="Model",values="Annualized Arithmetic Equivalent")
    comparison["Historical Mean"]=result["returns"].mean()*ANNUAL
    chart(px.bar(comparison.reset_index(),x="Ticker",y=list(comparison.columns),barmode="group"),
          "Expected annualized return by asset")
    chart(px.histogram(forecasts,x="Horizon Return",color="Model",barmode="overlay"),"Forecast dispersion by model")
    for name,importance in result["importance"].items():
        chart(px.bar(importance.rename("Effect").rename_axis("Feature").reset_index(),x="Feature",y="Effect"),
              f"{name} · {'Standardized coefficients' if name=='Elastic Net' else 'Tree feature importance'}")
    st.caption("Cumulative H-session predictions are converted to daily equivalents via the geometric root, "
               "then multiplied by 252 for compatibility with annualized optimizer inputs.")


def render_covariance(result):
    choices=list(result["covs"])
    if not choices:
        st.info("No valid covariance estimates.")
        return
    first=st.selectbox("Reference covariance",choices)
    second=st.selectbox("Comparison covariance",choices,index=choices.index("ML") if "ML" in choices else 0)
    for col,name in zip(st.columns(2),(first,second)):
        with col:
            cov=result["covs"][name]
            sd=np.sqrt(np.diag(cov))
            corr=cov.div(sd,axis=0).div(sd,axis=1)
            chart(px.imshow(cov,color_continuous_scale="RdBu",color_continuous_midpoint=0),f"{name} · Annual covariance")
            chart(px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale="RdBu"),f"{name} · Correlation")
            eigen=np.linalg.eigvalsh(cov)
            chart(px.line(y=eigen,labels={"index":"Eigenvalue index","y":"Annual covariance eigenvalue"}),f"{name} · Eigenvalue spectrum")
            st.caption(f"Min eigenvalue {eigen[0]:.3g} · PSD tolerance 1e−10 relative · Condition number {np.linalg.cond(cov):.3g}")
    st.caption("The ML estimator predicts next-session squared returns with a pooled random forest using "
               "1, 5, 21 and 63-session trailing squared-return means. Ledoit–Wolf correlation rescales "
               "the predicted asset variances into one PSD covariance matrix.")


def render_validation(result):
    if st.button("Run chronological validation",type="primary"):
        with st.spinner("Refitting on successive past-only windows…"):
            if result["models"]:
                try:
                    result["return_validation"]=validate_returns(result["prices"],result["horizon"],result["models"],result["stride"])
                except Exception as exc:
                    st.error(f"Return validation · {exc}")
            if result["covs"]:
                try:
                    result["cov_validation"]=validate_covariances(result["returns"],list(result["covs"]),result["stride"])
                except Exception as exc:
                    st.error(f"Covariance validation · {exc}")
    data=result.get("return_validation")
    if isinstance(data,pd.DataFrame) and not data.empty:
        metrics,ic=forecast_metrics(data)
        st.dataframe(metrics)
        chart(px.bar(metrics.melt(id_vars=["Model","Phase"],value_vars=["MAE","RMSE"]),
                     x="Model",y="value",color="variable",facet_col="Phase",barmode="group"),"Forecast errors by partition")
        model=st.selectbox("Diagnostic model",result["models"])
        sample=data.loc[data.Model.eq(model)].assign(Error=lambda d:d.Predicted-d.Actual)
        scatter=px.scatter(sample,x="Actual",y="Predicted",color="Phase",hover_data=["Date","Ticker"])
        low,high=sample[["Actual","Predicted"]].min().min(),sample[["Actual","Predicted"]].max().max()
        scatter.add_scatter(x=[low,high],y=[low,high],mode="lines",name="Perfect forecast")
        chart(scatter,"Predicted vs realized horizon return")
        chart(px.histogram(sample,x="Error",color="Phase"),"Forecast error distribution")
        chart(px.line(sample.groupby("Date").Error.mean()),"Mean forecast error over time")
        rank=ic.loc[ic.Model.eq(model)].set_index("Date")[["Rank IC"]]
        rank["Rolling mean (5 origins)"]=rank["Rank IC"].rolling(5).mean()
        if rank.notna().any().any():
            chart(px.line(rank),"Cross-sectional Rank IC")
        else:
            st.caption("Rank IC requires at least three assets and cross-sectional dispersion.")
        asset=st.selectbox("Diagnostic asset",sample.Ticker.unique())
        chart(px.line(sample.loc[sample.Ticker.eq(asset)],x="Date",y=["Predicted","Actual","Historical"]),
              f"{asset} · Horizon forecasts and outcomes")
        test=metrics.loc[metrics.Model.eq(model)&metrics.Phase.eq("Test")]
        if len(test):
            st.caption(f"Test OOS R² versus the trailing mean: {test.OOS_R2.iloc[0]:.3f}. "
                       "Positive values mean lower squared error. Rank IC measures cross-sectional ordering. "
                       "Overlapping outcomes can be dependent; these results are descriptive.")
        st.download_button("Download forecast diagnostics",data.to_csv(index=False),"ml_validation.csv","text/csv")
    cov=result.get("cov_validation")
    if isinstance(cov,pd.DataFrame) and not cov.empty:
        st.dataframe(cov.groupby("Method").mean(numeric_only=True))
        chart(px.line(cov,x="Date",y="Frobenius",color="Method"),"Covariance forecast error")
        chart(px.line(cov,x="Date",y="ConditionNumber",color="Method",log_y=True),"Covariance conditioning")
        chart(px.line(cov.melt(id_vars=["Date","Method"],value_vars=["PredictedVol","RealizedVol"]),
                      x="Date",y="value",color="Method",line_dash="variable"),"Predicted vs realized equal-weight volatility")
        st.caption("The covariance proxy is the average outer product of the next 21 daily returns. "
                   "It is noisy and uses a zero-mean convention. Smaller errors and larger predictive log-likelihood "
                   "indicate closer agreement on these forecast origins.")


def render_optimization(result,rf,cap,capital,allocation,display):
    cfg=st.selectbox("Estimation configuration",CONFIGS)
    model=st.selectbox("ML return estimator",result["models"] or ["Unavailable"]) if cfg in ("ML Returns","Full ML") else None
    base=st.selectbox("Historical covariance",COVARIANCES[:-1]) if cfg in ("Historical","ML Returns") else "Sample"
    solver=st.selectbox("Optimization method",OPTIMIZERS)
    config=dict(configuration=cfg,model=model,horizon=result["horizon"],optimizer=solver,
                historical_cov=base,market=result["market"])
    name=" · ".join([cfg,*([model] if model else []),base if cfg in ("Historical","ML Returns") else "ML risk",solver,
                     f"H{result['horizon']}" if model else "Daily"])
    if st.button("Optimize portfolio",type="primary",disabled=model=="Unavailable"):
        try:
            w,mu,cov=fit_portfolio(result["prices"],config,rf,cap)
            result["portfolios"][name]=dict(config=config,w=w,mu=mu,cov=cov,rf=rf,cap=cap)
        except (ValueError,ImportError,RuntimeError) as exc:
            detail = str(exc)
            summary = detail.partition(" Initial weights=")[0].partition(" Annualized inputs")[0]
            st.error(f"Portfolio optimization failed · {summary}")
            with st.expander("Technical optimization diagnostics"):
                st.text(detail)
    if name not in result["portfolios"]:
        return
    saved=result["portfolios"][name]
    if saved["rf"]!=rf or saved["cap"]!=cap:
        st.info("Run optimization again to apply the current risk-free rate and weight cap.")
        return
    w,mu,cov=saved["w"],saved["mu"],saved["cov"]
    vol=np.sqrt(w @ cov @ w)
    ret=w @ mu
    a,b,c,d=st.columns(4)
    a.metric("Expected annual return",f"{ret:.2%}")
    b.metric("Expected annual volatility",f"{vol:.2%}")
    c.metric("Expected Sharpe",f"{(ret-rf)/vol:.2f}" if vol else "—")
    d.metric("Weight concentration (HHI)",f"{(w*w).sum():.3f}")
    st.caption(f"Feasible allocation · Sum {w.sum():.6f} · Max {w.max():.2%} (cap {cap:.2%}). "
               "Minimum Volatility, Risk Parity and HRP are covariance-driven.")
    display(allocation(w,result["prices"],capital,True),True,cov)
    st.caption("Allocations assume fractional shares.")
    chart(px.bar(w.rename("Weight"),y="Weight"),"Target Portfolio Weights")
    if vol:
        rc=(w*(cov@w)/(vol*vol)).rename("Risk Share")
        chart(px.bar(rc,y="Risk Share"),"Signed Risk Contributions")
    with st.expander("Efficient frontier for these inputs"):
        if st.button("Calculate frontier"):
            frontier=efficient_frontier(mu,cov,points=25,max_weight=cap)
            if not frontier.empty:
                fig=px.line(frontier,x="volatility",y="return")
                fig.add_scatter(x=[vol],y=[ret],mode="markers",name=name)
                chart(fig,"Efficient frontier · selected mean and covariance")
    if st.button("Register for Backtesting"):
        registry=st.session_state.setdefault("ml_backtest_registry",{})
        registry[name]=config
        st.success("Configuration registered. Select it in Backtesting to refit on each historical training window.")


def render_comparison(result):
    if not result["portfolios"]:
        st.info("Optimize portfolios in the preceding tab to compare estimators.")
        return
    versions=list(result["portfolios"])
    selected=st.multiselect("Portfolios to compare",versions,default=versions)
    if not selected:
        return
    saved=result["portfolios"]
    weights=pd.DataFrame({name:saved[name]["w"] for name in selected})
    chart(px.imshow(weights,aspect="auto",color_continuous_scale="Blues"),"Allocation comparison · target weights")
    reference=st.selectbox("Common evaluation inputs",selected)
    mu,cov,rf=(saved[reference][key] for key in ("mu","cov","rf"))
    rows=[]
    for name in weights:
        w=weights[name]
        ret=float(w@mu)
        sigma=np.sqrt(w@cov@w)
        rows.append(dict(Portfolio=name,Return=ret,Volatility=sigma,
                         Sharpe=(ret-rf)/sigma if sigma else np.nan,HHI=(w*w).sum()))
    summary=pd.DataFrame(rows)
    st.dataframe(summary)
    chart(px.scatter(summary,x="Volatility",y="Return",color="Portfolio"),"Expected risk and return · common inputs")
    chart(px.bar(summary,x="Portfolio",y="Sharpe"),"Expected Sharpe · common inputs")
    chart(px.bar(summary,x="Portfolio",y="HHI"),"Weight concentration")
    baseline=weights[reference]
    st.dataframe(weights.sub(baseline,axis=0).style.format("{:+.2%}"))
    st.caption("These ex-ante comparisons use common estimated inputs. Historical predictive performance is evaluated in Backtesting.")
