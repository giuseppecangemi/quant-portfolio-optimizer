"""Shared Streamlit views for portfolio and asset volatility."""
import io
import json
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import get_prices
from src.portfolio_context import configure_portfolio, portfolio_scope, portfolio_settings
from src.risk_alerts import risk_alert
from src.volatility_analytics import (
    available_portfolios, complete_suffix, dated_returns, estimate,
    reconstruct_returns, target_risk, valid_weights, validate_models,
)
from src.volatility_forecasting import ANNUALIZATION, ARCH_AVAILABLE


MODELS = ["Historical", "Rolling", "EWMA", *(["GARCH(1,1)", "GJR-GARCH", "EGARCH"] if ARCH_AVAILABLE else [])]


@st.cache_data(ttl=3600, max_entries=40, show_spinner=False)
def market_prices(tickers, period, configuration=None):
    with portfolio_scope():
        configure_portfolio(configuration)
        return get_prices(list(tickers), period, preserve_missing=True)


def render_volatility(portfolio_mode, state, tickers, universe, metric):
    mode = "portfolio" if portfolio_mode else "asset"
    with st.container(key=f"kangemi_controls_volatility_{mode}"):
        if portfolio_mode:
            portfolios = available_portfolios(state)
            if not portfolios:
                st.info("Run Portfolio Optimization, Factor Models or the ML portfolio comparison to obtain target weights.")
                return
            name = st.selectbox("Portfolio Strategy", list(portfolios))
            allocation = portfolios[name]
            if set(allocation.weights.index) != set(tickers):
                st.warning("The current asset selection differs from the saved allocation. This analysis uses the saved universe.")
            st.caption(f"Source · {allocation.source} | Optimized · {allocation.optimized_at or 'Unavailable'} "
                       f"| Data through · {allocation.as_of.date() if pd.notna(allocation.as_of) else 'Unavailable'} "
                       f"| Weight cap · {allocation.max_weight if allocation.max_weight is not None else 'Unavailable'}")
            assets = allocation.weights[allocation.weights.abs() > 1e-10].index.tolist()
        else:
            allocation = None
            mapping = universe.set_index("ticker")["name"].to_dict()
            assets = st.multiselect("Asset Selection", list(mapping), default=[t for t in tickers if t in mapping][:30],
                                    format_func=lambda t:f"{mapping[t]} ({t})", max_selections=30)
            name = "Assets"
        a,b,c,d = st.columns(4)
        day = a.date_input("Analysis Date", max_value=date.today(), key=f"vol_date_{mode}")
        period = b.selectbox("Historical period", ["1y","2y","3y","5y","10y","max"], index=3, key=f"vol_history_{mode}")
        window = c.selectbox("Estimation Window (Sessions)", [252,504,756,1260], key=f"vol_estimation_{mode}")
        model = d.selectbox("Volatility Estimation Model", MODELS, index=2, key=f"vol_estimator_{mode}")
        a,b,c,d = st.columns(4)
        horizon = a.selectbox("Forecast Horizon (Sessions)", [1,5,21,63,126], index=2, key=f"vol_forecast_{mode}")
        rolling = b.selectbox("Rolling Window (Sessions)", [21,63,126,252], index=1, key=f"vol_rolling_{mode}")
        decay = c.number_input("EWMA lambda", min_value=.80,max_value=.99,value=.94,key=f"vol_lambda_{mode}")
        cov_method = d.selectbox("Covariance Estimation", ["Sample","EWMA","Ledoit–Wolf"], key="vol_cov") if portfolio_mode else "Sample"
        management = st.selectbox("Portfolio Management", ["Constant Weight","Buy & Hold"]) if portfolio_mode else None
        rebalance = st.selectbox("Rebalance every (sessions)", [1,21,63,126,252]) if management == "Constant Weight" else 0
        run = st.button("Run Volatility Analysis", type="primary", disabled=not assets, key=f"vol_run_{mode}")
    configuration = allocation.configuration if allocation else portfolio_settings()
    params = dict(portfolio_configuration=configuration, mode=mode, name=name, assets=assets, day=str(day), period=period, window=window,
                  rolling=rolling, horizon=horizon, model=model, decay=decay,
                  covariance=cov_method, management=management, rebalance=rebalance, sessions=ANNUALIZATION,
                  source=allocation.source if allocation else None,
                  optimized_at=allocation.optimized_at if allocation else None,
                  weight_cap=allocation.max_weight if allocation else None,
                  weights=allocation.weights.to_dict() if allocation else None)
    storage_key = f"volatility_analytics_{mode}"
    if run:
        state.pop(storage_key, None)
        outputs = {}
        errors = {}
        if allocation:
            if pd.isna(allocation.as_of) and not allocation.optimized_at:
                st.warning("The saved allocation has no dated inputs. Re-run its source optimization.")
                return
            available_date = max([t for t in (allocation.as_of, pd.Timestamp(allocation.optimized_at))
                                  if pd.notna(t)], key=lambda t:t.date()).date()
            if day < available_date:
                st.warning(f"These weights were unavailable on {day}. Choose {available_date} or later, "
                           "or run the source optimization with verified historical inputs.")
                return
        targets = [(name, tuple(assets))] if allocation else [(asset,(asset,)) for asset in assets]
        for label, symbols in targets:
            try:
                prices = market_prices(symbols, period, configuration)
                if configuration and len(symbols) > 1:
                    from src.hybrid_portfolio import hybrid_sessions

                    prices = prices.loc[hybrid_sessions(prices.index, configuration)]
                # Exclude today's bar, which may still be incomplete.
                cutoff = min(pd.Timestamp(day), pd.Timestamp(date.today())-pd.Timedelta(days=1))
                if allocation:
                    w = valid_weights(allocation.weights, prices)
                    prices = prices.loc[:,w.index]
                else:
                    w = None
                returns = complete_suffix(dated_returns(prices,cutoff))
                if len(returns)<window:
                    raise ValueError(f"Only {len(returns)} consecutive complete returns. Need {window}; "
                                     "choose a longer history or a shorter estimation window.")
                sample = returns.tail(window)
                series = reconstruct_returns(returns,w,rebalance) if allocation else returns.iloc[:,0]
                cov, sigma, contributions = target_risk(sample,w,cov_method,decay) if allocation else (None,None,None)
                fit_sample = series.tail(window)
                historical = fit_sample.std()*np.sqrt(ANNUALIZATION)
                curves = pd.DataFrame({"Rolling":series.rolling(rolling).std()*np.sqrt(ANNUALIZATION)})
                ewma_history = estimate(series,"EWMA",1,rolling,decay)[0]
                forecasts = {}
                conditional = {"EWMA": ewma_history}
                diagnostics = {}
                for candidate in MODELS:
                    try:
                        filtered, path, fitted = estimate(fit_sample,candidate,horizon,rolling,decay)
                        forecasts[candidate] = path
                        diagnostics[candidate] = fitted
                        if filtered is not None and candidate != "EWMA":
                            conditional[candidate] = filtered
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                        diagnostics[candidate] = {"error":str(exc)}
                outputs[label] = dict(series=series, asset_returns=sample, reconstruction_inputs=returns, curves=curves, forecasts=forecasts,
                                      conditional=conditional, diagnostics=diagnostics, historical=historical,
                                      cov=cov, sigma=sigma, contributions=contributions, validation=None)
            except Exception as exc:
                errors[label] = str(exc)
        state[storage_key] = (params,outputs,errors)
    result = state.get(storage_key)
    if not result or result[0] != params:
        if result:
            st.info("Run the analysis to apply the current settings.")
        methodology()
        return
    _, outputs, errors = result
    for asset, error in errors.items():
        st.warning(f"{asset} · {error}")
    if outputs:
        if not portfolio_mode:
            summary = pd.DataFrame({
                label:{"Historical annualized":output["historical"],
                       "Forecast annualized":annual_forecast(output["forecasts"].get(model))}
                for label, output in outputs.items()
            }).T
            st.dataframe(summary.style.format("{:.2%}", na_rep="—"))
            st.download_button("Download volatility summary CSV",summary.to_csv(),"volatility_summary.csv","text/csv")
        label = next(iter(outputs)) if portfolio_mode else st.selectbox("Inspect asset",list(outputs))
        render_result(outputs[label],dict(params, analyzed_series=label),metric)
    methodology()


def annual_forecast(path):
    return np.sqrt(path.mean()*ANNUALIZATION) if path is not None else np.nan


def plot_lines(frame,title,x,y):
    series = frame.dropna(axis=1,how="all")
    if series.empty:
        return
    fig = go.Figure()
    for name in series:
        fig.add_scatter(x=series.index,y=series[name],name=name,mode="lines")
    fig.update_layout(template="plotly_dark",title=title,title_font_color="#ffbf69",
                      xaxis_title=x,yaxis_title=y,hovermode="x unified")
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig)


def render_result(out,p,metric):
    source_date = out["series"].index[-1].date()
    st.caption(f"Forecast origin · {source_date} close | {p['horizon']} daily sessions | {p['model']}")
    if p["mode"] == "portfolio":
        st.caption("Historical portfolio returns are reconstructed from target weights and adjusted prices. "
                   "The reconstruction excludes trading costs and does not represent an executed portfolio.")
    path = out["forecasts"].get(p["model"])
    forecast = annual_forecast(path)
    cards = [("Historical Annualized Volatility",out["historical"],"blue"),
             ("Ex-ante Target Volatility",out["sigma"],"purple"),
             ("Forecast Volatility · Horizon",np.sqrt(path.sum()) if path is not None else None,"blue"),
             ("Annualized Forecast Volatility",forecast,"green")]
    for column,(label,value,color) in zip(st.columns(4),cards):
        with column:
            metric(label,f"{value:.2%}" if value is not None and np.isfinite(value) else "—",color)
    if np.isfinite(forecast):
        st.caption(f"Forecast less historical volatility · {(forecast-out['historical'])*100:+.2f} pp")
    rolling = out["curves"]["Rolling"].dropna()
    if len(rolling):
        percentile = (rolling <= rolling.iloc[-1]).mean()
        regime = "High" if percentile >= .75 else "Low" if percentile <= .25 else "Normal"
        st.caption(f"Rolling volatility percentile · {percentile:.0%} | Historical regime · {regime}")
    if out["conditional"].get(p["model"]) is not None:
        last = out["conditional"][p["model"]].iloc[-1]
        if np.isfinite(last):
            st.caption(f"Current filtered volatility · {np.sqrt(last*ANNUALIZATION):.2%} annualized")
    plot_lines(out["curves"][["Rolling"]].assign(**{"Historical estimate":out["historical"]}),
               "Historical & Rolling Volatility","Date","Annualized volatility")
    conditional = pd.DataFrame(out["conditional"]).mul(ANNUALIZATION).pow(.5)
    plot_lines(conditional,"Conditional Volatility","Date","Annualized volatility")
    paths = np.sqrt(pd.DataFrame(out["forecasts"],index=range(1,p["horizon"]+1))*ANNUALIZATION)
    plot_lines(paths,"Forward Volatility Forecast","Sessions ahead","Annualized volatility")
    if out["forecasts"]:
        st.dataframe(pd.Series({m:annual_forecast(v) for m,v in out["forecasts"].items()},
                               name="Annualized forecast").to_frame().style.format("{:.2%}"))
    for model,diagnostic in out["diagnostics"].items():
        if "error" in diagnostic:
            st.warning(f"{model} · {diagnostic['error']}")
    contribution = out["contributions"]
    if contribution is not None:
        st.subheader("Portfolio Risk Contributions")
        st.dataframe(contribution.style.format("{:.2%}",na_rep="—"))
        if out["sigma"]:
            fig=go.Figure(go.Bar(x=contribution.index,y=contribution["Percentage Risk Contribution"],
                                 marker_color=np.where(contribution["Percentage Risk Contribution"]<0,"#fa858e","#ffbf69")))
            fig.update_layout(template="plotly_dark",yaxis_tickformat=".0%",yaxis_title="Share of volatility")
            st.plotly_chart(fig)
        if out["forecasts"]:
            render_alerts(out["forecasts"])
    st.subheader("Volatility Model Comparison")
    a,b=st.columns(2)
    train=a.number_input("OOS training sessions", min_value=252,max_value=1260,value=p["window"],step=21)
    stride=b.selectbox("OOS step (sessions)",[5,21,63],index=1)
    if st.button("Validate all models out of sample"):
        out["validation"] = (train,stride,validate_models(out["series"], MODELS, p["horizon"],
                                                           train,stride,p["rolling"],p["decay"]))
    validation = out["validation"]
    if validation and validation[:2] == (train,stride):
        observations,scores,errors = validation[2]
        if scores.empty:
            st.info("Insufficient complete forecast windows. Increase history or reduce training length.")
        else:
            st.dataframe(scores)
            st.dataframe(observations.tail(30))
            st.download_button("Download forecast validation CSV", observations.to_csv(index=False),
                               "volatility_validation.csv","text/csv")
        for model,error in errors.items():
            st.warning(f"Validation · {model} · {error}")
    with st.expander("Research & Validation — Export Audit Data"):
        if st.button("Prepare Volatility Audit Data"):
            st.download_button("Download Volatility Audit Data",build_archive(out,p,source_date),
                               "volatility_audit.zip","application/zip")


def render_alerts(paths):
    with st.expander("Dynamic Risk Management — volatility alerts"):
        model=st.selectbox("Risk signal model",list(paths),index=list(paths).index("EWMA") if "EWMA" in paths else 0)
        target=st.number_input("Target annual volatility (%)",1.,50.,15.)
        current=st.number_input("Current invested exposure (%)",0.,100.,100.)
        cols=st.columns(3)
        caution=cols[0].number_input("Yellow from (%)",1.,100.,20.)
        warning=cols[1].number_input("Orange from (%)",1.,100.,25.)
        critical=cols[2].number_input("Red from (%)",1.,100.,30.)
        review=st.number_input("Review exposure gap (pp)",0.,50.,5.)
        rebalance=st.number_input("Rebalance review gap (pp)",1.,75.,10.)
        try:
            alert=risk_alert(annual_forecast(paths[model])*100,target,current,caution,warning,critical,review,rebalance)
            st.write(f"{alert['status']} · {alert['action']} · Exposure {alert['suggested_exposure']:.1f}% "
                     f"· Cash {alert['suggested_cash']:.1f}%")
        except ValueError as exc:
            st.warning(str(exc))


def build_archive(out,params,origin):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("parameters.json",json.dumps(dict(params,origin=str(origin),
                         diagnostics=out["diagnostics"],historical_annualized=out["historical"],
                         ex_ante_annualized=out["sigma"]),default=str,indent=2))
        tables={"returns":out["series"],"covariance_inputs":out["asset_returns"],
                "reconstruction_inputs":out["reconstruction_inputs"], "annual_covariance":out["cov"],"contributions":out["contributions"],
                "historical_volatility":out["curves"],"conditional_daily_variance":pd.DataFrame(out["conditional"]),
                "forecast_daily_variance":pd.DataFrame(out["forecasts"],index=range(1,params["horizon"]+1))}
        if out["validation"]:
            train,stride,(observed,scores,errors)=out["validation"]
            tables.update(validation=observed,validation_scores=scores)
            archive.writestr("validation_parameters.json",json.dumps(dict(train=train,stride=stride,errors=errors)))
        for name,table in tables.items():
            if table is not None:
                archive.writestr(f"{name}.csv",table.to_csv())
    return stream.getvalue()


def methodology():
    with st.expander("Methodology — Volatility Analytics & Forecasting"):
        st.write("La volatilità misura la dispersione dei rendimenti. Per l’annualizzazione usiamo 252 sedute. "
                 "Le serie storiche mostrano la dispersione osservata sul campione o sulla finestra mobile. "
                 "Per il portafoglio la serie è ricostruita con quantità fisse fra ribilanciamenti, "
                 "mentre la stima ex ante usa i pesi target e la covarianza degli asset.")
        st.write("EWMA dà più peso agli shock recenti e inizia dal quadrato del primo rendimento disponibile. "
                 "Le stime alla data t usano i rendimenti precedenti. GARCH, GJR ed EGARCH stimano la persistenza "
                 "della varianza e usano innovazioni Student-t. Shock ravvicinati possono produrre periodi "
                 "di volatilità elevata. Le curve GARCH storiche sono filtrate in-sample.")
        st.write("Il forecast guarda dalla chiusura indicata alle sedute successive. La volatilità di orizzonte "
                 "è la radice della somma delle varianze giornaliere previste, sotto l’ipotesi di rendimenti "
                 "non autocorrelati. Il confronto OOS valuta varianza e QLIKE sulle stesse finestre, "
                 "condizionatamente ai pesi selezionati.")
        st.write("Le correlazioni possono ridurre o amplificare il rischio aggregato. Il contributo marginale "
                 "misura la variazione della volatilità per unità di peso. Il prodotto per il peso dà il "
                 "contributo assoluto, che può essere negativo. La somma restituisce la volatilità ex ante. "
                 "La stima coincide con Monte Carlo per gli stessi rendimenti, pesi, covarianza e annualizzazione. "
                 "I forecast dipendono dalla finestra e dal modello e sono soggetti a errore di stima.")
