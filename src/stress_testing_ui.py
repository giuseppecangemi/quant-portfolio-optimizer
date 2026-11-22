"""Streamlit stress testing workspace."""
import io
import json
import zipfile
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.factor_covariance import FACTOR_NAMES
from src.factor_currency import factors_for_backtest_signal
from src.factor_eur import convert_factors_to_eur
from src.forward_mc import calibration, fit_factor
from src.forward_mc_ui import load_prices
from src.stress_testing import (
    allocation_matrix, joint_shock, linear_attribution, paired_simulation, replay_assets,
    replay_attribution, shock_path,
)
from src.transaction_costs import PRESETS
from src.volatility_analytics import available_portfolios


EVENTS = {
    "Global Financial Crisis": ("2008-09-15","2009-03-09"),
    "Eurozone Sovereign Debt Crisis": ("2011-07-01","2011-12-30"),
    "COVID-19 Market Crash": ("2020-02-20","2020-03-23"),
    "Inflation & Rate Shock": ("2022-01-03","2022-10-12"),
}
METHODS = ["Historical Stress Testing","Factor Sensitivity","Stressed Monte Carlo"]


def factor_market_data(model, market, benchmark, cutoff, rf_daily):
    from src.fama_french import load_us_ff3, load_europe_ff3
    from src.fama_french_5 import load_us_ff5, load_europe_ff5

    if model == "CAPM":
        proxy = load_prices((benchmark,)).iloc[:,0].sort_index().loc[:cutoff]
        historical = proxy.pct_change(fill_method=None).rename("MKT-RF").to_frame().dropna()
        historical["MKT-RF"] -= rf_daily
        historical["RF"] = rf_daily
        return historical,proxy,{}
    choices = {"FF3":(load_europe_ff3,load_us_ff3),"FF5":(load_europe_ff5,load_us_ff5)}
    raw = factors_for_backtest_signal(choices[model][market=="USA"]().sort_index(),cutoff)
    fx = load_prices(("EURUSD=X",)).iloc[:,0].sort_index().loc[:cutoff] if market=="Borsa Italiana" else None
    historical = convert_factors_to_eur(raw,fx) if fx is not None else raw
    names = list(FACTOR_NAMES[model.lower()])
    return historical[[*names,"RF"]].dropna(),None,{model:(raw,fx)}


def covered_events(data, assets):
    dates = data.index
    covered = {}
    for name,(a,b) in EVENTS.items():
        if dates[0] >= pd.Timestamp(a) or dates[-1] < pd.Timestamp(b):
            continue
        try:
            if len(replay_assets(data,a,b,assets)) >= 10:
                covered[name]=(pd.Timestamp(a).date(),pd.Timestamp(b).date())
        except ValueError:
            continue
    return covered


def render_stress_testing(state, market, benchmark, rf, theme):
    portfolios = available_portfolios(state)
    if not portfolios:
        st.info("Run Portfolio Optimization, Factor Models or the ML comparison to obtain saved allocations.")
        return
    if market not in ("Borsa Italiana","USA"):
        st.info("Stress Testing supports Italian and US portfolios.")
        return
    currency = "EUR" if market=="Borsa Italiana" else "USD"
    with st.container(key="kangemi_controls_stress_common"):
        st.subheader("Common Configuration")
        selected = st.multiselect("Portfolio Strategies",list(portfolios),default=list(portfolios)[:2])
        if not selected:
            return
        assets = tuple(dict.fromkeys(t for name in selected for t,w in portfolios[name].weights.items() if w > 1e-10))
        if any(t.endswith(".MI") != (currency=="EUR") for t in assets):
            st.warning("Select portfolios in the current market and native currency.")
            return
        try:
            prices = load_prices(assets).reindex(columns=assets).sort_index()
            zone = "Europe/Rome" if currency=="EUR" else "America/New_York"
            prices = prices.loc[prices.index.date < pd.Timestamp.now(tz=zone).date()]
            latest = prices.dropna().index.max()
            if pd.isna(latest):
                raise ValueError("No common completed market valuation.")
        except Exception as exc:
            st.warning(f"Market history unavailable: {exc}")
            return
        a,b,c=st.columns(3)
        chosen = a.date_input("Valuation Date", latest.date(), min_value=prices.index[0].date(),max_value=pd.Timestamp.now(tz=zone).date())
        capital = b.number_input(f"Initial Capital ({currency})", min_value=1.,value=10000.)
        method = c.selectbox("Stress Testing Method",METHODS)
        market_sample = prices.loc[:str(chosen)].dropna()
        if market_sample.empty:
            st.warning("No common market data at this date.")
            return
        cutoff = market_sample.index[-1]
        st.caption(f"Completed valuation close · {cutoff.date()}. Saved allocations are checked against the selected date.")
        replay = st.radio("Replay mode",["Direct Asset Historical Replay","Factor Historical Replay"],horizontal=True) if method==METHODS[0] else None
        factor_needed = replay != "Direct Asset Historical Replay"
        model = st.selectbox("Factor Risk Model",["CAPM","FF3","FF5"],index=1) if factor_needed else None
        sessions = st.number_input("Trading Sessions per Year",min_value=1,max_value=366,value=252) if factor_needed else 252
        window = st.selectbox("Estimation Window (Sessions)",[252,504,756,1260],index=2) if factor_needed else None
        if method != METHODS[1]:
            management = st.radio("Portfolio Management",["Buy & Hold","Constant Weight Rebalancing"],horizontal=True)
            rebalance = st.selectbox("Rebalance every (sessions)",[1,21,63,126,252],index=1) if management!="Buy & Hold" else 0
            costs = st.selectbox("Transaction Costs",["None",*PRESETS])
        else:
            rebalance,costs=0,"None"

    with st.container(key="kangemi_controls_stress_scenario"):
        st.subheader("Scenario Configuration")
        name = st.text_input("Scenario name",method)
        history, proxy, inputs = None,None,{}
        if factor_needed:
            try:
                history,proxy,inputs = factor_market_data(model,market,benchmark,cutoff,rf/sessions)
            except Exception as exc:
                st.warning(f"Factor data unavailable: {exc}")
                return
            if history.empty:
                st.warning("No factor observations are available at this cutoff.")
                return
            factor_names = [c for c in history.columns if c!="RF"]
        else:
            factor_names=[]
        available = history if factor_needed else prices.loc[:cutoff]
        stress_horizon = 1
        sim_paths,seed,sim_steps = 1,42,1
        timing, path_type = 0,"One-day Shock"
        volatility,residual_vol,corr = 1.,1.,False
        covariance_method,decay,expected = "Sample",.94,"Factor-implied"
        residual_mode = "Zero residual shock"
        manual = {}
        quantile,start,end = None,None,None
        if method == METHODS[0]:
            if factor_needed:
                event_options = {
                    label:(pd.Timestamp(a).date(),pd.Timestamp(b).date())
                    for label,(a,b) in EVENTS.items()
                    if available.index[0] <= pd.Timestamp(a) <= pd.Timestamp(b) <= available.index[-1]
                    and len(available.loc[a:b]) >= 10
                }
            else:
                event_options = covered_events(available, list(available.columns))
            event = st.selectbox("Historical Scenario",["Custom Historical Window",*event_options])
            if event in event_options:
                start,end = event_options[event]
            else:
                a,b=st.columns(2)
                end = b.date_input("Scenario end",available.index[-1].date(),
                                   min_value=available.index[0].date(),max_value=available.index[-1].date())
                start = a.date_input("Scenario start",available.index[max(0,len(available)-21)].date(),
                                     min_value=available.index[0].date(),max_value=available.index[-1].date())
            if factor_needed:
                residual_mode = st.radio("Historical residuals",["Zero residual shock","Residual bootstrap"])
                if residual_mode=="Residual bootstrap":
                    seed = st.number_input("Random Seed",min_value=0,value=42)
            calibration_mode = event
        else:
            a,b,c=st.columns(3)
            stress_horizon = a.number_input("Stress duration / calibration horizon (sessions)", min_value=1,max_value=1260,value=21)
            calibration_mode=b.selectbox("Shock Calibration",["Custom Manual Shock","Historical Joint Scenario","Historical Quantile Calibration"])
            if calibration_mode=="Custom Manual Shock":
                st.caption("User-defined hypothetical scenario — no probability assigned")
                for column,factor in zip(st.columns(len(factor_names)),factor_names):
                    manual[factor]=column.number_input(f"{factor} cumulative shock (pp)",value=0.,step=1.)/100
            elif calibration_mode=="Historical Quantile Calibration":
                quantile=c.selectbox("Market lower percentile",[.10,.05,.01],index=1,format_func=lambda q:f"{q:.0%}")
            else:
                end=c.date_input("Historical joint window end",available.index[-1].date(),
                                 min_value=available.index[0].date(),max_value=available.index[-1].date())
                selected_window=history.loc[:str(end)].tail(stress_horizon)
                start=selected_window.index.min().date()
            if method==METHODS[2]:
                a,b,c=st.columns(3)
                years=a.selectbox("Simulation Horizon (Years)",[.25,.5,1,2,3,5],index=2)
                sim_steps=max(1,round(years*sessions))
                sim_paths=b.selectbox("Number of Simulations",[1000,5000,10000,20000],index=1)
                seed=c.number_input("Random Seed",min_value=0,value=42)
                a,b,c=st.columns(3)
                expected=a.selectbox("Expected Return Model",["Factor-implied","Historical Mean"])
                covariance_method=b.selectbox("Covariance Estimation",["Sample","EWMA","Ledoit–Wolf"])
                if covariance_method=="EWMA":
                    decay=b.number_input("EWMA lambda",min_value=.01,max_value=.999,value=.94)
                timing=c.number_input("Shock Timing (sessions after start)",min_value=0,max_value=sim_steps-1,value=0)
                path_type=st.selectbox("Shock path",["One-day Shock","Distributed Shock","Persistent Regime Shift"])
                a,b,c=st.columns(3)
                volatility=a.number_input("Factor Volatility Multiplier",min_value=.01,value=1.)
                residual_vol=b.number_input("Residual Volatility Multiplier",min_value=.01,value=1.)
                corr_eligible=len(history.tail(window))*.20>=max(30,len(factor_names)+2)
                corr=c.selectbox("Factor Correlations",["Baseline Correlations"]+
                                 (["Historical Stressed Correlations"] if corr_eligible else []))
                corr=corr=="Historical Stressed Correlations"
                if not corr_eligible:
                    c.caption("Insufficient history for stressed correlation calibration.")
        run=st.button("Run Stress Test",type="primary")
    config=dict(name=name,method=method,replay=replay,day=str(chosen),cutoff=str(cutoff.date()),
                strategies=selected,currency=currency,capital=capital,sessions=sessions,window=window,
                model=model,expected=expected,rebalance=rebalance,costs=costs,
                cost_assumptions=asdict(PRESETS[costs]) if costs in PRESETS else None,
                calibration=calibration_mode,horizon=stress_horizon,quantile=quantile,start=str(start) if start else None,
                end=str(end) if end else None,residual_mode=residual_mode,seed=seed,steps=sim_steps,paths=sim_paths,
                timing=timing,path_type=path_type,volatility_multiplier=volatility,
                residual_multiplier=residual_vol,historical_correlations=corr,covariance=covariance_method,
                decay=decay,risk_free=rf)
    if run:
        try:
            with st.spinner("Estimating exposures and applying common shocks…"):
                weights = allocation_matrix(portfolios,selected,chosen).reindex(assets)
                result = calculate(config,weights,prices,history,proxy,inputs,market,manual)
                state["stress_active"] = result
                archive=state.setdefault("stress_scenarios",{})
                archive[name or method]=(result["config"].copy(),result["summary"].copy(),result["weights"].copy())
                if len(archive)>20:
                    del archive[next(iter(archive))]
        except (ValueError, KeyError, RuntimeError, np.linalg.LinAlgError) as exc:
            st.error(str(exc))
    if state.get("stress_active"):
        st.caption("Results refer to the last completed stress run.")
        render_stress_results(state["stress_active"],theme)
    render_comparison(state.get("stress_scenarios",{}),theme)
    methodology()


def calculate(c,weights,prices,history,proxy,inputs,market,manual):
    out=dict(config=c.copy(),weights=weights,tables={},paths={},fans={})
    costs=PRESETS.get(c["costs"])
    if c["model"]:
        returns,fits=calibration(prices,c["cutoff"],c["window"],proxy,inputs,market,
                                  c["risk_free"]/c["sessions"])
        beta,f,rate,residual,labels=fit_factor(fits[c["model"]])
        mean=rate+beta @ f.mean(axis=0)
        out["tables"].update(betas=pd.DataFrame(beta,index=weights.index,columns=labels),
                             historical_factors=history,calibration_returns=returns,
                             residual_covariance=pd.DataFrame(np.cov(residual,rowvar=False).reshape(len(weights),len(weights)),
                                                              index=weights.index,columns=weights.index))
        if c["expected"]=="Historical Mean":
            mean=returns.mean().to_numpy()
    if c["method"]==METHODS[0]:
        start,end=c["start"],c["end"]
        if start>end:
            raise ValueError("Scenario start must precede its end.")
        components=None
        if c["model"]:
            episode=history.loc[start:end]
            if episode.empty:
                raise ValueError("No available factor observations in the requested historical window.")
            residual_draw=np.zeros((len(episode),len(weights)))
            if c["residual_mode"]=="Residual bootstrap":
                residual_draw=residual[np.random.default_rng(c["seed"]).integers(len(residual),size=len(episode))]
            scenario=pd.DataFrame(episode[labels].to_numpy() @ beta.T+episode.RF.to_numpy()[:,None]+residual_draw,
                                  index=episode.index,columns=weights.index)
            factor_parts=[pd.DataFrame(np.outer(episode[label],beta[:,j]),index=episode.index,columns=weights.index)
                          for j,label in enumerate(labels)]
            rates=pd.DataFrame(np.broadcast_to(episode.RF.to_numpy()[:,None],scenario.shape),
                               index=episode.index,columns=weights.index)
            components=([*labels,"Risk Free","Residual"],[*factor_parts,rates,
                         pd.DataFrame(residual_draw,index=episode.index,columns=weights.index)])
            out["tables"]["historical_factor_episode"]=episode
        else:
            scenario=replay_assets(prices,start,end,list(weights.index))
        nav,assets,factors=replay_attribution(scenario,weights,c["capital"],c["rebalance"],costs,components)
        total=nav.iloc[-1]/c["capital"]-1
        out["summary"]=pd.DataFrame({"Portfolio Return":total,"Portfolio Loss":-total*c["capital"],
            "Maximum Drawdown":(nav/nav.cummax()-1).min(),"Worst Daily Return":nav.pct_change().min()})
        out["tables"].update(scenario_asset_returns=scenario,asset_attribution=assets,factor_attribution=factors)
        out["paths"]["historical"]=nav
        out["config"].update(actual_start=str(scenario.index[0].date()),actual_end=str(scenario.index[-1].date()),steps=len(scenario))
    else:
        if manual:
            shock=pd.Series(manual).reindex(labels)
        else:
            shock,start,end,count=joint_shock(history[labels],c["horizon"],start=c["start"],end=c["end"],quantile=c["quantile"])
            out["config"].update(shock_start=start,shock_end=end,calibration_windows=count)
        sensitivity,exposures,factor,asset,asset_shock=linear_attribution(weights,beta,shock,mean,c["horizon"],c["capital"])
        out["tables"].update(shock=shock,factor_exposures=exposures,factor_attribution=factor,
                             asset_attribution=asset,asset_shock=asset_shock)
        if c["method"]==METHODS[1]:
            out["summary"]=sensitivity
            out["config"]["steps"]=c["horizon"]
        else:
            delta=shock_path(shock,c["steps"],c["timing"],c["horizon"],c["path_type"])
            active=shock_path([1],c["steps"],c["timing"],c["horizon"],c["path_type"])[:,0]>0
            base,stress,mats=paired_simulation(weights,mean,beta,f,residual,delta,capital=c["capital"],
                paths=c["paths"],seed=c["seed"],method=c["covariance"],decay=c["decay"],sessions=c["sessions"],
                rf=c["risk_free"],rebalance=c["rebalance"],costs=costs,factor_multiplier=c["volatility_multiplier"],
                residual_multiplier=c["residual_multiplier"],historical_correlations=c["historical_correlations"],active=active)
            b,s=base["metrics"],stress["metrics"]
            out["summary"]=pd.DataFrame({"Baseline Mean Return":b["Mean Return"],"Stressed Mean Return":s["Mean Return"],
                "Return Impact":s["Mean Return"]-b["Mean Return"],"Baseline Volatility":b["Annualized Volatility"],
                "Stressed Volatility":s["Annualized Volatility"],"Probability of Loss under Stress":s["Probability of Loss"],
                "Stressed VaR 95%":s["Horizon VaR 95%"],"Stressed ES 95%":s["Horizon ES 95%"],
                "Median Stressed Maximum Drawdown":s["Median Max Drawdown"],"P5 Final Value under Stress":s["P5 Final Value"]})
            for title, result in (("baseline",base),("stress",stress)):
                out["fans"][title]=result["fans"]
                out["paths"][title]=result["paths"]
                out["tables"][f"{title}_terminal_returns"]=result["final"]
                out["tables"][f"{title}_metrics"]=result["metrics"]
            out["tables"].update(daily_shock_path=pd.DataFrame(delta,columns=labels),linear_sensitivity=sensitivity,
                                 expected_daily=pd.Series(mean,index=weights.index))
            out["config"]["stressed_observations"]=mats.pop("stressed_observations")
            out["tables"].update({key:pd.DataFrame(value,index=weights.index if key=="residual_cov" else labels,
                                                 columns=weights.index if key=="residual_cov" else labels)
                                  for key,value in mats.items()})
    return out


def draw(fig,theme,title,y=None):
    fig.update_layout(title=title,yaxis_title=y,hovermode="x unified")
    theme(fig)
    st.plotly_chart(fig)


def render_stress_results(out,theme):
    c=out["config"]
    tables=out["tables"]
    summary=out["summary"]
    st.subheader(f"Stress Results · {c['name']}")
    st.caption(f"{c['method']} · Valuation {c['cutoff']} · {c['steps']} sessions · {c['currency']}")
    if c["method"]==METHODS[0]:
        st.caption(f"Historical window · {c['actual_start']} → {c['actual_end']}")
    metric=st.selectbox("Stress resilience ranking",list(summary),key="stress_rank")
    ascending=("Loss" in metric or "VaR" in metric or " ES " in metric or "Volatility" in metric)
    ranked=summary.sort_values(metric,ascending=ascending)
    money={"Portfolio Loss","Impact EUR","Stressed VaR 95%","Stressed ES 95%","P5 Final Value under Stress"}
    formats={col:"{:,.2f}" if col in money else "{:.2%}" for col in ranked}
    st.dataframe(ranked.style.format(formats))
    if c["method"]==METHODS[0]:
        path=out["paths"]["historical"]
        chart=go.Figure([go.Scatter(x=path.index,y=path[name],name=name) for name in path])
        draw(chart,theme,"Historical Stress NAV",f"NAV ({c['currency']})")
    elif c["method"]==METHODS[2]:
        st.caption("Loss probabilities are conditional on the modeled stress scenario.")
        strategy=st.selectbox("Inspect strategy",list(summary.index),key="stress_inspect")
        fan,dist=go.Figure(),go.Figure()
        for key,color,fill in (("baseline","#7bcaff","rgba(123,202,255,.12)"),
                               ("stress","#ff8490","rgba(255,132,144,.12)")):
            values=out["fans"][key][strategy]
            for low,high,band in ((0,4,"P5–P95"),(1,3,"P25–P75")):
                fan.add_scatter(y=values[high],line=dict(width=0),showlegend=False)
                fan.add_scatter(y=values[low],line=dict(width=0),fill="tonexty",fillcolor=fill,name=f"{key} {band}")
            fan.add_scatter(y=values[2],name=f"{key} median",line=dict(color=color))
            terminal=tables[f"{key}_terminal_returns"][strategy]
            dist.add_histogram(x=(1+terminal)*c["capital"],name=key,histnorm="probability density",
                               marker_color=color,opacity=.6)
        fan.add_hline(y=c["capital"],line_dash="dash")
        draw(fan,theme,"Baseline vs Stress Fan Chart",f"NAV ({c['currency']})")
        dist.update_layout(barmode="overlay",xaxis_title=f"Terminal NAV ({c['currency']})")
        draw(dist,theme,"Terminal Value Distribution")
    st.subheader("Risk Attribution")
    if c["method"]==METHODS[2]:
        st.caption("Linear sensitivity at initial weights. The table does not reconcile simulated terminal P&L.")
    if "factor_exposures" in tables:
        st.dataframe(tables["factor_exposures"].style.format("{:.3f}"))
    factors=tables["factor_attribution"]
    if not factors.empty:
        historical = c["method"]==METHODS[0]
        factors = factors.T if historical else factors
        st.dataframe(factors.style.format("{:+.2%}"))
        selected=st.selectbox("Factor attribution strategy",list(factors.index))
        values=factors.loc[selected]
        waterfall=go.Figure(go.Waterfall(x=[*values.index,"Total"],y=[*values,0],
                                         measure=["relative"]*len(values)+["total"]))
        waterfall.update_yaxes(tickformat=".1%")
        draw(waterfall,theme,"Factor Attribution")
    asset=tables["asset_attribution"]
    choice=st.selectbox("Asset attribution strategy",list(asset.columns))
    details=pd.DataFrame({"Portfolio Contribution":asset[choice],"Contribution in "+c["currency"]:asset[choice]*c["capital"],
                          "Weight":out["weights"][choice]})
    if "asset_shock" in tables:
        details["Factor Shock Return"]=tables["asset_shock"]
    st.dataframe(details.style.format({col:"{:,.2f}" if col.startswith("Contribution in ") else "{:.2%}"
                                      for col in details},na_rep="—"))
    with st.expander("Research & Validation — Export Audit Data"):
        if st.button("Prepare Stress Audit Data"):
            st.download_button("Download Stress Audit Data",build_archive(out,ranked),
                               "stress_testing_audit.zip","application/zip")


def build_archive(out,ranking):
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,"w",compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("scenario.json",json.dumps(out["config"],indent=2))
        tables=dict(out["tables"],summary=out["summary"],weights=out["weights"],ranking=ranking)
        for name,table in tables.items():
            z.writestr(f"{name}.csv",table.to_csv())
        for name,paths in out["paths"].items():
            if isinstance(paths,pd.DataFrame):
                z.writestr(f"{name}_nav.csv",paths.to_csv())
            else:
                binary=io.BytesIO()
                np.save(binary,paths,allow_pickle=False)
                z.writestr(f"{name}_nav.npy",binary.getvalue())
        z.writestr("README.txt","NAV NPY shape: paths × sessions (including initial capital) × strategies, "
                    "ordered as in weights.csv. Returns and attribution are in decimals. Amounts use native currency.")
    return payload.getvalue()


def render_comparison(scenarios,theme):
    if len(scenarios)<2:
        return
    with st.expander("Scenario Comparison"):
        names=st.multiselect("Scenarios",list(scenarios),default=list(scenarios)[-2:])
        if len(names)<2:
            return
        baseline,first,base_weights=scenarios[names[0]]
        fields=["method","steps","currency","sessions","cutoff","capital","rebalance","costs","model","replay",
                "expected","covariance","window","risk_free","day"]
        if any(any(scenarios[n][0][k]!=baseline[k] for k in fields)
               or not scenarios[n][2].equals(base_weights) for n in names[1:]):
            st.info("Choose scenarios with the same method, valuation, horizon, currency, allocations and management assumptions.")
            return
        metric=st.selectbox("Heatmap metric",list(first))
        result=pd.DataFrame({name:scenarios[name][1][metric] for name in names})
        st.dataframe(result)
        chart=go.Figure(go.Heatmap(z=result.to_numpy(),x=result.columns,y=result.index,
                                  colorscale="RdBu",zmid=0))
        draw(chart,theme,"Stress Scenario Heatmap")


def methodology():
    with st.expander("Methodology — Stress Testing"):
        st.write("Lo stress testing misura la risposta a scenari specifici. La simulazione prospettica standard "
                 "descrive invece una distribuzione condizionata alla calibrazione senza lo shock aggiunto. "
                 "Il replay applica un episodio passato ai portafogli selezionati e ha natura controfattuale.")
        st.write("CAPM, FF3 e FF5 stimano i beta con regressioni su rendimenti semplici allineati. "
                 "Il buffer di due mesi per i fattori French approssima i tempi di pubblicazione. "
                 "I dati vendor possono essere rivisti. Il replay fattoriale usa beta alla data di valutazione "
                 "e residui nulli o ricampionati congiuntamente. La sensibilità lineare usa pesi iniziali "
                 "e shock cumulati additivi. Non coincide in generale con il NAV composto.")
        st.write("I quantili selezionano una finestra osservata in base alla coda del mercato e mantengono "
                 "il vettore congiunto. Le finestre possono sovrapporsi. Uno shock cumulato è ripartito "
                 "in quote additive sull’intervallo scelto. Per il regime persistente l’intervallo termina "
                 "alla fine della simulazione. I moltiplicatori agiscono sulla volatilità, quindi la "
                 "varianza scala con il loro quadrato.")
        st.write("La correlazione stressata dei fattori è stimata nel quintile peggiore di mercato, "
                 "mantenendo le deviazioni standard prima del moltiplicatore. Baseline e Stress "
                 "usano gli stessi normali standard. Il rischio residuo conserva la covarianza cross-asset. "
                 "I forecast sono condizionati ai parametri stimati e non quantificano la probabilità della crisi. "
                 "Liquidità, gap di esecuzione e rischi fuori modello possono amplificare le perdite.")
