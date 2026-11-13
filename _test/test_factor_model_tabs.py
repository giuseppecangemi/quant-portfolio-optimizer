import ast
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
source = (ROOT / "app_core.py").read_text()
tree=ast.parse(source)
names={'render_fama_french_model','render_fama_french_result','factor_estimation_caption','download_portfolio_weights'}
defs='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
app=AppTest.from_string('''
import streamlit as st
import pandas as pd
from unittest.mock import patch
st.session_state.setdefault("calls", [])
def get_prices(tickers, **kwargs):
    st.session_state.calls.append("prices")
    return pd.DataFrame({"A":[100,101]})
def estimate_factor_returns(*args, **kwargs):
    return pd.Series({"A":.1}),pd.DataFrame({"Observations":[811],"Beta MKT":[1.]},index=["A"])
def covariance_matrix(*args):
    st.session_state.calls.append("historical")
    return pd.DataFrame([[.04]],index=["A"],columns=["A"])
def factor_covariance(*args,**kwargs):
    st.session_state.calls.append("factor")
    return pd.DataFrame([[.09]],index=["A"],columns=["A"])
load_us_ff3=load_us_ff5=load_europe_ff3=load_europe_ff5=lambda:pd.DataFrame(index=range(9458))
calculate_returns=lambda prices: prices
validate_portfolio=lambda *args:True
optimize_maximum_sharpe=lambda *args,**kwargs:pd.Series({"A":1.})
portfolio_return=lambda weights,expected:float(weights @ expected)
portfolio_volatility=lambda weights,cov:float(cov.iloc[0,0]**.5)
prepare_portfolio_table=lambda *args:pd.DataFrame({"Invested":[100]})
colored_metric=render_diversification_var_kpis=portfolio_allocation_title=display_portfolio_table=lambda *args:None
get_company_name=lambda name:name
def capture_portfolio_configuration(tickers, market):
    return dict(universe=list(tickers), market=market, selection_mode="Manual Selection")
from src import trading_signals_ui
factor_tab, trading_tab = st.tabs(["Factor Models","Trading Signals"], key="main_navigation", on_change="rerun")

'''+defs+'''
with factor_tab, patch("src.factor_covariance.estimate_factor_covariance",factor_covariance):
    render_fama_french_model("FF3",["A"],"USA",100,1,0,True)
    render_fama_french_model("FF5",["A"],"USA",100,1,0,True)
if trading_tab.open:
    with trading_tab, patch.object(trading_signals_ui, "open_store", return_value=None):
        trading_signals_ui.render(colored_metric)
''').run()
assert not app.exception,app.exception
assert [b.label for b in app.button]==['Run FF3','Run FF5']
assert app.session_state['calls']==[]
app.button[0].click().run()
assert not app.exception and app.session_state['calls']==['prices','historical']
assert any('FF3 Historical Cov' in m.value for m in app.markdown),app.exception
assert app.session_state['ff3_results']['volatility']==.2
assert any('FF3 Historical Cov' in m.value for m in app.markdown)
app.get('button_group')[0].set_value('Factor Cov').run()
assert not app.exception and app.session_state['calls']==['prices','historical']
assert any('FF3 Historical Cov' in m.value for m in app.markdown)
app.button[0].click().run()
assert not app.exception and app.session_state['calls']==['prices','historical','prices','factor'],app.exception
assert app.session_state['ff3_factor_results']['volatility']==.3
app.get('button_group')[0].set_value('Historical').run()
assert app.session_state['calls']==['prices','historical','prices','factor']
assert any('FF3 Historical Cov' in m.value for m in app.markdown)
assert any('FF3 Factor Cov' in m.value for m in app.markdown)
next(b for b in app.button if b.label == "Run FF5").click().run()
assert not app.exception and 'ff5_results' in app.session_state
print('PASS: explicit Run, covariance branch, both results stay visible and no fetch on selector changes')

assert len(app.get("download_button")) == 3

next(b for b in app.button if b.key == "stage_trading_ff3_factor_cov").click().run()
assert not app.exception, app.exception
assert app.session_state["main_navigation"] == "Trading Signals"
assert app.session_state["ts_workspace"] == "Activate"
assert app.session_state["ts_portfolio_type"] == "Historical Activation"
draft = app.session_state["trading_draft"]
assert draft["configuration"]["return_model"] == "FF3"
assert draft["configuration"]["covariance_method"] == "Factor"
assert draft["configuration"]["universe"] == ["A"]
assert len(app.json) == 1
app.run()
assert not app.exception and app.session_state["trading_draft"] == draft
assert len(app.json) == 1
print("PASS: Factor Models handoff, navigation and configuration after rerun")
