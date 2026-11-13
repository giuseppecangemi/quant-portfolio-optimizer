import json, zipfile, sys, hashlib
import numpy as np
import pandas as pd
from src.monte_carlo import aligned_returns, simulate
p=sys.argv[1]
print("sha256",hashlib.sha256(open(p,"rb").read()).hexdigest())
with zipfile.ZipFile(p) as z:
    m=json.load(z.open("metadata.json"))
    r=pd.read_csv(z.open("calibration_returns.csv"),index_col=0,parse_dates=True)
    closed=pd.to_datetime(["2025-12-24","2025-12-25","2025-12-26","2025-12-31","2026-01-01","2026-04-03","2026-04-06","2026-05-01"])
    cal=pd.bdate_range(r.index[0]-pd.Timedelta(days=3),r.index[-1]).difference(closed)
    eq={}
    for s in m["strategies"]:
        i=s["id"]; label=s["label"]
        nav=pd.read_csv(z.open(f"nav_{i}.csv"),index_col=0,parse_dates=True)
        tr=pd.read_csv(z.open(f"trades_{i}.csv"),parse_dates=["Rebalance Date"]).set_index("Rebalance Date")
        w=pd.read_csv(z.open(f"weights_{i}.csv"),index_col=0,parse_dates=True)
        v=nav["Portfolio Value"].copy(); c=s["initial_capital"]; v.attrs["initial_capital"]=c; eq[label]=v
        a=v/v.shift()-1; a.iloc[0]=v.iloc[0]/c-1
        costs=tr["Total Cost"].reindex(v.index,fill_value=0)
        pretrade=tr["Traded Notional"]/tr["Executed Turnover"]
        print(label,len(v),v.index[0].date(),v.index[-1].date(),"duplicate",v.index.duplicated().sum(),"sorted",v.index.is_monotonic_increasing,"invalid",sum(~np.isfinite(v)|(v<=0)),"missing",list(cal[cal>=v.index[0]].difference(v.index).date),"extra",list(v.index.difference(cal).date),"daily err",abs(a-nav["Daily Return"]).max(),"cost",costs.sum(),"cost err",abs(costs-nav["Transaction Cost"]).max(),"trade NAV err",abs(pretrade-tr["Total Cost"]-v.reindex(tr.index)).max(),"weight sum err",abs(w.sum(axis=1)-1).max(),"weight dates match",w.index.equals(tr.index),"end net/gross",v.iloc[-1],nav["Gross Portfolio Value"].iloc[-1])
    a=aligned_returns(eq)
    assert r.index.equals(a.index)
    print("calibration max error",np.max(abs(r-a).to_numpy()),"n",len(r))
    # Recompute on an independent calendar, with no imputation.
    official=pd.concat(eq,axis=1).reindex(cal).pct_change(fill_method=None).dropna()
    print("official excluded",list(r.index.difference(official.index).date))
    print(pd.DataFrame(dict(mean=r.mean(),sd=r.std(),skew=r.skew(),kurt=r.kurt(),acf=r.apply(lambda x:x.autocorr()),sq_acf=(r*r).apply(lambda x:x.autocorr()),min=r.min(),max=r.max())).round(5).to_string())
    print("corr",r.corr().to_numpy()[np.triu_indices(len(eq),1)].min(), "condition",np.linalg.cond(r.cov()),"half cov difference",np.linalg.norm(r.iloc[:len(r)//2].cov()-r.iloc[len(r)//2:].cov())/np.linalg.norm(r.cov()))
cfg={k:v for k,v in m["config"].items() if k!="sessions_per_year"}
metrics,_,final,paths=simulate(a,**cfg,return_paths=True)
pd.testing.assert_frame_equal(final,simulate(a,**cfg)[2])
idx=(np.random.default_rng(cfg["seed"]).integers(0,len(a)-4,size=(1000,51))[:,:,None]+np.arange(5)).reshape(1000,-1)[:,:252]
for j,label in enumerate(a):
    x=a[label].to_numpy()[idx]
    wealth=np.cumprod(1+x,axis=1)*eq[label].iloc[0]
    np.testing.assert_allclose(wealth[:,-1],final[label],rtol=1e-13)
    pnl=np.sort(wealth[:,-1]-10000)
    np.testing.assert_allclose(metrics.loc[label,"ES 95% (€)"],max(0,-pnl[:50].mean()),atol=1e-8)
    peak=np.maximum.accumulate(np.column_stack([np.full(len(x),10000),wealth]),axis=1)
    dd=np.minimum((wealth/peak[:,1:]-1).min(axis=1),eq[label].iloc[0]/10000-1)
    np.testing.assert_allclose(metrics.loc[label,"Probability MDD ≤ -20% (%)"],100*np.mean(dd<=-.2))
    np.testing.assert_allclose(metrics.loc[label,"Probability of loss (%)"],100*np.mean(pnl<0))
    np.testing.assert_allclose(metrics.loc[label,"Daily ES 95% (€)"],max(0,-10000*np.sort(x.ravel())[:12600].mean()))
print(metrics.round(2).to_string())
base=metrics["Median final (€)"].rank()
for tag,sample,params in [
    ("iid",a,dict(method="IID bootstrap")),("gauss",a,dict(method="Multivariate Gaussian")),
    ("seed43",a,dict(seed=43)),("paths5000",a,dict(n_paths=5000)),
    *[(f"L{k}",a,dict(block_size=k)) for k in (3,10,20)],
    ("first",a.iloc[:len(a)//2],{}),("second",a.iloc[len(a)//2:],{}),("last75",a.iloc[len(a)//4:],{}),
]:
    out=simulate(sample,**(cfg|params))[0]
    print(tag,"rank rho",round(base.corr(out["Median final (€)"].rank()),4),
          "median delta max",round(abs(out["Median final (€)"]-metrics["Median final (€)"]).max(),2),
          "loss pp delta max",round(abs(out["Probability of loss (%)"]-metrics["Probability of loss (%)"]).max(),2))
