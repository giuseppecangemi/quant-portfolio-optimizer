import sys,types
sys.modules['yfinance']=types.ModuleType('yfinance')
sys.path.insert(0,'/mnt/data/qpo_work/quant-portfolio-optimizer')
import numpy as np,pandas as pd
from src.screening import screen_prices_point_in_time,screen_quant_point_in_time,score_quant_candidates
idx=pd.bdate_range('2024-10-01',periods=253)
prices=pd.DataFrame({f'T{i}':np.linspace(100+i,150+i*2,253)*(1+.002*np.sin(np.arange(253)+i)) for i in range(6)},index=idx)
ranking,sel=screen_prices_point_in_time(prices.iloc[:252],top_n=3)
assert len(sel)==3 and ranking['Selected'].sum()==3
assert set(ranking.loc[ranking.Selected,'Ticker'])==set(sel)
assert (ranking['Components Available']==2).all()
assert ranking['Value Score'].isna().all() and ranking['Quality Score'].isna().all()
r2,s2=screen_quant_point_in_time(prices.iloc[:252],prices.iloc[252],idx[252],None,top_n=3)
assert sel==s2
assert np.allclose(ranking['Quant Score'],r2['Quant Score'])
assert len(screen_prices_point_in_time(prices.iloc[:251],top_n=3)[1])==0
print('PASS: 252-price PIT, selected flags, two factors, no SEC, no 6-1 fallback')
