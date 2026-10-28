import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import pandas as pd
from src.factor_currency import estimate_factor_returns
rng=np.random.default_rng(44)
dates=pd.bdate_range('2024-01-01',periods=310)
f=pd.DataFrame(rng.normal(0,0.008,(310,5)),index=dates,columns=['MKT-RF','SMB','HML','RMW','CMA']);f['RF']=0.0001
ret=0.0001+1.2*f['MKT-RF']+0.3*f['SMB']-0.2*f['HML']+rng.normal(0,0.001,310)
p=pd.DataFrame({'TEST':100*(1+ret).cumprod()},index=dates)
fx1=pd.DataFrame({'EURUSD=X':np.linspace(1,1.5,310)},index=dates)
fx2=pd.DataFrame({'EURUSD=X':np.linspace(1.5,1,310)},index=dates)
for model in ['ff3','ff5']:
 a,stats=estimate_factor_returns(p,f,fx1,market='USA',model=model,return_stats=True)
 b,_=estimate_factor_returns(p,f,fx2,market='USA',model=model,return_stats=True)
 assert a.equals(b)
 assert abs(stats.loc['TEST','Beta MKT']-1.2)<0.08
 print(model,'USA independent FX, beta',round(stats.loc['TEST','Beta MKT'],3))
 try:estimate_factor_returns(p,f,None,market='Borsa Italiana',model=model)
 except ValueError:pass
 else:raise AssertionError('Italian model should require FX')
print('All synthetic checks passed')
