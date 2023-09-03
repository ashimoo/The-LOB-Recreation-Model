from joblib import Parallel, delayed
from run_model import main
from itertools import combinations

combined=[]
for dataset in ['MSFT','INTC','JPM']:
    for side in ['bid','ask']:
        for main_module in ['ode','simple']:
            for seed in range(5):
                if main_module == 'attention':
                    combined.append((dataset,side,main_module,False,False,seed))
                else:
                    combined.append((dataset,side,main_module,True,True,seed))

Parallel(n_jobs=18)(delayed(main)(dataset=d, side=s, main_module=m, HC=h, WS=w, seed=seed, gpu=0) for i,(d,s,m,h,w,seed) in enumerate(combined))