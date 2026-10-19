import sys, json, numpy as np; sys.path.insert(0,'/home/claude/icij')
from lib import *
dom=['Bahamas Leaks','Offshore Leaks','Panama Papers','Pandora Papers - Trident Trust','Paradise Papers - Appleby',
     'Paradise Papers - Barbados corporate registry','Paradise Papers - Malta corporate registry']
W=np.isin(node_src,dom); res={'real_bal':{},'perm_bal':{}}
for h in dom:
    tr=W&(node_src!=h)
    res['real_bal'][h]=run_draws(transfer_split(tr,node_src==h,name_seed,balance_by=node_src),SETS['full8'],3)['mean']
    a=[]
    for k in range(3):
        rng=np.random.default_rng(600+k); lab=name_seed.copy()
        for s in dom:
            if s==h: continue
            i=np.where(node_src==s)[0]; lab[i]=rng.permutation(lab[i])
        trd=[s for s in dom if s!=h]; m=min((lab&(node_src==s)).sum() for s in trd)
        Pp=np.concatenate([rng.choice(np.where(lab&(node_src==s))[0],m,replace=False) for s in trd])
        Q=np.concatenate([rng.choice(np.where(~lab&(node_src==s))[0],3*m,replace=False) for s in trd])
        TP=np.where((node_src==h)&name_seed)[0]; TQ=np.where((node_src==h)&~name_seed)[0]
        TQ=TQ if len(TQ)<=40000 else rng.choice(TQ,40000,replace=False)
        sc=fit_score(gb,X[np.r_[Pp,Q]][:,SETS['full8']],np.r_[np.ones(len(Pp)),np.zeros(len(Q))],X[np.r_[TP,TQ]][:,SETS['full8']])
        a.append(auc(sc[:len(TP)],sc[len(TP):]))
    res['perm_bal'][h]=float(np.mean(a))
    print(f"{h:50s} balanced real {res['real_bal'][h]:.3f}  balanced null {res['perm_bal'][h]:.3f}",flush=True)
json.dump(res,open('/home/claude/icij/res/balanced_sub.json','w'),indent=1)
