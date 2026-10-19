import sys, os, json; sys.path.insert(0, '/home/claude/icij')
import lib
from lib import *
from exp_confound import transfer_custom  # reuse (runs nothing new: confound.json cached)
corp = np.load('/home/claude/icij/corp_v2.npy')
F='/home/claude/icij/res/strata2.json'; OUT={}
def P(*a): print(*a, flush=True)
P('== CORPORATE-NAMED POOL ONLY (positives vs corporate-named unlabelled)')
res={'corp_share_pos':{l:float(corp[(leak==l)&name_seed].mean()) for l in LEAKS},
     'corp_share_unl':{l:float(corp[(leak==l)&~name_seed].mean()) for l in LEAKS}}
# restrict the pool: positives are kept whatever their suffix; unlabelled must be corporate-named
keep = name_seed | corp
lib_leak = leak.copy()
leak_c = np.where(keep, leak, None)
import exp_confound as ec
ec.leak = leak_c
res['loo']={}
for h in LEAKS:
    tl=[l for l in LEAKS if l!=h]
    res['loo'][h]=dict(real=ec.transfer_custom(tl,h,SETS['full8'],4), perm=ec.transfer_custom(tl,h,SETS['full8'],4,permute=True),
                       bal=ec.transfer_custom(tl,h,SETS['full8'],4,balanced=True))
    P(f"  {h:13s} real {res['loo'][h]['real']['mean']:.3f} perm {res['loo'][h]['perm']['mean']:.3f} balanced {res['loo'][h]['bal']['mean']:.3f}")
json.dump(res,open(F,'w'),indent=1); P('DONE')
