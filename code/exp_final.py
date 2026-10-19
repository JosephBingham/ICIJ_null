import sys, os, json; sys.path.insert(0, '/home/claude/icij')
from lib import *
import exp_confound as ec
corp = np.load('/home/claude/icij/corp_v2.npy')
F='/home/claude/icij/res/final.json'; OUT=json.load(open(F)) if os.path.exists(F) else {}
def P(*a): print(*a, flush=True)
def sv(): json.dump(OUT,open(F,'w'),indent=1)
L = np.array([str(x) for x in leak])

def matched(lab_fn, pool, cols, n_draws=5, scorer=None):
    aucs=[]
    for k in range(n_draws):
        rng=np.random.default_rng(400+k); lab=lab_fn(rng)
        trp,trn,tep,ten=[],[],[],[]
        totp=(pool&lab).sum()*0.3
        for l in LEAKS:
            Pp=rng.permutation(np.where((L==l)&pool&lab)[0]); Qq=rng.permutation(np.where((L==l)&pool&~lab)[0])
            ntr=int(.7*len(Pp)); trp.append(Pp[:ntr]); tep.append(Pp[ntr:])
            trn.append(Qq[:3*ntr]); ten.append(Qq[3*ntr:3*ntr+int(20000*len(Pp[ntr:])/totp)])
        trp,trn,tep,ten=map(np.concatenate,(trp,trn,tep,ten))
        if scorer is not None: s=scorer(trp); sp,sn=s[tep],s[ten]
        else:
            idx=np.r_[trp,trn]; y=np.r_[np.ones(len(trp)),np.zeros(len(trn))]
            s=fit_score(gb,X[idx][:,cols],y,X[np.r_[tep,ten]][:,cols]); sp,sn=s[:len(tep)],s[len(tep):]
        aucs.append(auc(sp,sn))
    return float(np.mean(aucs))
real=lambda rng: name_seed
perm=lambda rng: ec.permuted_labels(rng, LEAKS)

if 'corp_pool_incorpus' not in OUT:
    P('== IN-CORPUS, SOURCE-MATCHED, CORPORATE-NAMED POOL')
    pool = (name_seed|corp) & U
    OUT['corp_pool_incorpus']={k:dict(real=matched(real,pool,SETS[k]),perm=matched(perm,pool,SETS[k])) for k in ['connectivity','full8','full12']}
    P('  ',OUT['corp_pool_incorpus']); sv()
if 'corp_vs_person' not in OUT:
    P('== HOW WELL DOES STRUCTURE SEPARATE CORPORATE-NAMED FROM PERSON-NAMED OFFICERS? (unlabelled only, matched)')
    pool = U & ~name_seed
    OUT['corp_vs_person']={k:matched(lambda rng: corp, pool, SETS[k]) for k in ['connectivity','full8','full12']}
    P('  ',OUT['corp_vs_person']); sv()
if 'srbf_matched' not in OUT:
    P('== SRBF-INSPIRED PPR, IN-CORPUS WITH SOURCE-MATCHED NEGATIVES')
    import pandas as pd, scipy.sparse as sp
    r = pd.read_csv('/home/claude/icij/data/relationships.csv', usecols=['node_id_start','node_id_end','rel_type'],
                    dtype={'node_id_start':'int64','node_id_end':'int64','rel_type':str})
    g = r[r.rel_type.isin(['officer_of','registered_address','intermediary_of'])]
    nodes=np.unique(np.r_[g.node_id_start.values,g.node_id_end.values]); gi=pd.Series(np.arange(len(nodes)),index=nodes)
    a=gi.reindex(g.node_id_start.values).values; b=gi.reindex(g.node_id_end.values).values; N=len(nodes)
    A=sp.coo_matrix((np.ones(len(a)),(a,b)),shape=(N,N)).tocsr(); A=((A+A.T)>0).astype(float).tocsr()
    deg=np.asarray(A.sum(1)).ravel(); PT=(sp.diags(np.where(deg>0,1/np.maximum(deg,1),0))@A).T.tocsr(); og=gi.reindex(officers).values
    def ppr(seeds):
        s=np.zeros(N); s[og[seeds]]=1/len(seeds); x=s.copy()
        for _ in range(30): x=.85*(PT@x)+.15*s
        return x[og]
    OUT['srbf_matched']=dict(real=matched(real,U,None,3,scorer=ppr), perm=matched(perm,U,None,3,scorer=ppr))
    P('  ',OUT['srbf_matched']); sv()
if 'subsource_perm' not in OUT:
    P('== SUB-SOURCE LOO: PERMUTATION NULL (full8)')
    dom=['Bahamas Leaks','Offshore Leaks','Panama Papers','Pandora Papers - Trident Trust','Paradise Papers - Appleby',
         'Paradise Papers - Barbados corporate registry','Paradise Papers - Malta corporate registry']
    W=np.isin(node_src,dom); res={}
    for h in dom:
        aucs=[]
        for k in range(3):
            rng=np.random.default_rng(500+k); lab=name_seed.copy()
            for s in dom:
                if s==h: continue
                idx=np.where(node_src==s)[0]; lab[idx]=rng.permutation(lab[idx])
            tr=W&(node_src!=h); Pp=np.where(tr&lab)[0]; Q=np.where(tr&~lab)[0]; trn=rng.choice(Q,3*len(Pp),replace=False)
            TP=np.where((node_src==h)&name_seed)[0]; TQ=np.where((node_src==h)&~name_seed)[0]
            TQ=TQ if len(TQ)<=40000 else rng.choice(TQ,40000,replace=False)
            idx=np.r_[Pp,trn]; y=np.r_[np.ones(len(Pp)),np.zeros(len(trn))]
            s=fit_score(gb,X[idx][:,SETS['full8']],y,X[np.r_[TP,TQ]][:,SETS['full8']]); aucs.append(auc(s[:len(TP)],s[len(TP):]))
        res[h]=float(np.mean(aucs)); P(f'  {h:50s} perm {res[h]:.3f}')
    OUT['subsource_perm']=res; sv()
P('DONE')
