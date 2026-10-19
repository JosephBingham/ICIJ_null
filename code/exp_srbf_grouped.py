import sys, json; sys.path.insert(0,'/home/claude/icij')
from lib import *
import pandas as pd, scipy.sparse as sp
from exp_new import grouped_split
r = pd.read_csv('/home/claude/icij/data/relationships.csv', usecols=['node_id_start','node_id_end','rel_type'],dtype={'node_id_start':'int64','node_id_end':'int64','rel_type':str})
g = r[r.rel_type.isin(['officer_of','registered_address','intermediary_of'])]
nodes=np.unique(np.r_[g.node_id_start.values,g.node_id_end.values]); gi=pd.Series(np.arange(len(nodes)),index=nodes)
a=gi.reindex(g.node_id_start.values).values; b=gi.reindex(g.node_id_end.values).values; N=len(nodes)
A=sp.coo_matrix((np.ones(len(a)),(a,b)),shape=(N,N)).tocsr(); A=((A+A.T)>0).astype(float).tocsr()
deg=np.asarray(A.sum(1)).ravel(); PT=(sp.diags(np.where(deg>0,1/np.maximum(deg,1),0))@A).T.tocsr(); og=gi.reindex(officers).values
def ppr(seeds):
    s=np.zeros(N); s[og[seeds]]=1/len(seeds); x=s.copy()
    for _ in range(30): x=.85*(PT@x)+.15*s
    return x[og]
own=-np.arange(len(officers)).astype(float)-1
res={}
for nm,grp in [('random_officer',own),('dominant_intermediary',np.where(np.isnan(dom_int),own,dom_int)),('dominant_address',np.where(np.isnan(dom_addr),own,dom_addr))]:
    aucs=[]
    for k in range(3):
        trp,trn,tep,ten=grouped_split(grp)(np.random.default_rng(k)); s=ppr(trp); aucs.append(auc(s[tep],s[ten]))
    res[nm]=float(np.mean(aucs)); print(nm,round(res[nm],3),flush=True)
json.dump(res,open('/home/claude/icij/res/srbf_grouped.json','w'))
