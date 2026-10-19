"""Build the officer-level dataset for Paper B from the public ICIJ CSV export.

Outputs pre.npz / pre_meta.pkl with, for every officer that has >=1 non-beneficial-
ownership officer_of edge:
  - 8 tabular features (+ alternative address definitions for calibration)
  - 4 higher-order graph features
  - leak assignment by plurality officer_of edge source (the paper's rule) and purity
  - leak / sub-source from the officer node's own sourceID (extension; recovers Pandora)
  - name-derived proxy label, functional (underlying) label
  - person vs corporate flag, dominant intermediary / address groups
"""
import re, time, pickle
import numpy as np, pandas as pd, scipy.sparse as sp

T = time.time()
def log(*a): print(f"[{time.time()-T:6.1f}s]", *a, flush=True)
D = 'data/'

# ------------------------------------------------------------------ load
r = pd.read_csv(D+'relationships.csv', usecols=['node_id_start','node_id_end','rel_type','link','sourceID'],
                dtype={'node_id_start':'int64','node_id_end':'int64','rel_type':str,'link':str,'sourceID':str})
for c in ['rel_type','link','sourceID']: r[c] = r[c].astype('category')
off = pd.read_csv(D+'nodes-officers.csv', usecols=['node_id','name','sourceID'],
                  dtype={'node_id':'int64','name':str,'sourceID':str})
ent = pd.read_csv(D+'nodes-entities.csv', usecols=['node_id','jurisdiction','sourceID','service_provider'],
                  dtype={'node_id':'int64','jurisdiction':str,'sourceID':str,'service_provider':str})
log('loaded', len(r), len(off), len(ent))

def leak_of(s):
    if not isinstance(s, str): return None
    if s.startswith('Panama Papers'): return 'Panama'
    if s.startswith('Paradise Papers'): return 'Paradise'
    if s.startswith('Offshore Leaks'): return 'OffshoreLeaks'
    if s.startswith('Bahamas Leaks'): return 'Bahamas'
    if s.startswith('Pandora Papers'): return 'Pandora'
    return None

# ------------------------------------------------------------------ officer_of edges
offset = set(off.node_id.values)
oo = r[(r.rel_type == 'officer_of') & r.node_id_start.isin(offset)].copy()
linkl = oo.link.astype(str).str.lower().str.strip()
oo['bo'] = linkl.str.contains('beneficial|beneficiary', regex=True).values
oo['linkn'] = linkl.str.replace(r'\s+of$', '', regex=True).values   # 'director of' == 'director'
oo['leak'] = oo.sourceID.astype(str).map(lambda s: leak_of(s) if s != 'nan' else None)
nb = oo[~oo.bo]
log('officer_of', len(oo), 'non-BO', len(nb), 'BO frac', round(oo.bo.mean(), 4))

# leak assignment (paper rule): plurality source over non-BO officer_of edges with a tag
tagged = nb[nb.leak.notna()]
cnt = tagged.groupby(['node_id_start','leak']).size().rename('n').reset_index()
tot = cnt.groupby('node_id_start').n.sum()
top = cnt.sort_values('n', ascending=False).drop_duplicates('node_id_start').set_index('node_id_start')
edge_leak = top.leak
edge_purity = top.n / tot.reindex(top.index)
log('edge-assigned officers', len(edge_leak), edge_leak.value_counts().to_dict())

# ------------------------------------------------------------------ per-officer features
officers = np.array(sorted(nb.node_id_start.unique()))
oid2i = pd.Series(np.arange(len(officers)), index=officers)
nb = nb.assign(oi=oid2i.reindex(nb.node_id_start.values).values)
oe = nb[['oi','node_id_end']].drop_duplicates()
F = pd.DataFrame(index=np.arange(len(officers)))
F['n_ent'] = oe.groupby('oi').size()
F['n_roles'] = nb.groupby('oi').linkn.nunique()

jur = ent.set_index('node_id').jurisdiction
oe_j = oe.assign(j=jur.reindex(oe.node_id_end.values).values)
F['n_juris'] = oe_j.dropna(subset=['j']).groupby('oi').j.nunique()

io = r[r.rel_type == 'intermediary_of'][['node_id_start','node_id_end']]
io.columns = ['int_id','node_id_end']
oi_int = oe.merge(io, on='node_id_end')[['oi','int_id']]
F['n_int'] = oi_int.groupby('oi').int_id.nunique()

ra = r[r.rel_type == 'registered_address'][['node_id_start','node_id_end']]
ra.columns = ['src','addr']
# (a) officer's own registered addresses
own = ra[ra.src.isin(offset)]
own = own.assign(oi=oid2i.reindex(own.src.values).values).dropna(subset=['oi'])
own['oi'] = own.oi.astype(int)
# (b) addresses of the officer's entities
ea = ra.rename(columns={'src':'node_id_end'})
via = oe.merge(ea, on='node_id_end')[['oi','addr']]
F['n_addr_own'] = own.groupby('oi').addr.nunique()
F['n_addr_ent'] = via.groupby('oi').addr.nunique()
F['n_addr_union'] = pd.concat([own[['oi','addr']], via]).drop_duplicates().groupby('oi').size()
F = F.fillna(0)
log('tabular features done')

# dominant intermediary / address per officer (groups for grouped splits)
def dominant(df, col):
    c = df.groupby(['oi', col]).size().rename('n').reset_index().sort_values('n', ascending=False)
    return c.drop_duplicates('oi').set_index('oi')[col]
dom_int = dominant(oi_int, 'int_id').reindex(F.index)
dom_addr = dominant(pd.concat([own[['oi','addr']], via]), 'addr').reindex(F.index)
del oi_int, via, oe_j

# ------------------------------------------------------------------ graph features
g = r[r.rel_type.isin(['officer_of','registered_address','intermediary_of'])][['node_id_start','node_id_end']]
nodes = np.unique(np.concatenate([g.node_id_start.values, g.node_id_end.values]))
gi = pd.Series(np.arange(len(nodes)), index=nodes)
a = gi.reindex(g.node_id_start.values).values; b = gi.reindex(g.node_id_end.values).values
N = len(nodes)
A = sp.coo_matrix((np.ones(len(a), np.float32), (a, b)), shape=(N, N)).tocsr()
A = ((A + A.T) > 0).astype(np.float32).tocsr()
A.setdiag(0); A.eliminate_zeros()
deg = np.asarray(A.sum(1)).ravel()
log('graph', N, 'nodes', A.nnz // 2, 'edges')
# PageRank, damping .85, 30 power iterations, dangling mass spread uniformly
dinv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0).astype(np.float64)
P = sp.diags(dinv) @ A   # row-stochastic
x = np.full(N, 1.0 / N)
for _ in range(30):
    x = 0.85 * (P.T @ x) + 0.85 * x[deg == 0].sum() / N + 0.15 / N
pr = x
# mean / max neighbour degree
nbr_mean = np.where(deg > 0, (A @ deg) / np.maximum(deg, 1), 0)
indptr = A.indptr; dd = deg[A.indices]
nbr_max = np.zeros(N)
nz = np.diff(indptr) > 0
nbr_max[nz] = np.maximum.reduceat(dd, indptr[:-1][nz])
log('pagerank / nbr-degree done')
import igraph as ig
coo = sp.triu(A).tocoo()
G = ig.Graph(n=N, edges=np.column_stack([coo.row, coo.col]).tolist(), directed=False)
kcore = np.array(G.coreness())
del G, coo
log('k-core done')
oidx = gi.reindex(officers).values
H = pd.DataFrame({'pagerank': pr[oidx], 'kcore': kcore[oidx],
                  'nbr_mean_deg': nbr_mean[oidx], 'nbr_max_deg': nbr_max[oidx]})

# ------------------------------------------------------------------ labels and metadata
name = off.set_index('node_id').name.reindex(officers).fillna('').astype(str)
NAME_RX = re.compile(r'NOMINEE|NOMINEES|TRUSTEE|FIDUCIAR|NOMINI', re.I)
name_seed = name.str.contains(NAME_RX).values
und = r[(r.rel_type == 'underlying') & r.link.astype(str).str.startswith('Nominee')]
func = np.isin(officers, und.node_id_start.unique())
# person vs corporate (name-suffix rule; to be spot-checked by a human)
CORP_RX = re.compile(r"\b(LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|PLC|LLP|LP|"
                     r"S\.?A|S\.?A\.?R\.?L|S\.?L|N\.?V|B\.?V|AG|GMBH|SPA|S\.?P\.?A|PTE|PTY|"
                     r"COMPANY|CO|HOLDINGS?|SERVICES|NOMINEES|TRUSTEES|FIDUCIARIES|FIDUCIARY|"
                     r"FOUNDATION|ANSTALT|STIFTUNG|ESTABLISHMENT|BANK|TRUST|SECRETARIES|"
                     r"MANAGEMENT|INVESTMENTS?|ENTERPRISES?|GROUP|PARTNERS|ASSOCIATES|"
                     r"INTERNATIONAL|AGENCY|DIRECTORS|CORPORATE|SOCIETE|SOCIEDAD|BEARER)\b\.?", re.I)
corporate = name.str.contains(CORP_RX).values
node_src = off.set_index('node_id').sourceID.reindex(officers).astype(str).values
node_leak = np.array([leak_of(s) for s in node_src], dtype=object)

feat8 = np.column_stack([F.n_ent, np.log1p(F.n_ent), F.n_addr_union, F.n_int, F.n_juris, F.n_roles,
                         F.n_ent / np.maximum(F.n_addr_union, 1), F.n_ent / np.maximum(F.n_int, 1)]).astype(np.float64)
np.savez_compressed('pre.npz', officers=officers, feat8=feat8, hof=H.values.astype(np.float64),
                    n_addr_own=F.n_addr_own.values, n_addr_ent=F.n_addr_ent.values,
                    name_seed=name_seed, func=func, corporate=corporate,
                    edge_leak=edge_leak.reindex(officers).astype(object).values,
                    edge_purity=edge_purity.reindex(officers).values.astype(float),
                    node_leak=node_leak, node_src=node_src,
                    dom_int=dom_int.values.astype(float), dom_addr=dom_addr.values.astype(float))
pickle.dump({'names': name.values}, open('pre_meta.pkl', 'wb'))
log('saved', len(officers), 'officers; name_seed', name_seed.sum(), 'func', func.sum(), 'corporate', corporate.sum())
