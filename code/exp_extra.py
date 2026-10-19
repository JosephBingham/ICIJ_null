import sys, os, json; sys.path.insert(0, '/home/claude/icij')
from lib import *
import pandas as pd, scipy.sparse as sp
F = '/home/claude/icij/res/extra.json'
OUT = json.load(open(F)) if os.path.exists(F) else {}
def P(*a): print(*a, flush=True)
def sv(): json.dump(OUT, open(F, 'w'), indent=1)

# ---------------- SMD: random negatives vs the original's first-40k slice
if 'smd' not in OUT:
    P('== SMD (nominee vs unlabelled), random 40k vs first-40k-by-node-id')
    cols = {'n_ent': 0, 'n_addr': 2, 'n_int': 3, 'n_juris': 4, 'n_roles': 5}
    res = {'random': {}, 'first40k': {}}
    for nm, c in cols.items():
        res['random'][nm] = {}; res['first40k'][nm] = {}
        for lk in LEAKS:
            Pp = np.where((leak == lk) & name_seed)[0]; N = np.where((leak == lk) & ~name_seed)[0]
            def smd(Nn): return float((X[Pp, c].mean() - X[Nn, c].mean()) / (np.sqrt((X[Pp, c].std()**2 + X[Nn, c].std()**2) / 2) + 1e-9))
            res['first40k'][nm][lk] = smd(N[:40000])
            res['random'][nm][lk] = float(np.mean([smd(np.random.default_rng(s).choice(N, min(40000, len(N)), replace=False)) for s in range(5)]))
        P(f"  {nm:8s} random:", ' '.join(f"{res['random'][nm][l]:+.2f}" for l in LEAKS),
          '| first40k:', ' '.join(f"{res['first40k'][nm][l]:+.2f}" for l in LEAKS))
    Pn = np.where((leak == 'Panama') & ~name_seed)[0]
    res['panama_addr_frac'] = dict(first40k=float((X[Pn[:40000], 2] > 0).mean()), all=float((X[Pn, 2] > 0).mean()),
                                   pos=float((X[(leak == 'Panama') & name_seed, 2] > 0).mean()))
    res['bahamas_addr_frac'] = float((X[leak == 'Bahamas', 2] > 0).mean())
    P('  ', res['panama_addr_frac'], 'Bahamas any-address frac', res['bahamas_addr_frac'])
    OUT['smd'] = res; sv()

# ---------------- functional labels
if 'functional' not in OUT:
    P('== FUNCTIONAL LABELS')
    fpos = np.where(U & func & ~name_seed)[0]
    res = {}
    for negpool_name, negmask in [('pooled', U & ~name_seed & ~func), ('offshore_only', U & ~name_seed & ~func & (leak == 'OffshoreLeaks'))]:
        res[negpool_name] = {}
        for k in ['degree', 'connectivity', 'full8']:
            aucs = []
            for s in range(5):
                rng = np.random.default_rng(s)
                Pn = np.where(U & name_seed)[0]; Q = np.where(U & ~name_seed & ~func)[0]
                trn = rng.choice(Q, 3 * len(Pn), replace=False)
                te_n = rng.choice(np.setdiff1d(np.where(negmask)[0], trn), 20000, replace=False)
                idx = np.r_[Pn, trn]; y = np.r_[np.ones(len(Pn)), np.zeros(len(trn))]
                sc = fit_score(gb, X[idx][:, SETS[k]], y, X[np.r_[fpos, te_n]][:, SETS[k]])
                aucs.append(auc(sc[:len(fpos)], sc[len(fpos):]))
            res[negpool_name][k] = float(np.mean(aucs))
        rng = np.random.default_rng(0); te_n = rng.choice(np.where(negmask)[0], 20000, replace=False)
        res[negpool_name]['raw_degree'] = float(auc(X[fpos, 0], X[te_n, 0]))
        P(f'  name-trained -> functional ({negpool_name} negatives):', {k: round(v, 3) for k, v in res[negpool_name].items()})
    # within-definition (tiny n: 20 draws)
    wd = {}
    pool = U & ~name_seed
    for k in ['connectivity', 'full8']:
        wd[k] = run_draws(incorpus_split(pool, func), SETS[k], 20)
    wd['raw_degree'] = run_draws(incorpus_split(pool, func), None, 20, rawscore=X[:, 0])
    P('  within-definition:', {k: fmt(v) for k, v in wd.items()})
    res['within'] = wd
    fp = X[U & func, 0]; npos = X[U & name_seed, 0]
    res['degree_stats'] = dict(func_median=float(np.median(fp)), func_mean=float(fp.mean()), func_max=float(fp.max()),
                               name_median=float(np.median(npos)), name_max=float(npos.max()))
    P('  degree stats', res['degree_stats'])
    OUT['functional'] = res; sv()

# ---------------- graph for PPR / positive control
r = pd.read_csv('/home/claude/icij/data/relationships.csv', usecols=['node_id_start', 'node_id_end', 'rel_type'],
                dtype={'node_id_start': 'int64', 'node_id_end': 'int64', 'rel_type': str})
g = r[r.rel_type.isin(['officer_of', 'registered_address', 'intermediary_of'])]
nodes = np.unique(np.r_[g.node_id_start.values, g.node_id_end.values])
gi = pd.Series(np.arange(len(nodes)), index=nodes)
a = gi.reindex(g.node_id_start.values).values; b = gi.reindex(g.node_id_end.values).values
N = len(nodes)
A = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(N, N)).tocsr(); A = ((A + A.T) > 0).astype(float).tocsr()
deg = np.asarray(A.sum(1)).ravel()
PT = (sp.diags(np.where(deg > 0, 1 / np.maximum(deg, 1), 0)) @ A).T.tocsr()
off_g = gi.reindex(officers).values

def ppr(seed_officer_idx, alpha=0.15, iters=30):
    s = np.zeros(N); s[off_g[seed_officer_idx]] = 1.0 / len(seed_officer_idx)
    x = s.copy()
    for _ in range(iters): x = (1 - alpha) * (PT @ x) + alpha * s
    return x[off_g]

if 'srbf' not in OUT:
    P('== SRBF-INSPIRED PPR (restart 0.15, 30 iterations)')
    res = {}
    def mk_split(rng): return incorpus_split(U, name_seed)(rng)
    aucs = []
    for k in range(CFG['draws_incorpus']):
        rng = np.random.default_rng(k); trp, trn, tep, ten = incorpus_split(U, name_seed)(rng)
        sc = ppr(trp); aucs.append(auc(sc[tep], sc[ten]))
    res['incorpus'] = dict(mean=float(np.mean(aucs)), draws=[float(x) for x in aucs])
    res['loo'] = {}
    for h in LEAKS:
        sc = ppr(np.where(U & (leak != h) & name_seed)[0])
        tp = np.where((leak == h) & name_seed)[0]; tq = np.where((leak == h) & ~name_seed)[0]
        res['loo'][h] = dict(auc=float(auc(sc[tp], sc[tq])), mass_in_heldout=float(sc[leak == h].sum()))
    P('  in-corpus', round(res['incorpus']['mean'], 3), '| LOO', {h: round(v['auc'], 3) for h, v in res['loo'].items()},
      '| mass in held-out', {h: v['mass_in_heldout'] for h, v in res['loo'].items()})
    OUT['srbf'] = res; sv()

# ---------------- sampling ratio / bagging PU
if 'ratio' not in OUT:
    P('== RATIO SWEEP')
    res = {}
    for q in [1, 3, 5, 10]:
        a1 = run_draws(incorpus_split(U, name_seed, ratio=q), SETS['full12'], CFG['draws_incorpus'])['mean']
        a2 = run_draws(transfer_split(U & (leak != 'Panama'), leak == 'Panama', name_seed, ratio=q), SETS['full8'], CFG['draws_loo'])['mean']
        res[q] = dict(incorpus_full12=a1, panama_loo_full8=a2); P(f'  1:{q}', round(a1, 3), round(a2, 3))
    OUT['ratio'] = res; sv()

if 'bagging_pu' not in OUT:
    P('== BAGGING-PU (Mordelet & Vert), in-corpus full8')
    aucs = []
    for k in range(CFG['draws_incorpus']):
        rng = np.random.default_rng(k); trp, trn, tep, ten = incorpus_split(U, name_seed)(rng)
        Q = np.setdiff1d(np.where(U & ~name_seed)[0], ten)
        sc = np.zeros(len(tep) + len(ten))
        for t in range(10):
            bn = rng.choice(Q, 3 * len(trp), replace=False)
            idx = np.r_[trp, bn]; y = np.r_[np.ones(len(trp)), np.zeros(len(bn))]
            sc += fit_score(gb, X[idx][:, SETS['full8']], y, X[np.r_[tep, ten]][:, SETS['full8']])
        aucs.append(auc(sc[:len(tep)], sc[len(tep):]))
    OUT['bagging_pu'] = float(np.mean(aucs)); P('  ', round(OUT['bagging_pu'], 3)); sv()

# ---------------- positive control
if 'poscontrol' not in OUT:
    P('== POSITIVE CONTROL')
    ra = r[r.rel_type == 'registered_address'].groupby('node_id_end').node_id_start.nunique()
    it = r[r.rel_type == 'intermediary_of'].groupby('node_id_start').node_id_end.nunique()
    def top1(s): s = np.sort(s.values)[::-1]; k = max(1, int(round(0.01 * len(s)))); return float(s[:k].sum() / s.sum())
    res = dict(addr_ge100=int((ra >= 100).sum()), addr_ge1000=int((ra >= 1000).sum()), addr_max=int(ra.max()), addr_top1=top1(ra),
               int_ge100=int((it >= 100).sum()), int_ge1000=int((it >= 1000).sum()), int_max=int(it.max()), int_top1=top1(it))
    P('  ', res); OUT['poscontrol'] = res; sv()
P('DONE')
