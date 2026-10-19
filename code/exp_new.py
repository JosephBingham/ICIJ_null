import sys, os, json; sys.path.insert(0, '/home/claude/icij')
from lib import *
F = '/home/claude/icij/res/new.json'
OUT = json.load(open(F)) if os.path.exists(F) else {}
def P(*a): print(*a, flush=True)
def sv(): json.dump(OUT, open(F, 'w'), indent=1)
def row(d, keys): return ' | '.join(f"{k[:6]} {d[k]['mean']:.3f}({d[k]['below']}/{len(d[k]['draws'])})" for k in keys)

# ---------------- 1. person vs corporate
if 'strata' not in OUT:
    P('== PERSON vs CORPORATE (name-suffix rule)')
    res = {'corp_frac_pos': {}, 'corp_frac_unl': {}}
    for lk in LEAKS:
        res['corp_frac_pos'][lk] = float(corporate[(leak == lk) & name_seed].mean())
        res['corp_frac_unl'][lk] = float(corporate[(leak == lk) & ~name_seed].mean())
    P('  corporate share of name-positives:', {k: round(v, 3) for k, v in res['corp_frac_pos'].items()})
    P('  corporate share of unlabelled    :', {k: round(v, 3) for k, v in res['corp_frac_unl'].items()})
    # SMD within strata
    res['smd'] = {}
    for st, sm in [('person', ~corporate), ('corporate', corporate)]:
        res['smd'][st] = {}
        for nm, c in [('n_ent', 0), ('n_addr', 2), ('n_int', 3), ('n_roles', 5)]:
            res['smd'][st][nm] = {}
            for lk in LEAKS:
                Pp = np.where((leak == lk) & name_seed & sm)[0]; N = np.where((leak == lk) & ~name_seed & sm)[0]
                if len(Pp) < 20: res['smd'][st][nm][lk] = None; continue
                Nn = np.random.default_rng(0).choice(N, min(40000, len(N)), replace=False)
                res['smd'][st][nm][lk] = float((X[Pp, c].mean() - X[Nn, c].mean()) / (np.sqrt((X[Pp, c].std()**2 + X[Nn, c].std()**2) / 2) + 1e-9))
            P(f'  SMD {st:9s} {nm:7s}', ' '.join('  n/a' if res['smd'][st][nm][l] is None else f"{res['smd'][st][nm][l]:+.2f}" for l in LEAKS))
    # LOO within each stratum, and pooled-with-stratum-matched negatives
    res['loo'] = {}
    for st, sm in [('person', ~corporate), ('corporate', corporate)]:
        res['loo'][st] = {}
        for k in ['degree', 'connectivity', 'full8']:
            res['loo'][st][k] = {}
            for h in LEAKS:
                if ((leak == h) & name_seed & sm).sum() < 30: continue
                res['loo'][st][k][h] = run_draws(transfer_split(U & sm & (leak != h), (leak == h) & sm, name_seed), SETS[k], 4)
            P(f'  LOO {st:9s} {k:12s}', row(res['loo'][st][k], list(res['loo'][st][k])))
        sv()
    res['n_pos'] = {st: {lk: int(((leak == lk) & name_seed & sm).sum()) for lk in LEAKS} for st, sm in [('person', ~corporate), ('corporate', corporate)]}
    P('  positives per stratum', res['n_pos'])
    OUT['strata'] = res; sv()

# ---------------- 2. grouped in-corpus splits
def grouped_split(groups, pos_mask=name_seed, ratio=3, n_unl=20000):
    gid = groups.copy()
    def mk(rng):
        ug = np.unique(gid[U]); tr_g = set(ug[rng.random(len(ug)) < 0.7])
        intr = np.isin(gid, np.array(sorted(tr_g))) & U
        te = U & ~intr
        Ptr = np.where(intr & pos_mask)[0]; Qtr = np.where(intr & ~pos_mask)[0]
        Pte = np.where(te & pos_mask)[0]; Qte = np.where(te & ~pos_mask)[0]
        trn = rng.choice(Qtr, min(len(Qtr), ratio * len(Ptr)), replace=False)
        ten = rng.choice(Qte, min(len(Qte), n_unl), replace=False)
        return Ptr, trn, Pte, ten
    return mk
if 'grouped' not in OUT:
    P('== GROUPED IN-CORPUS SPLITS')
    own = -np.arange(len(officers)).astype(float) - 1           # singleton groups
    G = {'random_officer': own,
         'dominant_intermediary': np.where(np.isnan(dom_int), own, dom_int),
         'dominant_address': np.where(np.isnan(dom_addr), own, dom_addr)}
    res = {}
    for gname, grp in G.items():
        res[gname] = {}
        for k in ['connectivity', 'full8', 'full12']:
            res[gname][k] = run_draws(grouped_split(grp), SETS[k], 3)
        P(f'  {gname:22s}', row(res[gname], ['connectivity', 'full8', 'full12']))
        sv()
    for gname in ['dominant_intermediary', 'dominant_address']:
        grp = G[gname]; sizes = {}
        _, cnt = np.unique(grp[U & name_seed], return_counts=True)
        res[gname]['pos_in_shared_groups'] = float(cnt[cnt > 1].sum() / cnt.sum())
        P(f'  {gname}: share of positives sharing a group with another positive = {res[gname]["pos_in_shared_groups"]:.3f}')
    OUT['grouped'] = res; sv()

# ---------------- 3. leave-one-subsource-out (officer node sourceID; recovers Barbados and Pandora)
if 'subsource' not in OUT:
    P('== LEAVE-ONE-SUBSOURCE-OUT (node sourceID)')
    V = np.array([s != 'nan' for s in node_src])
    srcs, cnt = np.unique(node_src[V], return_counts=True)
    dom = [s for s in srcs if ((node_src == s) & name_seed).sum() >= 50 and (node_src == s).sum() >= 2000]
    res = {'domains': {s: dict(n=int((node_src == s).sum()), pos=int(((node_src == s) & name_seed).sum()),
                               corp_share=float(corporate[(node_src == s) & name_seed].mean())) for s in dom}}
    for s in dom: P(f"  {s:60s} n={res['domains'][s]['n']:7d} pos={res['domains'][s]['pos']:6d} corp={res['domains'][s]['corp_share']:.2f}")
    W = np.isin(node_src, dom)
    res['loo'] = {}
    for k in ['degree', 'full8']:
        res['loo'][k] = {}
        for s in dom:
            res['loo'][k][s] = run_draws(transfer_split(W & (node_src != s), node_src == s, name_seed), SETS[k], 3)
            sv()
        P(f'  {k}:', ' | '.join(f"{s.split(' - ')[-1][:14]} {res['loo'][k][s]['mean']:.3f}" for s in dom))
    # per-source normalised version
    Z = X.copy()
    for s in dom:
        m = node_src == s; Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)
    res['loo']['full8_z'] = {s: run_draws(transfer_split(W & (node_src != s), node_src == s, name_seed), SETS['full8'], 3, transform=Z) for s in dom}
    P('  full8 per-source z:', ' | '.join(f"{s.split(' - ')[-1][:14]} {res['loo']['full8_z'][s]['mean']:.3f}" for s in dom))
    OUT['subsource'] = res; sv()

# ---------------- 4. five-leak LOO with node-sourceID assignment (adds Pandora, Barbados, Nevis)
if 'five_leak' not in OUT:
    P('== FIVE-LEAK LOO (node sourceID)')
    L5 = LEAKS + ['Pandora']
    V5 = np.isin(node_leak.astype(str), L5)
    res = {'n': {l: int((node_leak == l).sum()) for l in L5}, 'pos': {l: int(((node_leak == l) & name_seed).sum()) for l in L5}}
    P('  ', res['n'], res['pos'])
    for k in ['degree', 'full8']:
        res[k] = {h: run_draws(transfer_split(V5 & (node_leak != h), node_leak == h, name_seed), SETS[k], 4) for h in L5}
        P(f'  {k:8s}', row(res[k], L5))
    OUT['five_leak'] = res; sv()
P('DONE')
