import sys; sys.path.insert(0, '/home/claude/icij')
from lib import *
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os, json
OUT = json.load(open('/home/claude/icij/res/core.json')) if os.path.exists('/home/claude/icij/res/core.json') else {}
def P(*a): print(*a, flush=True)

from scipy.stats import rankdata
# ---------------- Table 2: in-corpus
if 'incorpus' not in OUT:
    P('== IN-CORPUS (name-based, pooled)')
    T2 = {}
    for k in ['degree', 'connectivity', 'higher_order', 'full8', 'full12']:
        T2[k] = run_draws(incorpus_split(U, name_seed), SETS[k], CFG['draws_incorpus'])
        P(f'  {k:13s}', fmt(T2[k]))
    T2['raw_degree'] = run_draws(incorpus_split(U, name_seed), None, CFG['draws_incorpus'], rawscore=X[:, 0])
    P('  raw degree   ', fmt(T2['raw_degree']))
    OUT['incorpus'] = T2; save('core', OUT)


# ---------------- Table 4: leave-one-leak-out
if 'loo' not in OUT:
    P('== LEAVE-ONE-LEAK-OUT')
    T4 = {}
    for k in ['degree', 'connectivity', 'full8', 'full12', 'higher_order']:
        T4[k] = {}
        for h in LEAKS:
            T4[k][h] = run_draws(transfer_split(U & (leak != h), leak == h, name_seed), SETS[k], CFG['draws_loo'])
        P(f'  {k:13s}', ' | '.join(f"{h[:5]} {T4[k][h]['mean']:.3f}({T4[k][h]['below']}/8)" for h in LEAKS))
    T4['raw_degree'] = {h: run_draws(transfer_split(U & (leak != h), leak == h, name_seed), None, 1, rawscore=X[:, 0]) for h in LEAKS}
    P('  raw degree   ', ' | '.join(f"{h[:5]} {T4['raw_degree'][h]['mean']:.3f}" for h in LEAKS))
    OUT['loo'] = T4; save('core', OUT)


# ---------------- Table 6: transfer matrix (full8)
if 'matrix' not in OUT:
    P('== TRANSFER MATRIX (full8)')
    TM = {}
    for a in LEAKS:
        TM[a] = {}
        for b in LEAKS:
            if a == b: r = run_draws(within_split(leak == a, name_seed), SETS['full8'], 4)
            else: r = run_draws(transfer_split(leak == a, leak == b, name_seed), SETS['full8'], 4)
            TM[a][b] = r
        P(f'  {a:13s}', ' '.join(f"{TM[a][b]['mean']:.3f}" for b in LEAKS))
    OUT['matrix'] = TM; save('core', OUT)


# ---------------- Table 7: ablation (LOO)
if 'ablation' not in OUT:
    P('== ABLATION (LOO)')
    ABL = {'drop_addr': [0, 1, 3, 4, 5, 6, 7], 'drop_roles': [0, 1, 2, 3, 4, 6, 7],
           'drop_addr_channel': [0, 1, 3, 4, 5, 7], 'deg+addr': [0, 2], 'deg+roles': [0, 5]}
    AB = {}
    for k, cols in ABL.items():
        AB[k] = {h: run_draws(transfer_split(U & (leak != h), leak == h, name_seed), cols, CFG['draws_loo']) for h in LEAKS}
        P(f'  {k:18s}', ' | '.join(f"{h[:5]} {AB[k][h]['mean']:.3f}({AB[k][h]['below']}/8)" for h in LEAKS))
    AB['lofo_panama'] = {}
    for j, f in enumerate(FN8):
        cols = [c for c in range(8) if c != j]
        AB['lofo_panama'][f] = run_draws(transfer_split(U & (leak != 'Panama'), leak == 'Panama', name_seed), cols, CFG['draws_loo'])['mean']
    P('  LOFO Panama:', {k: round(v, 3) for k, v in AB['lofo_panama'].items()})
    OUT['ablation'] = AB; save('core', OUT)


# ---------------- Table 8: normalisation
if 'normalisation' not in OUT:
    P('== NORMALISATION (LOO, full8)')
    Z = X.copy(); R = X.copy()
    for lk in LEAKS:
        m = leak == lk
        Z[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)
        for c in range(X.shape[1]): R[m, c] = rankdata(X[m, c]) / m.sum()
    NR = {}
    for nm, T in [('per_source_z', Z), ('per_source_rank', R)]:
        NR[nm] = {h: run_draws(transfer_split(U & (leak != h), leak == h, name_seed), SETS['full8'], CFG['draws_loo'], transform=T) for h in LEAKS}
        P(f'  {nm:16s}', ' | '.join(f"{h[:5]} {NR[nm][h]['mean']:.3f}({NR[nm][h]['below']}/8)" for h in LEAKS))
    OUT['normalisation'] = NR; save('core', OUT)


# ---------------- permutation control
if 'permutation' not in OUT:
    P('== PERMUTATION (labels shuffled within each training leak)')
    PM = {}
    for h in LEAKS:
        aucs = []
        for k in range(4):
            rng = np.random.default_rng(100 + k); lab = name_seed.copy()
            for lk in LEAKS:
                if lk == h: continue
                idx = np.where(leak == lk)[0]; lab[idx] = rng.permutation(lab[idx])
            trm = U & (leak != h)
            Pp = np.where(trm & lab)[0]; Qq = np.where(trm & ~lab)[0]
            TP = np.where((leak == h) & name_seed)[0]; TQ = np.where((leak == h) & ~name_seed)[0]
            TQ = TQ if len(TQ) <= CFG['cap_loo'] else rng.choice(TQ, CFG['cap_loo'], replace=False)
            trn = rng.choice(Qq, 3 * len(Pp), replace=False)
            idx = np.r_[Pp, trn]; y = np.r_[np.ones(len(Pp)), np.zeros(len(trn))]
            s = fit_score(gb, X[idx][:, SETS['full8']], y, X[np.r_[TP, TQ]][:, SETS['full8']])
            aucs.append(auc(s[:len(TP)], s[len(TP):]))
        PM[h] = dict(mean=float(np.mean(aucs)), draws=aucs, below=int(sum(a < .5 for a in aucs)))
    P('  ', ' | '.join(f"{h[:5]} {PM[h]['mean']:.3f}({PM[h]['below']}/4)" for h in LEAKS))
    OUT['permutation'] = PM; save('core', OUT)


# ---------------- logistic + source-balanced
if 'logistic' not in OUT:
    P('== LOGISTIC (LOO full8) and SOURCE-BALANCED (GB)')
    LG = {h: run_draws(transfer_split(U & (leak != h), leak == h, name_seed), SETS['full8'], 4, model=logreg) for h in LEAKS}
    P('  logistic      ', ' | '.join(f"{h[:5]} {LG[h]['mean']:.3f}({LG[h]['below']}/4)" for h in LEAKS))
    SB = {h: run_draws(transfer_split(U & (leak != h), leak == h, name_seed, balance_by=leak), SETS['full8'], CFG['draws_loo']) for h in LEAKS}
    P('  source-bal    ', ' | '.join(f"{h[:5]} {SB[h]['mean']:.3f}({SB[h]['below']}/8)" for h in LEAKS))
    OUT['logistic'] = LG; OUT['source_balanced'] = SB; save('core', OUT)


# ---------------- source prediction
if 'source_pred' not in OUT:
    P('== SOURCE PREDICTION (8 features, 10k/leak, 5-fold)')
    rng = np.random.default_rng(7)
    idx = np.concatenate([rng.choice(np.where(leak == lk)[0], 10000, replace=False) for lk in LEAKS])
    y = np.repeat(np.arange(4), 10000)
    rf = cross_val_score(RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=0), X[idx][:, :8], y, cv=5).mean()
    lr = cross_val_score(logreg(), X[idx][:, :8], y, cv=5).mean()
    P(f'  RF {rf:.3f}  LR {lr:.3f}')
    OUT['source_pred'] = dict(rf=float(rf), lr=float(lr)); save('core', OUT)


P('DONE')
