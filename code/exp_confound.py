import sys, os, json; sys.path.insert(0, '/home/claude/icij')
from lib import *
F = '/home/claude/icij/res/confound.json'
OUT = json.load(open(F)) if os.path.exists(F) else {}
def P(*a): print(*a, flush=True)
def sv(): json.dump(OUT, open(F, 'w'), indent=1)

def permuted_labels(rng, leaks_to_perm):
    lab = name_seed.copy()
    for lk in leaks_to_perm:
        idx = np.where(leak == lk)[0]; lab[idx] = rng.permutation(lab[idx])
    return lab

def transfer_custom(train_leaks, h, cols, n_draws, permute=False, balanced=False, transform=None):
    aucs = []
    Z = X if transform is None else transform
    for k in range(n_draws):
        rng = np.random.default_rng(200 + k)
        lab = permuted_labels(rng, train_leaks) if permute else name_seed
        trp, trn = [], []
        if balanced:
            m = min(((leak == l) & lab).sum() for l in train_leaks)
        for l in train_leaks:
            Pp = np.where((leak == l) & lab)[0]; Qq = np.where((leak == l) & ~lab)[0]
            if balanced: Pp = rng.choice(Pp, m, replace=False)
            trp.append(Pp); trn.append(None if balanced else None)
        trp = np.concatenate(trp)
        if balanced:
            trn = np.concatenate([rng.choice(np.where((leak == l) & ~lab)[0], 3 * m, replace=False) for l in train_leaks])
        else:
            Q = np.where(np.isin(leak.astype(str), train_leaks) & ~lab)[0]
            trn = rng.choice(Q, 3 * len(trp), replace=False)
        TP = np.where((leak == h) & name_seed)[0]; TQ = np.where((leak == h) & ~name_seed)[0]
        TQ = TQ if len(TQ) <= CFG['cap_loo'] else rng.choice(TQ, CFG['cap_loo'], replace=False)
        idx = np.r_[trp, trn]; y = np.r_[np.ones(len(trp)), np.zeros(len(trn))]
        s = fit_score(gb, Z[idx][:, cols], y, Z[np.r_[TP, TQ]][:, cols])
        aucs.append(auc(s[:len(TP)], s[len(TP):]))
    return dict(mean=float(np.mean(aucs)), sd=float(np.std(aucs)), below=int(sum(a < .5 for a in aucs)), n=n_draws)

# ---- 1. real vs within-source permutation null, all holdouts and models
if 'perm_null' not in OUT:
    P('== REAL vs WITHIN-SOURCE PERMUTATION NULL (pooled LOO)')
    res = {}
    for k in ['connectivity', 'full8', 'full12']:
        res[k] = {}
        for h in LEAKS:
            tl = [l for l in LEAKS if l != h]
            res[k][h] = dict(real=transfer_custom(tl, h, SETS[k], 4), perm=transfer_custom(tl, h, SETS[k], 4, permute=True))
        P(f'  {k:12s}', ' | '.join(f"{h[:5]} real {res[k][h]['real']['mean']:.3f} perm {res[k][h]['perm']['mean']:.3f}" for h in LEAKS))
        sv()
    OUT['perm_null'] = res; sv()

# ---- 2. permuted + source-balanced (should sit at chance if the null is a prevalence effect)
if 'perm_balanced' not in OUT:
    P('== PERMUTED + SOURCE-BALANCED (full8)')
    res = {}
    for h in LEAKS:
        tl = [l for l in LEAKS if l != h]
        res[h] = dict(perm_bal=transfer_custom(tl, h, SETS['full8'], 4, permute=True, balanced=True),
                      real_bal=transfer_custom(tl, h, SETS['full8'], 4, balanced=True))
    P('  ', ' | '.join(f"{h[:5]} perm-bal {res[h]['perm_bal']['mean']:.3f} real-bal {res[h]['real_bal']['mean']:.3f}" for h in LEAKS))
    OUT['perm_balanced'] = res; sv()

# ---- 3. Panama held out: which training leaks produce the inversion?
if 'panama_subsets' not in OUT:
    P('== PANAMA HELD OUT, TRAINING-SOURCE SUBSETS (full8)')
    res = {}
    for tl in [['Paradise'], ['OffshoreLeaks'], ['Bahamas'], ['Paradise', 'OffshoreLeaks'],
               ['Paradise', 'Bahamas'], ['OffshoreLeaks', 'Bahamas'], ['Paradise', 'OffshoreLeaks', 'Bahamas']]:
        key = '+'.join(tl)
        res[key] = dict(real=transfer_custom(tl, 'Panama', SETS['full8'], 4),
                        perm=transfer_custom(tl, 'Panama', SETS['full8'], 4, permute=True) if len(tl) > 1 else None)
        P(f"  {key:32s} real {res[key]['real']['mean']:.3f}" + (f"  perm {res[key]['perm']['mean']:.3f}" if res[key]['perm'] else ''))
    OUT['panama_subsets'] = res; sv()

# ---- 4. in-corpus: permutation null and source-matched negatives
def incorpus_matched(permute=False, matched=True, cols=None, n_draws=5):
    aucs = []
    for k in range(n_draws):
        rng = np.random.default_rng(300 + k)
        lab = permuted_labels(rng, LEAKS) if permute else name_seed
        trp, trn, tep, ten = [], [], [], []
        for l in LEAKS:
            Pp = rng.permutation(np.where((leak == l) & lab)[0]); Qq = rng.permutation(np.where((leak == l) & ~lab)[0])
            ntr = int(0.7 * len(Pp)); trp.append(Pp[:ntr]); tep.append(Pp[ntr:])
            if matched:
                trn.append(Qq[:3 * ntr]); ten.append(Qq[3 * ntr: 3 * ntr + int(20000 * len(Pp[ntr:]) / ((U & lab).sum() * 0.3))])
        trp, tep = np.concatenate(trp), np.concatenate(tep)
        if matched:
            trn, ten = np.concatenate(trn), np.concatenate(ten)
        else:
            Q = rng.permutation(np.where(U & ~lab)[0]); trn = Q[:3 * len(trp)]; ten = Q[3 * len(trp): 3 * len(trp) + 20000]
        idx = np.r_[trp, trn]; y = np.r_[np.ones(len(trp)), np.zeros(len(trn))]
        s = fit_score(gb, X[idx][:, cols], y, X[np.r_[tep, ten]][:, cols])
        aucs.append(auc(s[:len(tep)], s[len(tep):]))
    return float(np.mean(aucs))
if 'incorpus' not in OUT:
    P('== IN-CORPUS: uniform vs source-matched unlabelled; real vs permuted labels')
    res = {}
    for k in ['degree', 'connectivity', 'full8', 'full12']:
        res[k] = dict(uniform_real=incorpus_matched(False, False, SETS[k]), uniform_perm=incorpus_matched(True, False, SETS[k]),
                      matched_real=incorpus_matched(False, True, SETS[k]), matched_perm=incorpus_matched(True, True, SETS[k]))
        P(f'  {k:12s}', {a: round(b, 3) for a, b in res[k].items()})
        sv()
    OUT['incorpus'] = res; sv()
P('DONE')
