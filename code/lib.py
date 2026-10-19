"""Shared pieces for every Paper B experiment (frozen configuration)."""
import json, time, numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CFG = dict(trees=200, depth=3, lr=0.1, ratio=3, n_unl_incorpus=20000, cap_loo=40000,
           draws_incorpus=5, draws_loo=8, boot_total=1000, pos_split=0.7)
LEAKS = ['Panama', 'Paradise', 'OffshoreLeaks', 'Bahamas']
FN8 = ['n_ent', 'log_n_ent', 'n_addr', 'n_int', 'n_juris', 'n_roles', 'ent_per_addr', 'ent_per_int']
FNHO = ['pagerank', 'kcore', 'nbr_mean_deg', 'nbr_max_deg']
SETS = {'degree': [0], 'connectivity': [0, 2, 3], 'full8': list(range(8)),
        'higher_order': [8, 9, 10, 11], 'full12': list(range(12))}

d = np.load('/home/claude/icij/pre.npz', allow_pickle=True)
officers = d['officers']
n_addr = d['n_addr_own'].astype(float)          # officer's own registered addresses (matches original)
f8 = d['feat8'].copy()
f8[:, 2] = n_addr
f8[:, 6] = f8[:, 0] / np.maximum(n_addr, 1)
X = np.column_stack([f8, d['hof']])               # 12 features
name_seed = d['name_seed'].astype(bool)
func = d['func'].astype(bool)
corporate = d['corporate'].astype(bool)
edge_leak = d['edge_leak']
node_leak = d['node_leak']
node_src = d['node_src']
dom_int = d['dom_int']; dom_addr = d['dom_addr']
U = np.array([l is not None and l == l for l in edge_leak])     # edge-assigned pool (paper rule)
leak = np.where(U, edge_leak, None)

def gb():
    return HistGradientBoostingClassifier(max_iter=CFG['trees'], max_depth=CFG['depth'],
                                          learning_rate=CFG['lr'], early_stopping=False)
def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = rankdata(s)   # average ranks for ties
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def boot(pos, neg, B, rng):
    out = np.empty(B)
    for b in range(B):
        out[b] = auc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))])
    return out

def fit_score(model, Xtr, ytr, Xte):
    m = model(); m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]

def run_draws(make_split, cols, n_draws, model=gb, transform=None, seed0=0, rawscore=None):
    """make_split(rng) -> (tr_pos, tr_neg, te_pos, te_neg) index arrays into X.
    Returns mean AUC, CI from pooled bootstrap across draws, per-draw AUCs."""
    aucs, boots = [], []
    B = max(1, CFG['boot_total'] // n_draws)
    for k in range(n_draws):
        rng = np.random.default_rng(seed0 + k)
        trp, trn, tep, ten = make_split(rng)
        Z = X if transform is None else transform
        if rawscore is not None:
            sp, sn = rawscore[tep], rawscore[ten]
        else:
            idx = np.concatenate([trp, trn]); y = np.r_[np.ones(len(trp)), np.zeros(len(trn))]
            s = fit_score(model, Z[idx][:, cols], y, Z[np.concatenate([tep, ten])][:, cols])
            sp, sn = s[:len(tep)], s[len(tep):]
        aucs.append(auc(sp, sn)); boots.append(boot(sp, sn, B, rng))
    bb = np.concatenate(boots)
    return dict(mean=float(np.mean(aucs)), lo=float(np.percentile(bb, 2.5)), hi=float(np.percentile(bb, 97.5)),
                draws=[float(a) for a in aucs], below=int(sum(a < 0.5 for a in aucs)))

def fmt(r): return f"{r['mean']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] (<.5: {r['below']}/{len(r['draws'])})"

# ---------------------------------------------------------------- split builders
def incorpus_split(pool, pos_mask, ratio=None, n_unl=None):
    ratio = ratio or CFG['ratio']; n_unl = n_unl or CFG['n_unl_incorpus']
    P = np.where(pool & pos_mask)[0]; Q = np.where(pool & ~pos_mask)[0]
    def mk(rng):
        p = rng.permutation(P); ntr = int(CFG['pos_split'] * len(p))
        q = rng.permutation(Q)
        trn = q[:ratio * ntr]; ten = q[ratio * ntr: ratio * ntr + n_unl]
        return p[:ntr], trn, p[ntr:], ten
    return mk

def transfer_split(train_mask, test_mask, pos_mask, ratio=None, cap=None, balance_by=None):
    """train on train_mask, test on test_mask (disjoint). balance_by: array of group labels
    for source-balanced training (equal positives per group)."""
    ratio = ratio or CFG['ratio']; cap = cap or CFG['cap_loo']
    P = np.where(train_mask & pos_mask)[0]; Q = np.where(train_mask & ~pos_mask)[0]
    TP = np.where(test_mask & pos_mask)[0]; TQ = np.where(test_mask & ~pos_mask)[0]
    def mk(rng):
        if balance_by is None:
            trp = P; trn = rng.choice(Q, min(len(Q), ratio * len(P)), replace=False)
        else:
            gs = [g for g in np.unique(balance_by[P])]
            m = min((balance_by[P] == g).sum() for g in gs)
            trp = np.concatenate([rng.choice(P[balance_by[P] == g], m, replace=False) for g in gs])
            trn = np.concatenate([rng.choice(Q[balance_by[Q] == g], ratio * m, replace=False) for g in gs])
        ten = TQ if len(TQ) <= cap else rng.choice(TQ, cap, replace=False)
        return trp, trn, TP, ten
    return mk

def within_split(mask, pos_mask, ratio=None, n_unl=None):
    return incorpus_split(mask, pos_mask, ratio, n_unl)

def save(name, obj):
    json.dump(obj, open(f'/home/claude/icij/res/{name}.json', 'w'), indent=1)
