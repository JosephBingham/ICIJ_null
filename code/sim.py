"""Simulation: when does the within-source permutation null leave 0.5, and does excess over it track signal?

Four sources in R^8 with identity covariance. Training sources H (high prevalence), A, B; held-out target T.
  means: T = 0, H = delta*e1, A = delta*e2, B = delta*e3
  prevalence: pi_H = rho*pi0, pi_A = pi_B = pi_T = pi0
  positives are shifted by beta*w, w = cos(theta) e1 + sin(theta) e4 (shared across sources unless reversed)
  PU labelling: each positive labelled with probability c (SCAR within source)
Pipeline mirrors the paper: GB (depth 3), 1:3 pseudo-negatives, labelled positives vs unlabelled of T.
"""
import sys, json, os, numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
sys.path.insert(0, '/home/claude/icij')
from lib import auc
from sklearn.ensemble import HistGradientBoostingClassifier
D, N, PI0, C = 8, 20000, 0.03, 0.5
F = '/home/claude/icij/res/sim.json'
OUT = json.load(open(F)) if os.path.exists(F) else {}
def P(*a): print(*a, flush=True)
def sv(): json.dump(OUT, open(F, 'w'), indent=1)
def gb(): return HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.1, early_stopping=False)

def make(rng, delta, rho, beta, theta, reverse_T=False):
    E = np.eye(D); w = np.cos(theta) * E[0] + np.sin(theta) * E[3]
    mus = {'T': np.zeros(D), 'H': delta * E[0], 'A': delta * E[1], 'B': delta * E[2]}
    pis = {'T': PI0, 'H': rho * PI0, 'A': PI0, 'B': PI0}
    X, Y, L, S = [], [], [], []
    for s in ['H', 'A', 'B', 'T']:
        y = rng.random(N) < pis[s]
        sgn = -1.0 if (reverse_T and s == 'T') else 1.0
        x = rng.standard_normal((N, D)) + mus[s] + sgn * beta * np.outer(y, w)
        l = y & (rng.random(N) < C)
        X.append(x); Y.append(y); L.append(l); S += [s] * N
    return np.vstack(X), np.concatenate(Y), np.concatenate(L), np.array(S), mus, pis, w

def fit_eval(X, lab, S, train_src, rng, balanced=False, test='T'):
    tr = np.isin(S, train_src)
    if balanced:
        m = min((lab & (S == s)).sum() for s in train_src)
        Pp = np.concatenate([rng.choice(np.where(lab & (S == s))[0], m, replace=False) for s in train_src])
        Q = np.concatenate([rng.choice(np.where(~lab & (S == s))[0], 3 * m, replace=False) for s in train_src])
    else:
        Pp = np.where(tr & lab)[0]; Q = rng.choice(np.where(tr & ~lab)[0], 3 * len(Pp), replace=False)
    m_ = gb(); m_.fit(X[np.r_[Pp, Q]], np.r_[np.ones(len(Pp)), np.zeros(len(Q))])
    te = S == test
    return m_.predict_proba(X[te])[:, 1]

def permute(lab, S, srcs, rng):
    lab = lab.copy()
    for s in srcs:
        i = np.where(S == s)[0]; lab[i] = rng.permutation(lab[i])
    return lab

def oracle_null(XT, mus, pis, train_src):
    """Population null score: r(x) = sum_s pi_s P(s|x) over the training pool (equal source sizes)."""
    ll = np.stack([-0.5 * ((XT - mus[s])**2).sum(1) for s in train_src], 1)   # ignores positive shift (small pi)
    post = np.exp(ll - logsumexp(ll, 1, keepdims=True))
    return np.round(post @ np.array([pis[s] for s in train_src]), 10)   # round away float noise when sources coincide

def linear_null(beta, mus, pis, w, train_src):
    """First-order prediction: AUC ~ Phi(beta * cos(grad r, w) / sqrt(2)), grad r at the target mean."""
    ll = np.array([-0.5 * (mus[s]**2).sum() for s in train_src]); post = np.exp(ll - logsumexp(ll))
    pv = np.array([pis[s] for s in train_src]); M = np.stack([mus[s] for s in train_src])
    mbar = post @ M; g = (post * pv) @ (M - mbar)
    if np.linalg.norm(g) < 1e-12: return 0.5
    return float(norm.cdf(beta * (g @ w) / np.linalg.norm(g) / np.sqrt(2)))

TR = ['H', 'A', 'B']
def one(delta, rho, beta, theta, reps, reverse_T=False, incorpus=False):
    keys = ['real', 'null', 'bal_real', 'bal_null', 'oracle_null', 'oracle_signal']
    if incorpus: keys += ['inc_uniform_null', 'inc_matched_null']
    acc = {k: [] for k in keys}
    for r in range(reps):
        seed = abs(hash((round(delta, 3), rho, round(beta, 3), round(theta, 4), reverse_T, incorpus))) % (2**31)
        rng = np.random.default_rng([seed, r])
        X, Y, L, S, mus, pis, w = make(rng, delta, rho, beta, theta, reverse_T)
        te = S == 'T'; lp, lu = L[te], ~L[te]
        def A_(s): return auc(s[lp], s[lu])
        acc['real'].append(A_(fit_eval(X, L, S, TR, rng)))
        Lp = permute(L, S, TR, rng)
        acc['null'].append(A_(fit_eval(X, Lp, S, TR, rng)))
        acc['bal_real'].append(A_(fit_eval(X, L, S, TR, rng, balanced=True)))
        acc['bal_null'].append(A_(fit_eval(X, Lp, S, TR, rng, balanced=True)))
        acc['oracle_null'].append(A_(oracle_null(X[te], mus, pis, TR)))
        acc['oracle_signal'].append(A_(X[te] @ w))
        if incorpus:
            # pooled in-corpus: 70/30 split of labelled positives over all four sources, permuted within source
            Lq = permute(L, S, ['H', 'A', 'B', 'T'], rng)
            Pp = rng.permutation(np.where(Lq)[0]); ntr = int(.7 * len(Pp)); trp, tep = Pp[:ntr], Pp[ntr:]
            Q = rng.permutation(np.where(~Lq)[0]); trn = Q[:3 * ntr]
            m_ = gb(); m_.fit(X[np.r_[trp, trn]], np.r_[np.ones(ntr), np.zeros(len(trn))])
            ten_u = Q[3 * ntr: 3 * ntr + 5000]
            ten_m = np.concatenate([rng.choice(np.setdiff1d(np.where(~Lq & (S == s))[0], trn),
                                               max(1, int(5000 * (S[tep] == s).mean())), replace=False) for s in ['H', 'A', 'B', 'T']])
            sp = m_.predict_proba(X[tep])[:, 1]
            acc['inc_uniform_null'].append(auc(sp, m_.predict_proba(X[ten_u])[:, 1]))
            acc['inc_matched_null'].append(auc(sp, m_.predict_proba(X[ten_m])[:, 1]))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out.update({k + '_sd': float(np.std(v)) for k, v in acc.items()})
    E = np.eye(D); mus = {'T': np.zeros(D), 'H': delta * E[0], 'A': delta * E[1], 'B': delta * E[2]}
    pis = {'T': PI0, 'H': rho * PI0, 'A': PI0, 'B': PI0}
    w = np.cos(theta) * E[0] + np.sin(theta) * E[3]
    out['linear_null'] = linear_null(beta, mus, pis, w, TR)
    return out

if __name__ == '__main__':
    REPS = 10
    if 'A' not in OUT:
        P('== SWEEP A: null vs prevalence ratio x source separation (theta=180 deg, beta=1)')
        OUT['A'] = {}
        for rho in [1, 2, 4, 8]:
            for delta in [0.0, 0.5, 1.0, 2.0]:
                o = one(delta, rho, 1.0, np.pi, REPS, incorpus=True); OUT['A'][f'{rho}_{delta}'] = o
                P(f"  rho={rho} delta={delta}: LOO null {o['null']:.3f} (oracle {o['oracle_null']:.3f}, linear {o['linear_null']:.3f}) "
                  f"real {o['real']:.3f} | in-corpus null uniform {o['inc_uniform_null']:.3f} matched {o['inc_matched_null']:.3f}")
                sv()
    if 'B' not in OUT:
        P('== SWEEP B: direction x signal strength (rho=4, delta=1)')
        OUT['B'] = {}
        for beta in [0.0, 0.5, 1.0]:
            for th in [0, 45, 90, 135, 180]:
                o = one(1.0, 4, beta, np.deg2rad(th), REPS); OUT['B'][f'{beta}_{th}'] = o
                P(f"  beta={beta} theta={th:3d}: real {o['real']:.3f} null {o['null']:.3f} (oracle {o['oracle_null']:.3f}, lin {o['linear_null']:.3f}) "
                  f"excess {o['real']-o['null']:+.3f} | bal real {o['bal_real']:.3f} bal null {o['bal_null']:.3f} | oracle signal {o['oracle_signal']:.3f}")
                sv()
    if 'C' not in OUT:
        P('== SWEEP C: genuine reversal in the target (signal sign flipped in T; rho=4, delta=1, theta=90)')
        OUT['C'] = {}
        for beta in [0.5, 1.0]:
            o = one(1.0, 4, beta, np.deg2rad(90), REPS, reverse_T=True); OUT['C'][str(beta)] = o
            P(f"  beta={beta}: real {o['real']:.3f} null {o['null']:.3f} excess {o['real']-o['null']:+.3f} | bal real {o['bal_real']:.3f} | oracle signal {o['oracle_signal']:.3f}")
            sv()
    P('DONE')
