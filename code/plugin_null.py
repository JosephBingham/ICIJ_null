"""Predict each held-out leak's permutation null without permuting: r(x) = sum_s pi_s P(s|x),
with P(s|x) from a source classifier trained on the training leaks and re-weighted to the pool composition."""
import sys, json, numpy as np; sys.path.insert(0, '/home/claude/icij')
from lib import *
from sklearn.ensemble import HistGradientBoostingClassifier
out = {}
L = np.array([str(x) for x in leak])
for k in ['connectivity', 'full8', 'full12']:
    out[k] = {}
    for h in LEAKS:
        tl = [l for l in LEAKS if l != h]
        rng = np.random.default_rng(0)
        idx = np.concatenate([rng.choice(np.where(L == l)[0], min(20000, (L == l).sum()), replace=False) for l in tl])
        ys = np.concatenate([[j] * min(20000, (L == l).sum()) for j, l in enumerate(tl)])
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, early_stopping=False)
        clf.fit(X[idx][:, SETS[k]], ys)
        n_s = np.array([(L == l).sum() for l in tl]); m_s = np.array([min(20000, (L == l).sum()) for l in tl])
        pi_s = np.array([name_seed[L == l].mean() for l in tl])
        TP = np.where((L == h) & name_seed)[0]; TQ = np.where((L == h) & ~name_seed)[0]
        TQ = TQ if len(TQ) <= 40000 else rng.choice(TQ, 40000, replace=False)
        pb = clf.predict_proba(X[np.r_[TP, TQ]][:, SETS[k]]) * (n_s / m_s)   # prior shift to pool composition
        pb /= pb.sum(1, keepdims=True)
        r = pb @ pi_s
        out[k][h] = float(auc(r[:len(TP)], r[len(TP):]))
    print(k, {h: round(v, 3) for h, v in out[k].items()}, flush=True)
json.dump(out, open('/home/claude/icij/res/plugin_null.json', 'w'), indent=1)
