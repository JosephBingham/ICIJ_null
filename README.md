# Paper B: reproduction and new experiments

Data: ICIJ Offshore Leaks CSV export, GENERATED_ON_20260909 (SHA-256 in checksums.txt).
Unzip the export into `data/` next to the scripts. Paths in the scripts assume /home/claude/icij; edit the
absolute paths in lib.py and the exp_*.py headers to your location. Requires numpy, pandas, scipy,
scikit-learn, python-igraph, matplotlib. Single CPU core, <4 GB RAM, ~1 hour total.

Run order (each script caches finished sections in results/*.json and skips them on re-run):
1. prep.py              -> pre.npz (features, labels, source tags, groups)
2. exp_core.py          -> in-corpus, leave-one-leak-out, transfer matrix, ablation, normalisation,
                           permutation (within-leak), logistic, source-balanced, source prediction
3. exp_extra.py         -> SMD (random vs the original first-40k slice), functional labels, SRBF-inspired PPR,
                           ratio sweep, bagging-PU, descriptive hub stats
4. exp_new.py           -> grouped splits, leave-one-sub-source-out, five-leak LOO with Pandora
                           (its 'strata' section used a circular corporate rule; superseded by step 6)
5. exp_confound.py      -> permutation nulls for LOO, permuted+balanced, Panama training subsets,
                           in-corpus uniform vs source-matched sampling
6. corp_rule.py (corporate-name rule v2) -> corp_v2.npy, then exp_strata2.py and exp_final.py
7. exp_srbf_grouped.py  -> PPR under grouped splits
8. make_figs.py         -> figures

Model: sklearn HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, early_stopping=False).
The address feature is the officer's own registered-address count (this reproduces the original
mechanism.py exactly, including its first-40k artifact).

Added for the DMKD version:
9.  sim.py                -> simulation sweeps A (prevalence x separation), B (direction x signal), C (reversal);
                             run with PYTHONHASHSEED=0 for identical seeds
10. plugin_null.py        -> predicted permutation nulls from a source classifier (Eq. 1)
11. exp_balanced_sub.py   -> source-balanced leave-one-sub-source-out and its null
12. make_fig_sim.py       -> Figure 5

AML replication (aml/): Altman et al. (2023) IBM synthetic AML, HI-Small / LI-Small / HI-Medium from Kaggle.
  build.py NAME TRANS_CSV   -> account features (expects NAME_accounts.csv in /mnt/user-data/uploads; edit paths)
  exp_aml.py NAME MINPOS    -> natural + reporting batteries (FEATS=cur adds currency shares; DRAWS sets draws)
  him_lite.py               -> reduced HI-Medium leave-one-country-out (7 largest countries, 1 draw)
  dose.py                   -> in-corpus dose-response (Fig. 6a); fig_aml.py, aml_section.py -> figure and table
