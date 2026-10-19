#!/usr/bin/env bash
# Reproduce every number, table and figure. Edit the absolute paths at the top of lib.py, exp_*.py,
# aml/code/*.py and revision/*.py to point at your copy of the data and this directory first.
set -euo pipefail
# --- ICIJ Offshore Leaks (unzip the ICIJ CSV export into data/)
python3 code/prep.py
python3 code/corp_rule.py
for s in exp_core exp_extra exp_new exp_confound exp_strata2 exp_final exp_srbf_grouped plugin_null exp_balanced_sub; do python3 code/$s.py; done
PYTHONHASHSEED=0 python3 code/sim.py
python3 code/make_figs.py; python3 code/make_fig_sim.py; python3 code/fig_schematic.py
# --- revision: repeated draws and extended simulations
PYTHONHASHSEED=0 python3 revision/sim_ext.py
python3 revision/exp_stats.py
# --- IBM AML (Altman et al. 2023; download HI-Small, LI-Small, HI-Medium from Kaggle)
python3 aml/code/build.py HI-Small HI-Small_Trans.csv
python3 aml/code/build.py LI-Small LI-Small_Trans.csv
python3 aml/code/build.py HI-Medium HI-Medium_Trans.csv
python3 aml/code/exp_aml.py HI-Small 100; python3 aml/code/exp_aml.py LI-Small 100
FEATS=cur python3 aml/code/exp_aml.py HI-Small 100
python3 aml/code/him_lite.py; python3 aml/code/dose.py; python3 aml/code/fig_aml.py
