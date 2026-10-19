"""Corporate-name rule v2 (run after prep.py; writes corp_v2.npy used by exp_strata2/exp_final/make_figs).
v1 in prep.py wrongly counted NOMINEES/TRUSTEES/FIDUCIARY tokens as corporate, which is circular."""
import re, pickle, numpy as np
names = pickle.load(open('pre_meta.pkl', 'rb'))['names']
CORP_RX = re.compile(r"\b(LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|PLC|LLP|LP|"
                     r"S\.?A|S\.?A\.?R\.?L|S\.?L|N\.?V|B\.?V|AG|GMBH|SPA|S\.?P\.?A|PTE|PTY|"
                     r"COMPANY|CO|HOLDINGS?|SERVICES|FOUNDATION|ANSTALT|STIFTUNG|ESTABLISHMENT|BANK|TRUST|"
                     r"MANAGEMENT|INVESTMENTS?|ENTERPRISES?|GROUP|PARTNERS|ASSOCIATES|"
                     r"INTERNATIONAL|AGENCY|SOCIETE|SOCIEDAD|BEARER|S\.?A\.?S|SRL|KG|OY|AB|AS|SDN|BHD|PVT)\b\.?", re.I)
np.save('corp_v2.npy', np.array([bool(CORP_RX.search(n)) for n in names]))
