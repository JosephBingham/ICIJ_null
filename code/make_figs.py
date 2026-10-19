import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, pandas as pd, sys
sys.path.insert(0,'/home/claude/icij'); from lib import *
R={k:json.load(open(f'res/{k}.json')) for k in ['core','extra','new','confound','final','strata2']}
corp=np.load('corp_v2.npy')
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
C={'real':'#c0392b','perm':'#7f8c8d','raw':'#2c3e50'}
# Fig 1: in-corpus
ic=R['confound']['incorpus']; ms=['degree','connectivity','full8','full12']; lab=['degree','connectivity','full-8','full-12']
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6),sharey=True)
for a,(cond,t) in zip(ax,[('uniform','(a) uniform unlabelled sample (as in prior work)'),('matched','(b) source-matched unlabelled sample')]):
    x=np.arange(4); w=.38
    a.bar(x-w/2,[ic[m][cond+'_real'] for m in ms],w,color=C['real'],label='real labels')
    a.bar(x+w/2,[ic[m][cond+'_perm'] for m in ms],w,color=C['perm'],label='labels permuted within source')
    for i,m in enumerate(ms):
        a.text(i-w/2,ic[m][cond+'_real']+.01,f"{ic[m][cond+'_real']:.2f}",ha='center',fontsize=7)
        a.text(i+w/2,ic[m][cond+'_perm']+.01,f"{ic[m][cond+'_perm']:.2f}",ha='center',fontsize=7)
    a.axhline(.5,ls='--',c='k',lw=.7); a.set_xticks(x); a.set_xticklabels(lab); a.set_ylim(.4,.9); a.set_title(t,fontsize=8.5)
ax[0].set_ylabel('in-corpus ROC-AUC'); ax[0].legend(fontsize=7,frameon=False,loc='upper left')
plt.tight_layout(); plt.savefig('figs/fig_incorpus_null.pdf'); plt.savefig('figs/fig_incorpus_null.png',dpi=200); plt.close()
# Fig 2: LOO real vs perm
pn=R['confound']['perm_null']; ss=R['new']['subsource']; sp_=R['final']['subsource_perm']
fig,ax=plt.subplots(1,2,figsize=(7.0,2.9),gridspec_kw={'width_ratios':[1,1.25]})
a=ax[0]; mk={'connectivity':'s','full8':'o','full12':'^'}
for j,m in enumerate(['connectivity','full8','full12']):
    for i,h in enumerate(LEAKS):
        xr=i+(j-1)*.22; r_,p_=pn[m][h]['real']['mean'],pn[m][h]['perm']['mean']
        a.plot([xr,xr],[p_,r_],c='#bbb',lw=1,zorder=1); a.scatter(xr,p_,marker=mk[m],c=C['perm'],s=22,zorder=2); a.scatter(xr,r_,marker=mk[m],c=C['real'],s=22,zorder=3)
a.axhline(.5,ls='--',c='k',lw=.7); a.set_xticks(range(4)); a.set_xticklabels(['Panama','Paradise','Offshore\nLeaks','Bahamas']); a.set_ylabel('held-out ROC-AUC')
a.set_title('(a) leave-one-leak-out (□ conn., ○ full-8, △ full-12)',fontsize=8.5)
from matplotlib.lines import Line2D
a.legend(handles=[Line2D([],[],marker='o',ls='',c=C['real'],label='real labels'),Line2D([],[],marker='o',ls='',c=C['perm'],label='permutation null')],fontsize=7,frameon=False,loc='upper right')
a=ax[1]; dom=list(sp_.keys()); short=['Bahamas','Offshore','Panama','Pandora:\nTrident','Paradise:\nAppleby','Paradise:\nBarbados','Paradise:\nMalta']
for i,d in enumerate(dom):
    r_=ss['loo']['full8'][d]['mean']; p_=sp_[d]
    a.plot([i,i],[p_,r_],c='#bbb',lw=1); a.scatter(i,p_,c=C['perm'],s=22); a.scatter(i,r_,c=C['real'],s=22)
    a.scatter(i,ss['loo']['degree'][d]['mean'],marker='x',c=C['raw'],s=18)
a.axhline(.5,ls='--',c='k',lw=.7); a.set_xticks(range(len(dom))); a.set_xticklabels(short,fontsize=6.5); a.set_title('(b) leave-one-sub-source-out, full-8 (× degree null)',fontsize=8.5)
plt.tight_layout(); plt.savefig('figs/fig_loo_null.pdf'); plt.savefig('figs/fig_loo_null.png',dpi=200); plt.close()
# Fig 3: label populations
fig,ax=plt.subplots(1,2,figsize=(7.0,2.5))
a=ax[0]
for m,lb,c in [(U,'all leak-assigned officers','#999'),(U&name_seed,'name-derived proxy','#2471a3'),(U&func,'functional proxy','#c0392b')]:
    v=np.sort(X[m,0]); a.plot(v+1,np.arange(1,len(v)+1)/len(v),label=f'{lb} (n={m.sum():,})',c=c,lw=1.2)
a.set_xscale('log'); a.set_xlabel('entity degree + 1'); a.set_ylabel('cumulative fraction'); a.legend(fontsize=6.5,frameon=False,loc='lower right'); a.set_title('(a) degree of the two label families',fontsize=8.5)
a=ax[1]; x=np.arange(4); w=.38
cp=[corp[(leak==l)&name_seed].mean() for l in LEAKS]; cu=[corp[(leak==l)&~name_seed].mean() for l in LEAKS]
a.bar(x-w/2,cp,w,color='#2471a3',label='name-derived positives'); a.bar(x+w/2,cu,w,color='#bbb',label='unlabelled officers')
a.set_xticks(x); a.set_xticklabels(['Panama','Paradise','Offshore','Bahamas*']); a.set_ylabel('share with corporate name'); a.set_ylim(0,1.25); a.set_yticks([0,.25,.5,.75,1]); a.legend(fontsize=7,frameon=False,loc='upper center',ncol=2); a.set_title('(b) the proxy names companies; the pool varies',fontsize=8.5)
plt.tight_layout(); plt.savefig('figs/fig_labels.pdf'); plt.savefig('figs/fig_labels.png',dpi=200); plt.close()
# Fig 4: hubs
r=pd.read_csv('data/relationships.csv',usecols=['node_id_start','node_id_end','rel_type'],dtype={'node_id_start':'int64','node_id_end':'int64','rel_type':str})
ra=r[r.rel_type=='registered_address'].groupby('node_id_end').node_id_start.nunique().values
it=r[r.rel_type=='intermediary_of'].groupby('node_id_start').node_id_end.nunique().values
fig,ax=plt.subplots(1,2,figsize=(7.0,2.4))
for a,v,t,c in [(ax[0],ra,'(a) nodes per registered address','#2471a3'),(ax[1],it,'(b) entities per intermediary','#c0392b')]:
    s=np.sort(v); cc=1-np.arange(len(s))/len(s); a.loglog(s,cc,'.',ms=2,c=c); a.set_title(t,fontsize=8.5); a.set_ylabel('P(degree ≥ x)'); a.set_xlabel('degree')
plt.tight_layout(); plt.savefig('figs/fig_hubs.pdf'); plt.savefig('figs/fig_hubs.png',dpi=200); plt.close()
print('ok')
