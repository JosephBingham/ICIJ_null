import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False})
S=json.load(open('res/sim.json')); PN=json.load(open('res/plugin_null.json')); CF=json.load(open('res/confound.json'))
fig,ax=plt.subplots(1,3,figsize=(7.6,2.5),gridspec_kw={'width_ratios':[1,1.15,1]})
rhos=[1,2,4,8]; ds=[0.0,0.5,1.0,2.0]
M=np.array([[S['A'][f'{r}_{d}']['null'] for d in ds] for r in rhos])
a=ax[0]; im=a.imshow(M,cmap='RdBu',vmin=0.25,vmax=0.75,origin='lower',aspect='auto')
for i in range(4):
    for j in range(4): a.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=7)
a.set_xticks(range(4)); a.set_xticklabels(ds); a.set_yticks(range(4)); a.set_yticklabels(rhos)
a.set_xlabel('source separation $\\delta$'); a.set_ylabel('prevalence ratio $\\rho$'); a.set_title('(a) held-out permutation null',fontsize=8.5)
a=ax[1]; th=[0,45,90,135,180]
g=lambda k: [S['B'][f'1.0_{t}'][k] for t in th]
a.plot(th,g('oracle_signal'),'k:',lw=1,label='true within-target AUC')
a.plot(th,g('real'),'o-',c='#c0392b',ms=3.5,label='pooled, real labels')
a.plot(th,g('null'),'o-',c='#7f8c8d',ms=3.5,label='pooled, permutation null')
a.plot(th,g('oracle_null'),'--',c='#7f8c8d',lw=1,label='predicted null $r(x)$')
a.plot(th,g('bal_real'),'s-',c='#2471a3',ms=3.5,label='source-balanced, real')
a.plot(th,g('bal_null'),'s-',c='#aed6f1',ms=3.5,label='source-balanced, null')
a.axhline(.5,ls='--',c='k',lw=.6); a.set_xticks(th); a.set_xlabel('signal angle $\\theta$ (deg)')
a.set_ylabel('held-out AUC'); a.set_ylim(.05,.8); a.legend(fontsize=5.6,frameon=False,loc='lower left',ncol=2,columnspacing=.5,handlelength=1.3); a.set_title('(b) $\\rho=4$, $\\delta=1$, $\\beta=1$',fontsize=8.5)
a=ax[2]; L=['Panama','Paradise','OffshoreLeaks','Bahamas']; mk={'connectivity':'s','full8':'o','full12':'^'}
for k in mk:
    xs=[PN[k][h] for h in L]; ys=[CF['perm_null'][k][h]['perm']['mean'] for h in L]
    a.scatter(xs,ys,marker=mk[k],s=22,c=['#c0392b','#2471a3','#27ae60','#8e44ad'])
    a.scatter([],[],marker=mk[k],s=22,c='#555',label={'connectivity':'connectivity','full8':'full-8','full12':'full-12'}[k])
a.plot([.25,.75],[.25,.75],'k:',lw=.8); a.set_xlim(.28,.72); a.set_ylim(.28,.72)
a.set_xlabel('predicted null'); a.set_ylabel('permutation null'); a.set_title('(c) ICIJ: predicted vs. observed',fontsize=8)
a.legend(fontsize=6,frameon=False,loc='upper left')
for h,c in zip(['Panama','Paradise','Offshore','Bahamas'],['#c0392b','#2471a3','#27ae60','#8e44ad']): pass
a.text(.71,.30,'colour = held-out leak:\nPanama, Paradise,\nOffshore, Bahamas',fontsize=5.5,ha='right',va='bottom')
plt.tight_layout(w_pad=1.2); plt.savefig('figs/fig_sim.pdf'); plt.savefig('figs/fig_sim.png',dpi=200); print('ok')
x=[PN[k][h] for k in mk for h in L]; y=[CF['perm_null'][k][h]['perm']['mean'] for k in mk for h in L]
print('corr',np.corrcoef(x,y)[0,1],'MAE',np.mean(np.abs(np.array(x)-np.array(y))))
