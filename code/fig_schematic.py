import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import rankdata
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
mus={'H':np.array([2.2,0.]),'A':np.array([-1.3,1.9]),'B':np.array([-1.3,-1.9])}
lam_pool={'H':.05,'A':.01,'B':.01}; lam_bal={'H':.02,'A':.02,'B':.02}
def r(P,lam):
    ll=np.stack([-0.5*((P-mus[s])**2).sum(1) for s in mus],1); post=np.exp(ll-logsumexp(ll,1,keepdims=True))
    return post@np.array([lam[s] for s in mus])
def auc(p,n): rr=rankdata(np.r_[p,n]); return (rr[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))
g=np.linspace(-4,4.5,220); GX,GY=np.meshgrid(g,g); grid=np.c_[GX.ravel(),GY.ravel()]
rng=np.random.default_rng(3)
fig,ax=plt.subplots(2,2,figsize=(5.4,5.3)); ax=ax.ravel()
def base(a,lam,title,vmax=.05):
    Z=r(grid,lam).reshape(GX.shape)
    if np.ptp(Z) > 1e-9: a.contourf(GX,GY,Z,levels=np.linspace(0,vmax,21),cmap='Greys',alpha=.55,vmin=0,vmax=vmax*1.3)
    else: a.set_facecolor('#e9e9e9')
    for s,c in mus.items():
        pts=c+0.55*rng.standard_normal((60,2)); a.scatter(pts[:,0],pts[:,1],s=3,c='#7f8c8d',alpha=.6,lw=0)
        k={'H':25,'A':5,'B':5}[s] if lam is lam_pool else 10
        lp=c+0.55*rng.standard_normal((k,2)); a.scatter(lp[:,0],lp[:,1],s=7,c='#c0392b',lw=0)
        a.text(c[0],c[1]+1.25,f'{s}: {int(lam[s]*100)}% labelled',ha='center',fontsize=6.3,bbox=dict(fc='white',ec='none',alpha=.75,pad=.5))
    a.set_xlim(-4,4.5); a.set_ylim(-4,4); a.set_xticks([]); a.set_yticks([]); a.set_aspect('equal'); a.set_title(title,fontsize=7.6)
base(ax[0],lam_pool,'(a) pooled training learns\n"resembles a high-rate source"')
ax[0].annotate('',xy=(1.6,-3.3),xytext=(-1.6,-3.3),arrowprops=dict(arrowstyle='->',lw=1))
ax[0].text(0,-3.75,'shuffled-label score $r(x)$ rises',ha='center',fontsize=6.3)
for a,shift,title in [(ax[1],+1.0,'(b) held-out positives lean\ntoward the high-rate source'),(ax[2],-1.0,'(c) held-out positives lean\naway from it (e.g. Panama)')]:
    Z=r(grid,lam_pool).reshape(GX.shape); a.contourf(GX,GY,Z,levels=np.linspace(0,.05,21),cmap='Greys',alpha=.55,vmin=0,vmax=.065)
    for s,c in mus.items(): a.add_patch(plt.Circle(c,1.0,fill=False,ls=':',ec='#999',lw=.8)); a.text(c[0],c[1],s,ha='center',va='center',fontsize=6.5,color='#777')
    U=0.6*rng.standard_normal((400,2)); Pp=0.6*rng.standard_normal((60,2))+np.array([shift*0.45,0])
    a.scatter(U[:,0],U[:,1],s=3,c='#7f8c8d',alpha=.6,lw=0,label='unlabelled'); a.scatter(Pp[:,0],Pp[:,1],s=8,c='#2471a3',lw=0,label='labelled positives')
    A_=auc(r(Pp+0,lam_pool),r(U,lam_pool))
    a.text(0.25,-3.55,f'held-out source T\nnull = {A_:.2f}',ha='center',fontsize=6.6,fontweight='bold',linespacing=1.1)
    a.set_xlim(-4,4.5); a.set_ylim(-4,4); a.set_xticks([]); a.set_yticks([]); a.set_aspect('equal'); a.set_title(title,fontsize=7.6)
ax[1].legend(fontsize=5.6,frameon=False,loc='upper left',handletextpad=.2,borderaxespad=0)
base(ax[3],lam_bal,'(d) source-balanced training:\nequal rates, flat score')
ax[3].text(0.25,-3.3,'null = 0.50 everywhere',ha='center',fontsize=6.6,fontweight='bold')
ax[3].text(0.25,-3.85,'(in-corpus: matched sampling)',ha='center',fontsize=6.0)
plt.tight_layout(w_pad=.6,h_pad=2.0); plt.savefig('/home/claude/icij/sn/fig_schematic.pdf'); plt.savefig('/home/claude/icij/figs/fig_schematic.png',dpi=220); print('ok')
