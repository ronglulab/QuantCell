"""Supplementary Fig 11 spatial feature-family comparison, LightGBM runs (spatial_lgbm/), all text 20 pt,
panel letters 28 pt. Content = fig_spatial_v3.py (AP only, bars at k=10 / k=20, dashed line = no spatial features)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S=f'{QCP.OUT}/spatial_lgbm'; out=f'{QCP.OUT}/retrain_lgbm/figures/suppl11_spatial_lgbm.png'
FS,LETTER=24,30; plt.rcParams["font.size"]=FS; fig,axs=plt.subplots(3,1,figsize=(15,23))  # main-figure fonts; third row = lymph node cHL1
fams={'F1':'Neighbour-mean\nintensities','F2':'Rule-based\ncomposition','F3':'Local\ndensity','all':'All\nfamilies'}
get=lambda v:v['average_precision_macro']
for row,(ds,title,let) in enumerate((('bm','Bone marrow','a'),('colon','Colon','b'),('chl1','Lymph node','c'))):
    r=json.load(open(f'{S}/{ds}/spatial_results.json')); base=r['base']; keys=[f for f in fams if f'{f}_k10' in r]
    ax=axs[row]; x=np.arange(len(keys)); w=0.36
    v10=[get(r[f'{f}_k10']) for f in keys]; v20=[get(r[f'{f}_k20']) for f in keys]; b=get(base)
    print(ds, 'base', round(b,4), {f:(round(a,4),round(c,4)) for f,a,c in zip(keys,v10,v20)})
    ax.bar(x-w/2, v10, w, color='white', edgecolor='black'); ax.bar(x-w/2, v10, w, color='#7f7f7f', alpha=0.45, edgecolor='black', label='k = 10')
    ax.bar(x+w/2, v20, w, color='white', edgecolor='black'); ax.bar(x+w/2, v20, w, color='#7f7f7f', alpha=1.0, edgecolor='black', label='k = 20')
    ax.axhline(b, color='black', linestyle='--', linewidth=1.5, label='No spatial features')
    lo=min(v10+v20+[b]); hi=max(v10+v20+[b]); pad=(hi-lo)*1.5+0.01; ax.set_ylim(lo-pad, hi+pad)
    ax.set_xticks(x, [fams[f] for f in keys]); ax.set_ylabel('Average precision'); ax.spines[['top','right']].set_visible(False)
    ax.set_title(title, loc='left'); ax.text(-0.18,1.05,let.upper(),transform=ax.transAxes,fontsize=LETTER,fontweight='bold')
    if row==0: ax.legend(frameon=False, loc='upper right')
fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches='tight'); print('written ->', out)
