"""Fig 4 v2 three rows (bone marrow, colon, cHL2), each in the published row-1 layout —
a) assigned-only precision per method (bar = macro over the published classes, dots = per-class, coverage printed under the bar),
b) per-class precision vs class frequency (log x), c) annotation runtime (log x). Colours as published + CELESTA in steel blue.
Methods scored on the same held-out cells within each dataset (fig4_v2_data.py)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import json, os, sys, numpy as np, matplotlib
from scipy.stats import wilcoxon
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patches as mpatches
D=f'{QCP.OUT}/retrain/figures/fig4_v2'
TITLES={'bm':'Bone marrow','colon':'Colon','colon16':'Colon\n(16 types)','chl1':'Lymph node','chl2':'Lymph node\n(MIBI, cHL2)','codex':'Lymph node\n(CODEX)'}
keys=sys.argv[1].split(',') if len(sys.argv)>1 else ['bm','colon','chl1']; OUT=f'{D}/'+(sys.argv[2] if len(sys.argv)>2 else 'fig4_v2.png')
rows=[(k,TITLES[k]) for k in keys]; avail=[(k,t) for k,t in rows if os.path.exists(f'{D}/fig4_data_{k}.json')]
METHODS=['AnnoSpat','Astir','CELESTA','MAPS','QuantCell']   # alphabetical
COL={'Astir':'crimson','AnnoSpat':'purple','MAPS':'orange','CELESTA':'steelblue','QuantCell':'green'}
default_fontsize = 24; small_fontsize = default_fontsize; big_fontsize = 30   # main-figure font set (24 pt text, 30 pt letters)
plt.rcParams['font.size']=default_fontsize; rng=np.random.default_rng(42); STATS={}
def stars(p): return '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'
n=len(rows); fig=plt.figure(figsize=(24, 8.3*n+1)); H=0.19 if n==3 else (0.29 if n==2 else 0.6); Y0={0:0.77,1:0.43,2:0.09} if n==3 else ({0:0.58,1:0.08} if n==2 else {0:0.1})   # taller rows so the stacked brackets clear the row above
for r,(key,title) in enumerate(rows):
    axA=fig.add_axes([0.11,Y0[r],0.26,H]); axB=fig.add_axes([0.45,Y0[r],0.22,H]); axC=fig.add_axes([0.76,Y0[r],0.21,H])
    for j,ax in enumerate((axA,axB,axC)): fig.text(ax.get_position().x0-0.045, ax.get_position().y1+0.01, 'ABCDEFGHIJKLMNO'[3*r+j], fontsize=big_fontsize, fontweight='bold')
    fig.text(0.005, Y0[r]+H/2, title, fontsize=default_fontsize, fontweight='bold', rotation=90, va='center')
    if (key,title) not in avail:
        for ax in (axA,axB,axC): ax.axis('off')
        axB.text(0.5,0.5,f'{title}: pending', ha='center', va='center', fontsize=default_fontsize, transform=axB.transAxes); continue
    d=json.load(open(f'{D}/fig4_data_{key}.json')); M=d['methods']; freq=d['frequency']; classes=d['classes']
    # a) bars + per-class dots + coverage
    x=np.arange(len(METHODS)); macro=[M[m]['macro_precision_all_classes'] for m in METHODS]
    axA.bar(x, macro, color='white', edgecolor='black'); axA.bar(x, macro, color=[COL[m] for m in METHODS], edgecolor='black', alpha=0.5)
    for i,m in enumerate(METHODS):
        v=np.array([M[m]['per_class_precision'][c] for c in classes]); axA.scatter(i+(rng.random(len(v))-0.5)/2, v, color='black', alpha=0.8, s=18)
    # published-style statistics — two-sided Wilcoxon signed-rank on per-class precision, each method vs
    # QuantCell, Bonferroni x (number of comparisons); brackets stacked above the bars, farthest comparison highest
    others=[m for m in METHODS if m!='QuantCell']; iq=METHODS.index('QuantCell'); qc=np.array([M['QuantCell']['per_class_precision'][c] for c in classes]); STATS[key]={}
    for lvl,m in enumerate(others[::-1]):   # MAPS (nearest) lowest bracket
        v=np.array([M[m]['per_class_precision'][c] for c in classes]); res=wilcoxon(v, qc); p_adj=min(1.0, res.pvalue*len(others))
        STATS[key][m]={'n_classes':int(len(v)),'statistic':float(res.statistic),'p_raw':float(res.pvalue),'p_bonferroni':float(p_adj),'label':stars(p_adj)}
        i=METHODS.index(m); yb=1.08+0.095*lvl
        axA.plot([i,i,iq,iq],[yb-0.03,yb,yb,yb-0.03], color='black', lw=1.5, clip_on=False); axA.text((i+iq)/2, yb+0.005, stars(p_adj), ha='center', va='bottom', fontsize=default_fontsize, clip_on=False)
    axA.set_xticks(x, METHODS, fontsize=default_fontsize, rotation=35, ha='right', rotation_mode='anchor'); axA.set_ylim(0,1.03); axA.set_yticks([0,0.2,0.4,0.6,0.8,1]); axA.set_ylabel('Macro precision', fontsize=default_fontsize); axA.spines[['top','right']].set_visible(False); axA.tick_params(axis='y', labelsize=default_fontsize)
    # b) precision vs frequency
    for m in METHODS:
        axB.scatter([freq[c] for c in classes], [M[m]['per_class_precision'][c] for c in classes], color=COL[m], alpha=0.6, label=m, s=40)
    axB.axvline(0.1, color='black', linestyle='--', linewidth=1.5); axB.set_xscale('log'); axB.set_ylim(-0.03,1.03); axB.set_xlabel('Frequency', fontsize=default_fontsize); axB.set_ylabel('Per-class precision', fontsize=default_fontsize)
    axB.spines[['top','right']].set_visible(False); axB.tick_params(labelsize=default_fontsize)
    # c) runtime
    MR=METHODS[::-1]; t=[d['runtime_seconds'].get(m) or np.nan for m in MR]; yy=np.arange(len(MR))
    axC.barh(yy, t, color='white', edgecolor='black'); axC.barh(yy, t, color=[COL[m] for m in MR], edgecolor='black', alpha=0.5)
    for i,v in enumerate(t):
        if np.isfinite(v): axC.text(v*1.15, i, f'{v:.0f}', va='center', fontsize=small_fontsize)
    axC.set_xscale('log'); axC.set_xlim(1, 3e5); axC.set_xticks([1,10,100,1000,10000,100000], ['10$^0$','10$^1$','10$^2$','10$^3$','10$^4$','10$^5$'], fontsize=default_fontsize); axC.set_yticks(yy, MR, fontsize=default_fontsize)
    axC.set_xlabel('Annotation runtime (seconds)', fontsize=default_fontsize); axC.spines[['top','right']].set_visible(False)
fig.legend(handles=[mpatches.Patch(facecolor=COL[m], edgecolor='black', alpha=0.6, label=m) for m in METHODS], loc='upper center', bbox_to_anchor=(0.53, Y0[len(rows)-1]-0.07), ncol=5, frameon=False, fontsize=default_fontsize)
fig.savefig(OUT, dpi=90, bbox_inches='tight'); print('written ->', OUT, '| rows drawn:', [k for k,_ in avail])
json.dump(STATS, open(OUT.replace('.png','_wilcoxon.json'),'w'), indent=1)
for k in STATS:
    for m,r in STATS[k].items(): print(f"  {k:6s} {m:9s} vs QuantCell  n={r['n_classes']:2d}  W={r['statistic']:.1f}  p={r['p_raw']:.2e}  Bonferroni p={r['p_bonferroni']:.2e}  {r['label']}")
