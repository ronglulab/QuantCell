#!/usr/bin/env python
"""Supplementary Fig 4b on the LightGBM retrain artifacts: per-cell-type PCA (500 conventional / 500 QuantCell-only /
500 other, sampled with replacement, seed 42) on the marker columns of that cell type's conventional combination.
Procedure = data_processing/notebooks/domain_shift.ipynb cell 8 (the published source), inputs switched to the
corrected codex_conventional / codex_quantcell CSVs. Panel a (schematic, originally BioRender) is re-drawn in matplotlib so the whole figure is generated from code.
Column selection is the published substring match ('CD3' also picks CD31/CD34 columns; see --exact for the
whole-marker alternative)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
EXACT='--substring' not in sys.argv  # exact marker match is the figure; --substring = published column selection
project_path=f'{QCP.OUT}/retrain_lgbm/QuantCellPaper'; project_name='QuantCellPaper'
OUT=f"{QCP.OUT}/retrain_lgbm/figures/suppl4_lgbm{'' if EXACT else '_substring'}.png"  # a + b
J=f'{QCP.COMBOS}'
marker_combos=json.load(open(J))
TYPES=['CD8 T Cells','Arterial ECs','MEP','CMP']
hdr=pd.read_csv(f'{project_path}/codex_quantcell_{project_name}.csv',nrows=0).columns
need=set(['cell_type'])
for ct in TYPES:
    markers=[m[:-1] for m in marker_combos[ct]]
    if EXACT: need|={c for c in hdr if ':' in c and c.split(':')[0] in markers}
    else:     need|={c for c in hdr if ':' in c and any(m in c for m in markers)}
usecols=[c for c in hdr if c in need]
conv=pd.read_csv(f'{project_path}/codex_rule_based_{project_name}.csv',index_col=0,usecols=lambda c:c in need or c=='Unnamed: 0' or c=='')
rel =pd.read_csv(f'{project_path}/codex_quantcell_{project_name}.csv',  index_col=0,usecols=lambda c:c in need or c=='Unnamed: 0' or c=='')
assert (conv.index==rel.index).all(), 'row order differs'
rel['is_quantcell']=np.logical_and(rel['cell_type']!='Other', conv['cell_type']=='Other')
rel['is_conventional']=conv['cell_type']!='Other'
print('cells',len(rel),'| conventional labelled',int(rel.is_conventional.sum()),'| QuantCell-only',int(rel.is_quantcell.sum()))
FS,LETTER=24,30  # same fonts as the main figures (24 pt text, 30 pt letters)
plt.rcParams['font.size']=FS
# ---- layout: shared legend on top, a = schematic row (three L-shaped axes), b = 2x2 PCA grid
fig=plt.figure(figsize=(14.5,20),dpi=150)
BLUE,GREEN,BLACK=(0.25,0.25,1),(0.25,0.6,0.25),'black'
def blob(rng, cx, cy, r, n, dmin):
    """n points spread evenly inside a disc of radius r (min spacing dmin), deterministic for a given rng"""
    pts=[]
    while len(pts)<n:
        a,rr=rng.uniform(0,2*np.pi),r*np.sqrt(rng.uniform())
        q=(cx+rr*np.cos(a),cy+rr*np.sin(a))
        if all((q[0]-u)**2+(q[1]-v)**2>=dmin**2 for u,v in pts): pts.append(q)
    return np.array(pts)
rng=np.random.default_rng(4)
schem=[('Unreliable',[((0.47,0.50),0.20,30,[BLUE,GREEN,BLACK]*10)]),
       ('Domain Shift',[((0.26,0.60),0.12,10,[BLACK]*10),((0.73,0.66),0.12,12,[BLUE]*12),((0.66,0.33),0.12,12,[GREEN]*12)]),
       ('Optimal',[((0.30,0.76),0.12,11,[BLACK]*11),((0.70,0.34),0.15,18,[BLUE,GREEN]*9)])]
axsA=[fig.add_axes([0.08+i*0.31,0.74,0.25,0.15]) for i in range(3)]
for ax,(lab,blobs) in zip(axsA,schem):
    for (cx,cy),r,n,cols in blobs:
        pts=blob(rng,cx,cy,r,n,0.055); cols=list(cols); rng.shuffle(cols)
        ax.scatter(pts[:,0],pts[:,1],s=60,c=cols,edgecolors='none')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([]); ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel(lab,fontsize=FS)
axs=[fig.add_axes([x,y,0.33,0.20]) for y in (0.42,0.08) for x in (0.12,0.62)]
summary={}
for n,ct in enumerate(TYPES):
    sc=rel.loc[rel.is_conventional&(rel.cell_type==ct)].sample(500,replace=True,random_state=42)
    sq=rel.loc[rel.is_quantcell&(rel.cell_type==ct)].sample(500,replace=True,random_state=42)
    so=rel.loc[rel.cell_type!=ct].sample(500,replace=True,random_state=42)
    comb=pd.concat([sc,sq,so])
    markers=[m[:-1] for m in marker_combos[ct]]
    if EXACT: cols=[c for c in rel.columns if ':' in c and c.split(':')[0] in markers]
    else:     cols=[c for c in rel.columns if ':' in c and any(m in c for m in markers)]
    reducer=PCA(); emb=reducer.fit_transform(comb.loc[:,cols])
    colors=[(0.25,0.25,1) if (r.is_conventional and r.cell_type==ct) else (0.25,0.6,0.25) if (r.is_quantcell and r.cell_type==ct) else 'black' for r in comb.itertuples()]
    ax=axs[n]; ax.scatter(emb[:,0],emb[:,1],alpha=0.5,c=colors,s=10,edgecolors='none'); ax.spines[['top','right']].set_visible(False)
    ev=reducer.explained_variance_ratio_*100
    ax.set_xlabel(f'PC1 ({ev[0]:.1f}%)'); ax.set_ylabel(f'PC2 ({ev[1]:.1f}%)'); ax.set_title(ct)
    summary[ct]={'n_conventional':int((rel.is_conventional&(rel.cell_type==ct)).sum()),'n_quantcell_only':int((rel.is_quantcell&(rel.cell_type==ct)).sum()),
                 'n_cols':len(cols),'markers_in_cols':sorted({c.split(':')[0] for c in cols}),'PC1':round(ev[0],1),'PC2':round(ev[1],1)}
    print(ct, summary[ct])
h=[plt.Line2D([0],[0],marker='o',color='w',label=l,markerfacecolor=c,markersize=10) for l,c in (('Rule-based',BLUE),('QuantCell',GREEN),('Unannotated',BLACK))]   # 'Other' -> 'Unannotated'
fig.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,0.985),frameon=False,fontsize=FS,ncol=3,columnspacing=4)
fig.text(0.01,0.91,'A',fontsize=LETTER,fontweight='bold'); fig.text(0.01,0.655,'B',fontsize=LETTER,fontweight='bold')
fig.savefig(OUT,bbox_inches='tight'); print('written ->',OUT)
json.dump(summary,open(OUT.replace('.png','.json'),'w'),indent=1)
