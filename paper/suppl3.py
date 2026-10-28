#!/usr/bin/env python
""" Supplementary Fig 3 v3 3x3 — a across the top; b/c/d the FDR sweeps; e confusion, f/g heatmaps) a = segmentation overlay spanning the whole row with the published-style legend
strip (DAPI / Nuclei-based / Membrane-based); b = column-normalised confusion matrix (nested held-out, 5 % FDR); c/d/e = the three
FDR-sweep analyses as in main Fig 3 (c realised FDR vs target, nested; d fraction annotated vs target, full-data recipe, rule-based
dotted baseline; e macro precision / recall, nested); f/g = feature inclusion / exclusion heatmaps (10-fold CV). Membrane-based
(Cellpose) codex, LightGBM, renamed labels. 24 pt text, 21 pt small, 30 pt letters. Usage: suppl3_render_v2.py [panel_a_png]"""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import sys, json, joblib, numpy as np, pandas as pd, warnings, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix, average_precision_score
warnings.filterwarnings('ignore')
NAME='QuantCellPaperCellpose'; P=f'{QCP.OUT}/retrain_lgbm/{NAME}'; MS=f'{P}/model_selection'
FIG=f'{QCP.OUT}/retrain_lgbm/figures'
A=sys.argv[1] if len(sys.argv)>1 else f'{QCP.OUT}/fig1_image/suppl3a_wide_split.png'  # wide field, 27.8 x 6.2 in
OUT=f'{FIG}/suppl3_v3_cellpose_lgbm.png'
FS,SM,BIG=24,21,30; plt.rcParams['font.size']=FS
cell_order=['KLS','MEP','CMP','GMP','CLP','CFU-E','MkP','Erythroblasts','Erythrocytes','Megakaryocytes','B Cells','CD4 T Cells','CD8 T Cells',
 'cDC','Double Negative T Cells','Monocytes','Neutrophils','Arterial ECs','Capillary ECs','Other BM ECs','Lepr+ Perivascular Cells']
# b
enc=joblib.load(f'{P}/label_encoder.joblib'); lab=pd.read_csv(f'{P}/quantcell_labels.csv').iloc[:,0]
tru=pd.read_csv(f'{P}/quantcell_true_labels.csv', index_col=0).loc[:,'cell_type']
assert set(cell_order)==set(enc.classes_)
cm=pd.DataFrame(confusion_matrix(tru,lab,labels=enc.classes_), index=enc.classes_, columns=enc.classes_).loc[cell_order,cell_order]
norm=cm.divide(cm.sum(axis=0),axis=1).fillna(0)
# c/d/e (same loaders as fig3_render_v2.py, per-project sweep files)
sw=json.load(open(f'{FIG}/fdr_sweep_bm_v2_{NAME}.json')); swf=json.load(open(f'{FIG}/fdr_sweep_bm_full_{NAME}.json'))
assert abs(swf['rule_based_fraction']-sw['rule_based_fraction'])<1e-9
SC=sw['cutoffs']; pc=np.array([c['cutoff']*100 for c in SC]); g=lambda k: np.array([c[k] if c[k] is not None else np.nan for c in SC]); XMAX=10.0
pcf=np.array([c['cutoff']*100 for c in swf['cutoffs']]); annot_full=np.array([c['annotated_fraction_all_cells'] for c in swf['cutoffs']])
# f/g
LOC=['Membrane','Cytoplasm','Nucleus','Cell','All']; ST=['Min','Max','Std.Dev.','Median','Mean','All']
def build(path, skip=False):
    D=joblib.load(path); m={}; s={}
    for L in LOC:
        m[L]={}; s[L]={}
        for S in ST:
            e=D.get(L,{}).get(S)
            if e is None or e[0] is None or (skip and L=='All' and S=='All'): m[L][S]=np.nan; s[L][S]=np.nan; continue
            yt,pr=e[0],e[2]; ap=[average_precision_score(yt[n],pr[n],average='macro') for n in yt]; m[L][S]=np.mean(ap); s[L][S]=np.std(ap)
    return pd.DataFrame(m).T, pd.DataFrame(s).T
inc,inc_sd=build(f'{MS}/stats_loc_dict_included.joblib'); exc,exc_sd=build(f'{MS}/stats_loc_dict_excluded.joblib', skip=True)
rows=[L for L in LOC if not inc.loc[L].isna().all()]; inc=inc.loc[rows]; inc_sd=inc_sd.loc[rows]; exc=exc.loc[rows]; exc_sd=exc_sd.loc[rows]
base=inc.loc['All','All']; dexc=base-exc
txt=lambda m,s: pd.DataFrame({c:[('' if np.isnan(m.loc[r,c]) else f'{m.loc[r,c]:.3f}\n±{s.loc[r,c]:.3f}') for r in m.index] for c in m.columns}, index=m.index)
# ---- layout (inches, from the top-left; canvas 30 x 37 in): a across the top; row 2 = FDR sweeps b/c/d; row 3 = e confusion, f/g heatmaps
WIN,HIN=27.2,21.0; fig=plt.figure(figsize=(WIN,HIN))
def axin(x,y_top,w,h): return fig.add_axes([x/WIN,1-(y_top+h)/HIN,w/WIN,h/HIN])
X0=[4.3,12.1,20.4]; PW=4.2; CB=0.36; ANN=14   # smaller canvas so 24 pt text reads larger; heatmap annotations 14 pt to fit 0.7-in cells
im=plt.imread(A); AW=X0[2]+PW+0.7-X0[0]; AH=PW   # panel a image area = row height; the image was rendered at exactly AW x AH (suppl3a_wide.py)
assert abs(im.shape[1]/im.shape[0]-AW/AH)<0.01, (im.shape, AW/AH)
axA=axin(X0[0],0.5,AW,AH); axL=axin(X0[0],0.5+AH,AW,0.9)
R2=7.0; axC=axin(X0[0],R2,PW,PW); axD=axin(X0[1],R2,PW,PW); axE=axin(X0[2],R2,PW,PW)
R3=13.4; axB=axin(X0[0],R3,PW,PW); axBc=axin(X0[0]+PW+0.3,R3,CB,PW)
axF=axin(X0[1],R3,PW,PW); axFc=axin(X0[1]+PW+0.3,R3,CB,PW); axG=axin(X0[2],R3,PW,PW); axGc=axin(X0[2]+PW+0.3,R3,CB,PW)
for l,x,y in [('a',0.5,0.2),('b',0.5,R2-0.6),('c',X0[1]-2.0,R2-0.6),('d',X0[2]-2.0,R2-0.6),('e',0.5,R3-0.6),('f',X0[1]-2.0,R3-0.6),('g',X0[2]-2.0,R3-0.6)]:
    fig.text(x/WIN,1-y/HIN,l.upper(),fontsize=BIG,fontweight='bold',va='top',ha='left')
# a + legend strip (published Suppl 3a format: black strip, line swatches, white text)
axA.imshow(im, interpolation='nearest'); axA.axis('off')
axL.set_facecolor('black'); axL.set_xticks([]); axL.set_yticks([]); [s.set_visible(False) for s in axL.spines.values()]
h=[Line2D([0],[0],color='blue',lw=4),Line2D([0],[0],color='white',lw=4),Line2D([0],[0],color='red',lw=4)]
axL.legend(h,['DAPI','Nuclei-based','Membrane-based'],loc='center',ncol=3,frameon=False,fontsize=FS,labelcolor='white',handlelength=2.5,columnspacing=4,handletextpad=1)
# b
sns.heatmap(norm, linecolor='grey', linewidth=0.01, xticklabels=cell_order, yticklabels=cell_order, cmap=sns.color_palette('flare',as_cmap=True), vmin=0, vmax=1,
            cbar_ax=axBc, cbar_kws={'format':FuncFormatter(lambda x,_:f'{x*100:.0f}%')}, ax=axB)
axB.set_xticklabels(cell_order, rotation=90, fontsize=ANN); axB.set_yticklabels(cell_order, rotation=0, fontsize=ANN)   # 21 classes in a 4.2-in panel: 14 pt; axBc.tick_params(labelsize=FS)
axB.set_ylabel('Rule-based annotation', fontsize=FS); axB.set_xlabel('QuantCell (membrane-based)', fontsize=FS)
# c/d/e
ax=axC; ax.plot(pc, g('realised_fdr_pooled')*100, color='black', lw=3.5); ax.plot([0,XMAX],[0,XMAX], ls='--', color='black', lw=1.5); ax.set_ylabel('Realized FDR (%)'); ax.set_ylim(0,XMAX)
ax=axD; ax.plot(pcf, annot_full, color='black', lw=3.5); ax.axhline(sw['rule_based_fraction'], color='black', ls='--', lw=1.5, label='Rule-based annotation'); ax.set_ylabel('Fraction annotated'); ax.set_ylim(0,1.02)
ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.0,1.02), fontsize=FS)
ax=axE; ax.plot(pc, g('macro_precision'), color='#1f77b4', lw=3.5, label='Macro precision'); ax.plot(pc, g('macro_recall'), color='#d62728', lw=3.5, label='Macro recall'); ax.set_ylabel('Score'); ax.set_ylim(0,1.02)
ax.legend(frameon=False, loc='lower right', fontsize=FS)
for ax in (axC,axD,axE):
    ax.set_xlim(0,XMAX); ax.set_xticks([0,2,4,6,8,10]); ax.set_xlabel('FDR target (%)'); ax.axvline(5, color='lightgray', ls='-', lw=1.5, zorder=0); ax.spines[['top','right']].set_visible(False); ax.tick_params(labelsize=FS)
# f/g
sns.heatmap(inc, annot=txt(inc,inc_sd), fmt='', vmin=0, vmax=1, cmap='Reds', annot_kws={'size':ANN}, ax=axF, cbar_ax=axFc); axFc.set_ylabel('Average precision', fontsize=FS); axFc.tick_params(labelsize=FS)
vmax=np.nanmax(np.abs(dexc.values)); cmap=sns.diverging_palette(10,260,as_cmap=True)
sns.heatmap(dexc, annot=txt(dexc,exc_sd), fmt='', vmin=-vmax, vmax=vmax, cmap=cmap, annot_kws={'size':ANN}, ax=axG, cbar_ax=axGc, mask=dexc.isna()); axGc.set_ylabel('Δ Average precision', fontsize=FS); axGc.tick_params(labelsize=FS)
for ax in (axF,axG): ax.set_xticklabels(ST, rotation=90, fontsize=FS); ax.set_yticklabels(rows, rotation=0, fontsize=FS); ax.set_xlabel(''); ax.set_ylabel('')
fig.savefig(OUT, dpi=100); print('written ->', OUT)
i5=int(np.argmin(np.abs(pc-5))); i5f=int(np.argmin(np.abs(pcf-5)))
print(f"b diagonal {np.trace(cm.values):,}/{cm.values.sum():,} = {np.trace(cm.values)/cm.values.sum():.4f}")
print(f"c realised FDR at 5 %: {g('realised_fdr_pooled')[i5]*100:.2f} %   coverage {g('coverage_heldout')[i5]:.4f}" )
print(f"d annotated fraction at 5 % (full recipe): {annot_full[i5f]:.4f}  rule-based {sw['rule_based_fraction']:.4f}; nested at 5 %: {g('annotated_fraction_all_cells')[i5]:.4f}")
print(f"e macro P {g('macro_precision')[i5]:.4f}  R {g('macro_recall')[i5]:.4f} at 5 %;  f All/All {base:.4f}")
print(inc.round(3)); print(dexc.round(3))
