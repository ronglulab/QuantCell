#!/usr/bin/env python
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os
"""
Suppl Fig 9  — confusion matrices for Astir / AnnoSpat / MAPS (cell 114)
Suppl Fig 10 — per-marker exclusion, 12 additional cell types (cell 107)
Both from published artifacts; no model fitting.
"""
import pandas as pd, numpy as np, joblib, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix, f1_score
warnings.filterwarnings('ignore')

P=f'{QCP.OUT}/retrain_lgbm/QuantCellPaper'
OUT=f'{QCP.OUT}/retrain_lgbm/figures'
cell_order=['KLS','MEP','CMP','GMP','CLP','CFU-E','MkP','Erythroblasts','Erythrocytes',
 'Megakaryocytes','B Cells','CD4 T Cells','CD8 T Cells','cDC','Double Negative T Cells',
 'Monocytes','Neutrophils','Arterial ECs','Capillary ECs','Other BM ECs','Lepr+ Perivascular Cells']
enc=joblib.load(f'{P}/label_encoder.joblib')

def cmplot(cm,ax,cbar_ax):
    norm=cm.divide(cm.sum(axis=0),axis=1)
    sns.heatmap(norm, linecolor='grey', linewidth=0.01, xticklabels=cell_order, yticklabels=cell_order,
        cmap=sns.color_palette("flare", as_cmap=True), vmin=0, vmax=1,
        cbar_kws={'format':FuncFormatter(lambda x,_:f'{x*100:.0f}%')}, ax=ax, cbar_ax=cbar_ax)
    ax.set_xticklabels(cell_order, rotation=90, fontsize=8)
    ax.set_yticklabels(cell_order, rotation=0, fontsize=8)
    ax.set_ylabel('Conventional annotation', fontsize=11)

# ---------------- Suppl 10 (Suppl 9 needs benchmark predictions; not available on the retrain basis) ----------------
gko=joblib.load(f'{P}/model_selection/gene_ko_dict.joblib')
T=f'{QCP.OUT}/retrain/QuantCellPaper/base_models_tuned'; tuned=[joblib.load(f'{T}/LightGBM_y_test.joblib'),joblib.load(f'{T}/LightGBM_y_pred.joblib')]   # 10-fold CV of the tuned LightGBM
gene_order=sorted(gko.keys(), key=str.lower)
bl=joblib.load(f'{P}/model_selection/gene_ko_baseline_nested.joblib') if os.environ.get('QC_KO_BASELINE','nested')=='nested' else tuned  # matched baseline
base=np.mean([f1_score(bl[0][n],bl[1][n],average=None) for n in range(10)],axis=0)
f1m=np.zeros((len(gene_order),10,len(enc.classes_)))
for n,g in enumerate(gene_order):
    for fold in range(10):
        f1m[n,fold,:]=f1_score(gko[g][0][fold],gko[g][1][fold],average=None)
rows=[]
for n,g in enumerate(gene_order):
    for fold in range(10):
        for ct in enc.classes_:
            ci=list(enc.classes_).index(ct); rows.append((g,fold,ct,f1m[n,fold,ci]-base[ci]))
long_df=pd.DataFrame(rows,columns=['Gene','Fold','Cell Type','F1 Score'])
FLOOR,FRAC=0.02,0.25  # annotation rule mean dF1 <= -0.02 and >= 25% of the panel's largest drop (floor 0.05 -> 0.02 after the B-cell clipping fix)
ms=long_df.groupby(['Gene','Cell Type'])['F1 Score'].std().groupby('Cell Type').mean()

panels=['Arterial ECs','B Cells','CD4 T Cells','CD8 T Cells','CFU-E','CLP',
        'CMP','Capillary ECs','Double Negative T Cells','Erythroblasts','Erythrocytes','GMP']
pub={'Arterial ECs':['CD3','CD31','Endomucin'],'B Cells':['CD19'],'CD4 T Cells':['CD4'],
     'CD8 T Cells':['CD4','CD8'],'CFU-E':['CD117'],'CLP':['CD127'],'CMP':['CD3','CD34'],
     'Capillary ECs':['CD117','Endomucin'],'Double Negative T Cells':['CD4','CD8'],
     'Erythroblasts':['CD71','Ter119'],'Erythrocytes':['CD71','Ter119'],'GMP':['Fcgr']}
pub.update({'KLS':['CD117','Sca1'],'Lepr+ Perivascular Cells':['Ter119','Lepr'],'MEP':['CD117','CD3','CD34'],
     'Megakaryocytes':['CD4','CD41','Ter119'],'MkP':['CD117','CD4','CD41'],'Monocytes':['CD11b','CD115'],
     'Neutrophils':['CD4','F4_80','Ly6g'],'Other BM ECs':['CD105','CD4','Ter119'],'cDC':['CD11c','MHCII']})
PAGES=[(panels,4,3,'suppl10_lgbm.png'),(['KLS','Lepr+ Perivascular Cells','MEP','Megakaryocytes','MkP','Monocytes','Neutrophils','Other BM ECs','cDC'],3,3,'suppl10_lgbm_p2.png')]
FS=24; plt.rcParams["font.size"]=FS  # same fonts as the main figures
for panels,nr,nc,fname in PAGES:
  fig,axs=plt.subplots(nr,nc,figsize=(26,nr*6.5+2)); axs=axs.flatten()
  print("\nSuppl 10 — annotated markers per panel:")
  ok=0
  for i,ct in enumerate(panels):
      sub=long_df[long_df['Cell Type']==ct]
      sns.swarmplot(data=sub,x='Gene',y='F1 Score',ax=axs[i],color='black',size=3,order=gene_order)
      mv=sub.groupby('Gene')['F1 Score'].mean().reindex(gene_order)
      sns.pointplot(x=mv.index,y=mv.values,ax=axs[i],color='red',errorbar=None,linestyle='none',order=gene_order)
      axs[i].axhline(0,color='black',linestyle='--',linewidth=1.5)
      axs[i].set_title(ct); axs[i].set_xlabel(''); axs[i].set_ylabel('Δ F1 Score')
      axs[i].set_xticklabels([' ']*len(gene_order))
      axs[i].spines[['top','right']].set_visible(False)
      thr=max(FLOOR,FRAC*(-sub.groupby('Gene')['F1 Score'].mean().min())); flagged=[]; offset=-1; ylim=0
      for n,g in enumerate(gene_order):
          md=sub[sub.Gene==g]['F1 Score'].mean(); mn=sub[sub.Gene==g]['F1 Score'].min()
          if md<=-thr:
              flagged.append(g)
              if n<5: offset=1
              if n>len(gene_order)-4: offset=-1
              if g=='Endomucin': offset=1
              axs[i].text(n+1.6*offset+0.2*len(g)*offset,1.1*mn,g,ha='center',va='bottom',fontsize=FS)
              offset*=-1
              if mn<ylim: ylim=mn
      axs[i].set_ylim([1.2*min(ylim, sub['F1 Score'].min()),None])  # never clip: bottom from the whole panel, not only labelled markers
      m=set(flagged)==set(pub[ct]); ok+=m
      print(f"  {ct:<26} mine {str(flagged):<34} pub {str(pub[ct]):<30} {'OK' if m else '<--'}")
  print(f"\n  panels whose annotated markers equal the PUBLISHED figure: {ok}/{len(panels)} (differences expected: M15 fix + new labels)")
  fig.tight_layout(); fig.savefig(f'{OUT}/{fname}',dpi=60)
  print(f"  written -> {OUT}/{fname}")
