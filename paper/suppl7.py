#!/usr/bin/env python
"""Supplementary Fig 7 — annotated confusion matrix (counts + column-normalised %)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import pandas as pd, numpy as np, joblib, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

P=f'{QCP.OUT}/retrain_lgbm/QuantCellPaper'
OUT=f'{QCP.OUT}/retrain_lgbm/figures/suppl7_lgbm.png'
cell_order=['KLS','MEP','CMP','GMP','CLP','CFU-E','MkP','Erythroblasts','Erythrocytes',
 'Megakaryocytes','B Cells','CD4 T Cells','CD8 T Cells','cDC','Double Negative T Cells',
 'Monocytes','Neutrophils','Arterial ECs','Capillary ECs','Other BM ECs','Lepr+ Perivascular Cells']
enc=joblib.load(f'{P}/label_encoder.joblib')
lab=pd.read_csv(f'{P}/quantcell_labels.csv').iloc[:,0]
tru=pd.read_csv(f'{P}/quantcell_true_labels.csv', index_col=0).loc[:,'cell_type']
for o,n in {'Arteries':'Arterial ECs','Capillaries/Sinusoids':'Capillary ECs',
            'Endothelial Niche':'Other BM ECs','Perivascular Niche':'Lepr+ Perivascular Cells',
            'Preplatelets':'Megakaryocytes'}.items():
    lab[lab==o]=n; tru[tru==o]=n
cm=pd.DataFrame(confusion_matrix(tru,lab,labels=enc.classes_),
                index=enc.classes_,columns=enc.classes_).loc[cell_order,cell_order]
norm=cm.divide(cm.sum(axis=0),axis=1)
annot=cm.astype(str)+'\n('+norm.round(2).astype(str)+')'

FS=24; ANN=16  # 24 pt like the main figures; 2026-10-23: canvas shrunk 30x27 -> 24x22 so the labels read larger (cell annotations 16 pt to fit ~0.9-in cells)
plt.rcParams['font.size']=FS
fig,ax=plt.subplots(figsize=(24,22))
sns.heatmap(norm, annot=annot, fmt='', linecolor='grey', linewidth=0.01,
    xticklabels=cell_order, yticklabels=cell_order,
    cmap=sns.color_palette("flare", as_cmap=True),
    annot_kws={"size":ANN}, vmin=0, vmax=1,
    cbar_kws={'format':FuncFormatter(lambda x,_:f'{x*100:.0f}%')}, ax=ax)
ax.set_xticklabels(cell_order, rotation=90, fontsize=FS)
ax.set_yticklabels(cell_order, rotation=0, fontsize=FS)
ax.set_ylabel('Rule-based annotation', fontsize=FS)
ax.set_xlabel('QuantCell', fontsize=FS); ax.collections[0].colorbar.ax.tick_params(labelsize=FS)
fig.tight_layout(); fig.savefig(OUT, dpi=100)
print(f"written -> {OUT}")
print(f"diagonal {np.trace(cm.values):,} / {cm.values.sum():,} = {np.trace(cm.values)/cm.values.sum()*100:.4f}%")
