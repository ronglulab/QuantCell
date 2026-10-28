"""v2 of bm_benchmarks_score.py: adds CELESTA (paper combos as prior) and QuantCell-LightGBM to the BM benchmark table;
same two accountings and the same 20,154 held-out cells; Suppl 9 v2 with four competitor panels. Writes benchmark_scores_v2.json."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, json, numpy as np, pandas as pd, joblib, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _common import PROJ, NAME
B=f'{QCP.OUT}/retrain/benchmarks'; F=f'{QCP.OUT}/retrain/figures'; LG=f'{QCP.OUT}/retrain_lgbm/QuantCellPaper'
enc=joblib.load(f'{PROJ}/label_encoder.joblib'); classes=list(enc.classes_)
cell_order=['KLS','MEP','CMP','GMP','CLP','CFU-E','MkP','Erythroblasts','Erythrocytes','Megakaryocytes','B Cells','CD4 T Cells','CD8 T Cells','cDC','Double Negative T Cells','Monocytes','Neutrophils','Arterial ECs','Capillary ECs','Other BM ECs','Lepr+ Perivascular Cells']
cx=pd.read_csv(f'{PROJ}/codex_rule_based_{NAME}.csv', usecols=['cell_type']); lab_idx=np.where(cx.cell_type.values!='Other')[0]
Xtr,Xte,ytr,yte=joblib.load(f'{PROJ}/ComparisonsTrainTest.joblib'); pos=Xte.index.values
qc_true=pd.read_csv(f'{PROJ}/quantcell_true_labels.csv', index_col=0)['cell_type'].values; qc_pred=pd.read_csv(f'{PROJ}/quantcell_labels.csv').iloc[:,0].values; assert (qc_true==yte.values).all()
lg_true=pd.read_csv(f'{LG}/quantcell_true_labels.csv', index_col=0)['cell_type'].values; lg_pred=pd.read_csv(f'{LG}/quantcell_labels.csv').iloc[:,0].values; assert (lg_true==qc_true).all()
maps_true=enc.inverse_transform(pd.read_csv(f'{B}/MAPS/data/cell_phenotyping/test_features.csv', usecols=['cell_label'])['cell_label'].values)
maps_pred=enc.inverse_transform(pd.read_csv(f'{B}/MAPS/results/cell_phenotyping/pred_labels.csv', index_col=0).iloc[:,0].values)
ast_true=pd.read_csv(f'{B}/Astir/astir_true_labels.csv')['cell_type'].values; ast_pred=pd.read_csv(f'{B}/Astir/astir_labels.csv', index_col=0)['cell_type'].values
ann_true=pd.read_csv(f'{B}/AnnoSpat/annospat_true_labels.csv', index_col=0)['cell_type'].values
ann_pred=pd.read_csv(f'{B}/AnnoSpat/outputdir/trte_labels_ELM_IMC_T1D_AnnoSpat.csv', index_col=0)['label'].astype(str).replace({'Lepr Perivascular Cells':'Lepr+ Perivascular Cells'}).values
# CELESTA: all 245,719 cells in codex order; restrict to labelled cells (codex order == Astir/AnnoSpat input order)
cel=pd.read_csv(f'{B}/CELESTA/celesta_bm_all.csv').sort_values('CellID'); assert len(cel)==len(cx) and (cel.CellID.values==np.arange(len(cx))).all()
cel_pred_all=cel.final_cell_type.astype(str).values; cel_true=cx.cell_type.values[lab_idx]; cel_pred=cel_pred_all[lab_idx]; assert (cel_true==ast_true).all()
print('CELESTA label vocabulary:', pd.Series(cel_pred_all).value_counts().to_dict())
def score(name, y, yp):
    yp=np.array([p if p in classes else 'Other' for p in yp], dtype=object); asg=yp!='Other'; labs=[c for c in classes if (y==c).any()]
    r={'n':int(len(y)),'coverage':float(asg.mean()),
       'assigned_only':{'precision_macro':float(precision_score(y[asg],yp[asg],average='macro',labels=labs,zero_division=0)),'recall_macro':float(recall_score(y[asg],yp[asg],average='macro',labels=labs,zero_division=0)),
                        'f1_macro':float(f1_score(y[asg],yp[asg],average='macro',labels=labs,zero_division=0)),'accuracy':float(accuracy_score(y[asg],yp[asg]))},
       'abstentions_wrong':{'precision_macro':float(precision_score(y,yp,average='macro',zero_division=0)),'recall_macro':float(recall_score(y,yp,average='macro',labels=labs,zero_division=0)),'accuracy':float(accuracy_score(y,yp))}}
    print(f"  {name:<36} n {r['n']:>7,}  coverage {r['coverage']:.4f} | assigned-only P {r['assigned_only']['precision_macro']:.4f} R {r['assigned_only']['recall_macro']:.4f} F1 {r['assigned_only']['f1_macro']:.4f} acc {r['assigned_only']['accuracy']:.4f} | abstentions-wrong P {r['abstentions_wrong']['precision_macro']:.4f} acc {r['abstentions_wrong']['accuracy']:.4f}", flush=True)
    return r
res={'QuantCell_RF_heldout_5pctFDR':score('QuantCell RF @5% FDR (held-out)', qc_true, qc_pred),'QuantCell_LGBM_heldout_5pctFDR':score('QuantCell LightGBM @5% FDR (held-out)', lg_true, lg_pred),
     'MAPS_heldout':score('MAPS (held-out)', maps_true, maps_pred),
     'CELESTA_all_labelled':score('CELESTA (all labelled cells)', cel_true, cel_pred),'CELESTA_same_heldout_cells':score('CELESTA (same held-out cells)', cel_true[pos], cel_pred[pos]),
     'Astir_all_labelled':score('Astir (all labelled cells)', ast_true, ast_pred),'Astir_same_heldout_cells':score('Astir (same held-out cells)', ast_true[pos], ast_pred[pos]),
     'AnnoSpat_all_labelled':score('AnnoSpat (all labelled cells)', ann_true, ann_pred),'AnnoSpat_same_heldout_cells':score('AnnoSpat (same held-out cells)', ann_true[pos], ann_pred[pos]),
     'CELESTA_per_class_precision_same_heldout':{c:float(v) for c,v in zip(classes, precision_score(cel_true[pos], np.array([p if p in classes else 'Other' for p in cel_pred[pos]],dtype=object), average=None, labels=classes, zero_division=0))}}
old=json.load(open(f'{B}/benchmark_scores.json')); res.update({k:v for k,v in old.items() if k.endswith('_seconds')})
try: res['CELESTA_seconds']=float(open(f'{B}/CELESTA/time_elapsed_s.txt').read())
except Exception: pass
json.dump(res, open(f'{B}/benchmark_scores_v2.json','w'), indent=2)
def cmplot(cm,ax,cbar_ax):
    norm=cm.divide(cm.sum(axis=0),axis=1)
    sns.heatmap(norm, linecolor='grey', linewidth=0.01, xticklabels=cell_order, yticklabels=cell_order, cmap=sns.color_palette("flare", as_cmap=True), vmin=0, vmax=1, cbar_kws={'format':FuncFormatter(lambda x,_:f'{x*100:.0f}%')}, ax=ax, cbar_ax=cbar_ax)
    ax.set_xticklabels(cell_order, rotation=90, fontsize=FS); ax.set_yticklabels(cell_order, rotation=0, fontsize=FS); ax.set_ylabel('Rule-based annotation', fontsize=FS); cbar_ax.tick_params(labelsize=FS)
FS=24; plt.rcParams["font.size"]=FS  # same fonts as the main figures
fig,axs=plt.subplots(2,2,figsize=(30,28)); axs=axs.flatten(); CAX=fig.add_axes([0.93,0.3,0.015,0.4])
for ax,(name,t,l) in zip(axs,[('AnnoSpat',ann_true,ann_pred),('Astir',ast_true,ast_pred),('CELESTA',cel_true,cel_pred),('MAPS',maps_true,maps_pred)]):
    l=np.array([p if p in classes else 'Other' for p in l],dtype=object)
    cm=pd.DataFrame(confusion_matrix(t,l,labels=classes),index=classes,columns=classes).loc[cell_order,cell_order]
    cmplot(cm,ax,CAX); ax.set_xlabel(name, fontsize=FS)
fig.tight_layout(rect=[0,0,0.91,1]); fig.savefig(f'{F}/suppl9_retrain_v2.png', dpi=60); print(f"written -> {F}/suppl9_retrain_v2.png"); print("DONE")
