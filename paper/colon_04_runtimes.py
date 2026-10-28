"""Five-method colon table . Two accountings on the SAME held-out 48,901 cells (B1 split):
 (A) ours: coverage + assigned-only macro P/R/F1/acc in the published 27-type space (prior-driven tools' 16 CELESTA types
 mapped by same name; 'aSMA+ stroma' -> 'smooth muscle' per the draft; everything else counts as Other/abstain);
 (B) CELESTA's own protocol (Zhang 2022): per CELESTA type, one-vs-rest vs the same-named published type, per region,
 types with <5 cells in a region excluded, precision / F1 / 'accuracy' = (TP+TN)/all, mean +- SD over regions.
Prior-driven tools (CELESTA, Astir, AnnoSpat) are additionally scored under (B) on ALL labelled cells / all 70 regions."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import json, numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
C=f'{QCP.OUT}/celesta'; O=f'{C}/colon/priors'; W=f'{QCP.OUT}/workspace_out'
truth=pd.read_csv(f'{O}/truth.csv'); n=len(truth); y_all=truth.cell_type.values; reg_all=truth.Region.values
SAME={'vasculature':'vasculature','tumor cells':'tumor cells','aSMA+ stroma':'smooth muscle','lymphatics':'lymphatics','immune cells':'immune cells','CD3+ T cells':'CD3+ T cells',
      'CD15+ granulocytes':'granulocytes','B cells':'B cells','CD11c+ DCs':'CD11c+ DCs','CD68+CD163+ macrophages':'CD68+CD163+ macrophages','plasma cells':'plasma cells','NK cells':'NK cells',
      'CD8+ T cells':'CD8+ T cells','CD4+ T cells':'CD4+ T cells','CD4+ T cells CD45RO+':'CD4+ T cells CD45RO+','Tregs':'Tregs'}
# predictions, all in the published vocabulary (or 'Other' = abstain / unmapped)
PUB=set(y_all)
def to_pub(s): return np.array([SAME.get(v, v if v in PUB else 'Other') for v in s], dtype=object)
cel=pd.read_csv(f'{C}/colon/celesta_all_regions_v2.csv').sort_values('CellID'); assert (cel.CellID.values==truth.CellID.values).all()
P={'CELESTA':to_pub(cel.final_cell_type.astype(str).values)}
ast=pd.read_csv(f'{O}/Astir/astir_labels.csv', index_col=0); assert len(ast)==n; P['Astir']=to_pub(ast['cell_type'].astype(str).values)
nm=json.load(open(f'{O}/AnnoSpat/name_map.json')); ann=pd.read_csv(f'{O}/AnnoSpat/outputdir/trte_labels_ELM_IMC_T1D_AnnoSpat.csv', index_col=0); assert len(ann)==n
P['AnnoSpat']=to_pub(ann['label'].astype(str).map(lambda v: nm.get(v, v)).values)
for k in P: print(f'  {k:9s} vocabulary after mapping:', pd.Series(P[k]).value_counts().head(20).to_dict())
# held-out cells + label-trained methods
rf=pd.read_csv(f'{W}/colon/heldout_predictions.csv'); lg=pd.read_csv(f'{W}/colon_lgbm/heldout_predictions.csv'); assert (rf.row.values==lg.row.values).all()
rows=rf.row.values; assert (y_all[rows]==rf.true.values).all(); reg_ho=reg_all[rows]; y_ho=y_all[rows]
meta=json.load(open(f'{W}/colon/MAPS/maps_meta.json')); mcls=meta['classes']; yte=np.load(f'{W}/colon/MAPS/y_test.npy'); assert (np.array(mcls)[yte]==y_ho).all()
maps=np.array(mcls)[pd.read_csv(f'{W}/colon/MAPS/results/cell_phenotyping/pred_labels.csv', index_col=0).iloc[:,0].values]
H={'QuantCell RF @5% FDR':rf.pred_5pct.values.astype(object),'QuantCell LightGBM @5% FDR':lg.pred_5pct.values.astype(object),'QuantCell RF (no threshold)':rf.pred.values.astype(object),
   'QuantCell LightGBM (no threshold)':lg.pred.values.astype(object),'MAPS':maps.astype(object),'CELESTA':P['CELESTA'][rows],'Astir':P['Astir'][rows],'AnnoSpat':P['AnnoSpat'][rows]}
def ours(y, yp):
    asg=yp!='Other'; labs=sorted(set(y))
    return {'n':int(len(y)),'coverage':float(asg.mean()),'precision_macro':float(precision_score(y[asg],yp[asg],average='macro',labels=labs,zero_division=0)),
            'recall_macro':float(recall_score(y[asg],yp[asg],average='macro',labels=labs,zero_division=0)),'f1_macro':float(f1_score(y[asg],yp[asg],average='macro',labels=labs,zero_division=0)),
            'accuracy':float(accuracy_score(y[asg],yp[asg])),'abstentions_wrong_precision_macro':float(precision_score(y,yp,average='macro',labels=labs,zero_division=0))}
def celesta_protocol(y, yp, reg):
    per={}
    for ct,pub in SAME.items():
        P_,F_,A_=[],[],[]
        for r in np.unique(reg):
            m=reg==r; t=(y[m]==pub); p=(yp[m]==pub)
            if t.sum()<5: continue
            tp=int((t&p).sum()); fp=int((~t&p).sum()); fn=int((t&~p).sum()); tn=int((~t&~p).sum())
            pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
            P_.append(pr); F_.append(2*pr*rc/(pr+rc) if pr+rc else 0.0); A_.append((tp+tn)/(tp+tn+fp+fn))
        if P_: per[ct]={'n_regions':len(P_),'precision':float(np.mean(P_)),'precision_sd':float(np.std(P_)),'f1':float(np.mean(F_)),'f1_sd':float(np.std(F_)),'rand_accuracy':float(np.mean(A_))}
    return {'per_type':per,'mean_over_types':{k:float(np.mean([v[k] for v in per.values()])) for k in ('precision','f1','rand_accuracy')},'n_types_scored':len(per)}
res={'held_out_ours':{k:ours(y_ho,v) for k,v in H.items()},'held_out_celesta_protocol':{k:celesta_protocol(y_ho,v,reg_ho) for k,v in H.items()},
     'all_cells_celesta_protocol':{k:celesta_protocol(y_all[y_all!='Other'],P[k][y_all!='Other'],reg_all[y_all!='Other']) for k in P},
     'all_cells_ours':{k:ours(y_all[y_all!='Other'],P[k][y_all!='Other']) for k in P},'name_pairs':SAME,'n_heldout':int(len(rows))}
for f in ('Astir/astir_time.csv',): 
    try: res['Astir_seconds']=float(pd.read_csv(f'{O}/{f}', index_col=0).iloc[0,0])
    except Exception: pass
try: res['AnnoSpat_seconds']=float(open(f'{O}/AnnoSpat/time_elapsed_s.txt').read())
except Exception: pass
json.dump(res, open(f'{C}/colon/fivemethod_scores.json','w'), indent=2)
print('\n== (A) ours, held-out 48,901 cells, 27-type space'); [print(f"  {k:34s} cov {v['coverage']:.3f}  P {v['precision_macro']:.3f}  R {v['recall_macro']:.3f}  F1 {v['f1_macro']:.3f}  acc {v['accuracy']:.3f}  abst-wrong P {v['abstentions_wrong_precision_macro']:.3f}") for k,v in res['held_out_ours'].items()]
print('== (B) CELESTA protocol, held-out, per-region mean over scored types'); [print(f"  {k:34s} types {v['n_types_scored']:2d}  P {v['mean_over_types']['precision']:.3f}  F1 {v['mean_over_types']['f1']:.3f}  Rand-acc {v['mean_over_types']['rand_accuracy']:.3f}") for k,v in res['held_out_celesta_protocol'].items()]
print('== (B) CELESTA protocol, ALL labelled cells, prior-driven tools'); [print(f"  {k:34s} types {v['n_types_scored']:2d}  P {v['mean_over_types']['precision']:.3f}  F1 {v['mean_over_types']['f1']:.3f}  Rand-acc {v['mean_over_types']['rand_accuracy']:.3f}") for k,v in res['all_cells_celesta_protocol'].items()]
print('== (A) ours, ALL labelled cells, prior-driven tools'); [print(f"  {k:34s} cov {v['coverage']:.3f}  P {v['precision_macro']:.3f}  R {v['recall_macro']:.3f}  F1 {v['f1_macro']:.3f}") for k,v in res['all_cells_ours'].items()]
