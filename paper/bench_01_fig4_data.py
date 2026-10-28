"""Fig 4 v2 data for each dataset (bm, colon, chl2) and each of the five methods, on the SAME held-out
cells: coverage, assigned-only per-class precision over the published class set (a class a method never names scores 0, as
the published Fig 4 did), macro precision, class frequencies (held-out ground truth), and runtime seconds. Usage: fig4_v2_data.py bm|colon|chl2"""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, json, numpy as np, pandas as pd, joblib
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
ds=sys.argv[1]; assert ds in ('bm','colon','chl2','chl1','codex','colon16'); V=f'{QCP.OUT}'; W=f'{QCP.OUT}/workspace_out'; OUT=f'{V}/retrain/figures/fig4_v2'; os.makedirs(OUT, exist_ok=True)
METHODS=['Astir','AnnoSpat','MAPS','CELESTA','QuantCell']
def tosec(x):
    try: return float(x)
    except Exception: return None
if ds=='bm':
    P=f'{V}/retrain/QuantCellPaper'; LG=f'{V}/retrain_lgbm/QuantCellPaper'; B=f'{V}/retrain/benchmarks'
    cx=pd.read_csv(f'{P}/codex_rule_based_QuantCellPaper.csv', usecols=['cell_type']); lab_idx=np.where(cx.cell_type.values!='Other')[0]
    Xtr,Xte,ytr,yte=joblib.load(f'{P}/ComparisonsTrainTest.joblib'); pos=Xte.index.values; y=yte.values.astype(object)
    enc=joblib.load(f'{P}/label_encoder.joblib'); classes=list(enc.classes_)
    pred={'QuantCell':pd.read_csv(f'{LG}/quantcell_labels.csv').iloc[:,0].values.astype(object),
          'MAPS':enc.inverse_transform(pd.read_csv(f'{B}/MAPS/results/cell_phenotyping/pred_labels.csv', index_col=0).iloc[:,0].values).astype(object),
          'Astir':pd.read_csv(f'{B}/Astir/astir_labels.csv', index_col=0)['cell_type'].values[pos].astype(object),
          'AnnoSpat':pd.read_csv(f'{B}/AnnoSpat/outputdir/trte_labels_ELM_IMC_T1D_AnnoSpat.csv', index_col=0)['label'].astype(str).replace({'Lepr Perivascular Cells':'Lepr+ Perivascular Cells'}).values[pos].astype(object)}
    cel=pd.read_csv(f'{B}/CELESTA/celesta_bm_all.csv').sort_values('CellID'); pred['CELESTA']=cel.final_cell_type.astype(str).values[lab_idx][pos].astype(object)
    assert (pd.read_csv(f'{LG}/quantcell_true_labels.csv', index_col=0)['cell_type'].values==y).all()
    sc=json.load(open(f'{B}/benchmark_scores_v2.json')); times={'Astir':sc.get('Astir_seconds'),'AnnoSpat':sc.get('AnnoSpat_seconds'),'MAPS':sc.get('MAPS_seconds'),'CELESTA':sc.get('CELESTA_seconds'),'QuantCell':199.0}
    time_note='QuantCell runtime as measured on the paper server (LightGBM fit and 5% FDR annotation, 199 s); other methods from their own runs'
elif ds in ('colon','colon16'):
    tr=pd.read_csv(f'{V}/celesta/colon/priors/truth.csv'); y_all=tr.cell_type.values.astype(object)
    if ds=='colon16': KEEP16={'tumor cells','CD68+CD163+ macrophages','smooth muscle','granulocytes','CD8+ T cells','CD4+ T cells CD45RO+','B cells','vasculature','plasma cells','immune cells','Tregs','CD4+ T cells','CD11c+ DCs','lymphatics','NK cells','CD3+ T cells'}   # the 16 published classes with a CELESTA signature type (smooth muscle <- aSMA+ stroma); y_all=np.where(pd.Series(y_all).isin(KEEP16).values, y_all, 'Other').astype(object) # cells outside the signature excluded from truth
    classes=sorted(set(y_all)-{'Other'}); WC=f'{W}/colon16' if ds=='colon16' else f'{W}/colon'
    SAME={'vasculature':'vasculature','tumor cells':'tumor cells','aSMA+ stroma':'smooth muscle','lymphatics':'lymphatics','immune cells':'immune cells','CD3+ T cells':'CD3+ T cells','CD15+ granulocytes':'granulocytes','B cells':'B cells','CD11c+ DCs':'CD11c+ DCs','CD68+CD163+ macrophages':'CD68+CD163+ macrophages','plasma cells':'plasma cells','NK cells':'NK cells','CD8+ T cells':'CD8+ T cells','CD4+ T cells':'CD4+ T cells','CD4+ T cells CD45RO+':'CD4+ T cells CD45RO+','Tregs':'Tregs'}
    PUB=set(classes); tp=lambda s: np.array([SAME.get(v, v if v in PUB else 'Other') for v in s], dtype=object)
    lg=pd.read_csv(f'{WC}/heldout_predictions.csv'); rows=lg.row.values; y=y_all[rows]; assert (lg.true.values==y).all()   # colon QuantCell = RANDOM FOREST one closed class instead of three)
    meta=json.load(open(f'{WC}/MAPS/maps_meta.json')); mcls=np.array(meta['classes']); assert (mcls[np.load(f'{WC}/MAPS/y_test.npy')]==y).all()
    cel=pd.read_csv(f'{V}/celesta/colon/celesta_all_regions_v2.csv').sort_values('CellID'); ast=pd.read_csv(f'{V}/celesta/colon/priors/Astir/astir_labels.csv', index_col=0)['cell_type'].astype(str).values
    nm=json.load(open(f'{V}/celesta/colon/priors/AnnoSpat/name_map.json')); ann=pd.read_csv(f'{V}/celesta/colon/priors/AnnoSpat/outputdir/trte_labels_ELM_IMC_T1D_AnnoSpat.csv', index_col=0)['label'].astype(str).map(lambda v: nm.get(v,v)).values
    pred={'QuantCell':lg.pred_5pct.values.astype(object),'MAPS':mcls[pd.read_csv(f'{WC}/MAPS/results/cell_phenotyping/pred_labels.csv', index_col=0).iloc[:,0].values].astype(object),
          'CELESTA':tp(cel.final_cell_type.astype(str).values)[rows],'Astir':tp(ast)[rows],'AnnoSpat':tp(ann)[rows]}
    fm=json.load(open(f'{V}/celesta/colon/fivemethod_scores.json')); times={'Astir':fm.get('Astir_seconds'),'AnnoSpat':fm.get('AnnoSpat_seconds'),'MAPS':tosec(pd.read_csv(f'{WC}/MAPS/results/cell_phenotyping/time_elapsed_maps.csv', index_col=0).iloc[0,0]),'CELESTA':214.0,'QuantCell':(float(os.environ['QC_COLON16_FIT_S']) if ds=='colon16' else 183.0)}
    time_note='QuantCell random forest fit 183 s (colon_01); CELESTA 214 s all regions; MAPS full patience'
else:
    C=f'{V}/chl'; RAW=f'{QCP.EXT}'
    CFG={'chl2':dict(files=[f'{RAW}/cHL_2_MIBI/annotation_csv/{i}.csv' for i in range(1,7)], label='Annotation', size='cellSize', drop={'cellLabel','X_cent','Y_cent','cellSize','Annotation','Annotation_num'}, unlab={'Other'}),
         'chl1':dict(files=[f'{RAW}/cHL_1_MIBI/annotation_csv/cHL1_MIBI.csv'], label='Annotation', size='cellSize', drop={'cellLabel','Annotation','centroidX','centroidY','cellSize','identifier'}, unlab={'Other'}),
         'codex':dict(files=[f'{RAW}/cHL_CODEX/annotation_csv/cHL_CODEX_annotation.csv'], label='cellType', size='cellSize', drop={'cellLabel','X_cent','Y_cent','cellSize','cellType','DAPI.01'}, unlab={'Other','Seg Artifact'})}[ds]
    full=pd.concat([pd.read_csv(f, low_memory=False) for f in CFG['files']], ignore_index=True); full.columns=[c.strip('"') for c in full.columns]
    y_all=full[CFG['label']].astype(str).where(~full[CFG['label']].astype(str).isin(CFG['unlab']),'Other').values.astype(object); classes=sorted(set(y_all)-{'Other'})
    rows=pd.read_csv(f'{C}/{ds}_maps/heldout_rows.csv').iloc[:,0].values; y=y_all[rows]
    meta=json.load(open(f'{C}/{ds}_maps/maps_meta.json')); mcls=np.array(meta['classes']); mt=pd.read_csv(f'{C}/{ds}_maps/data/cell_phenotyping/test_features.csv', usecols=['cell_label'])['cell_label'].values; assert (mcls[mt]==y).all()
    # label-trained LightGBM @5% on the same split (recompute, ~1 min) -- identical protocol to chl2_native.py
    import re, warnings; warnings.filterwarnings('ignore'); os.environ['QC_CLF']='lgbm'; R=f'{QCP.REPO}'; sys.path.insert(0,f'{QCP.CODE}'); sys.path.insert(0,f'{QCP.PAPER}'); cwd=os.getcwd(); os.chdir(f'{QCP.CODE}')
    from quantcell import quantcell_project, find_thresholds, apply_thresholds; from _common import make_clf; from sklearn.preprocessing import StandardScaler
    markers=[c for c in full.columns if c not in CFG['drop']]; safe=lambda m: re.sub(r'[^A-Za-z0-9_]','_',m)
    F=full[markers].copy(); F.columns=[f'{safe(m)}: Cell: Mean' for m in markers]
    p=quantcell_project(); cx=F.copy(); cx.insert(0,'cell_type',y_all); p.codex=cx; p.process_data(); p.set_clf(make_clf(24)); import time as _t; t0=_t.time(); p.fit(); qc_time=_t.time()-t0
    lab=np.where(y_all!='Other')[0]; trr,_=train_test_split(np.arange(len(lab)), train_size=0.8, stratify=p.target_encoder.transform(y_all[lab]), random_state=42); sc=StandardScaler().fit(F.values[lab[trr]])
    thr=np.array(find_thresholds(p.clf,p.X_test,p.y_test,FDR_cutoff=0.05)); labs=apply_thresholds(p.clf,sc.transform(F.values[rows]),thr)
    qc=np.where(labs==-1,'Other',p.target_encoder.inverse_transform(np.where(labs==-1,0,labs))).astype(object); os.chdir(cwd)
    cel=pd.read_csv((f'{C}/chl2_priors/CELESTA/celesta_chl2_all.csv' if ds=='chl2' else f'{C}/{ds}_priors/CELESTA/celesta_all.csv')).sort_values('CellID'); assert len(cel)==len(y_all)
    ast=pd.read_csv(f'{C}/{ds}_priors/Astir/astir_labels.csv', index_col=0)['cell_type'].astype(str).values; nmp=json.load(open(f'{C}/{ds}_priors/AnnoSpat/name_map.json')) if os.path.exists(f'{C}/{ds}_priors/AnnoSpat/name_map.json') else {}; ann=pd.read_csv(f'{C}/{ds}_priors/AnnoSpat/outputdir/trte_labels_ELM_IMC_T1D_AnnoSpat.csv', index_col=0)['label'].astype(str).map(lambda v: nmp.get(v,v)).values
    pred={'QuantCell':qc,'MAPS':mcls[pd.read_csv(f'{C}/{ds}_maps/results/cell_phenotyping/pred_labels.csv', index_col=0).iloc[:,0].values].astype(object),
          'CELESTA':cel.final_cell_type.astype(str).values[rows].astype(object),'Astir':ast[rows].astype(object),'AnnoSpat':ann[rows].astype(object)}
    times={'Astir':tosec(pd.read_csv(f'{C}/{ds}_priors/Astir/astir_time.csv', index_col=0).iloc[0,0]),'AnnoSpat':tosec(open(f'{C}/{ds}_priors/AnnoSpat/time_elapsed_s.txt').read()),
           'MAPS':tosec(pd.read_csv(f'{C}/{ds}_maps/results/cell_phenotyping/time_elapsed_maps.csv', index_col=0).iloc[0,0]),'CELESTA':tosec(open(f'{C}/{ds}_priors/CELESTA/time_elapsed_s.txt').read()),'QuantCell':qc_time}
    time_note=f'all five measured on this server ({ds}); QuantCell = LightGBM fit on the labelled 80% split'
freq=pd.Series(y).value_counts(normalize=True); present=[c for c in classes if (y==c).any()]
res={'dataset':ds,'quantcell_classifier':('RandomForest' if ds in ('colon','colon16') else 'LightGBM'),'n_heldout':int(len(y)),'classes':present,'frequency':{c:float(freq[c]) for c in present},'runtime_seconds':times,'runtime_note':time_note,'methods':{}}
PRED_SAVE={}
for m in METHODS:
    yp=np.array([v if v in present else 'Other' for v in pred[m]], dtype=object); asg=yp!='Other'
    per=precision_score(y[asg], yp[asg], average=None, labels=present, zero_division=0) if asg.any() else np.zeros(len(present))
    named=[c for c in present if (yp==c).any()]
    res['methods'][m]={'coverage':float(asg.mean()),'per_class_precision':{c:float(v) for c,v in zip(present,per)},'macro_precision_all_classes':float(np.mean(per)),
                       'macro_precision_named_classes':float(np.mean([v for c,v in zip(present,per) if c in named])) if named else 0.0,'n_classes_named':len(named)}
    # extra benchmark metrics on the same cells: recall over ALL held-out cells (unassigned = missed), F1 from that
    # precision/recall pair, accuracy over all cells (unassigned = wrong) and over assigned cells only; rare (<10 % held-out
    # frequency) vs abundant (>=10 %) macro averages. Unnamed classes score 0, as in Fig 4a.
    rec=np.array([float(((yp==c)&(y==c)).sum()/max((y==c).sum(),1)) for c in present]); f1=np.where(per+rec>0, 2*per*rec/np.maximum(per+rec,1e-12), 0.0)
    rare=[i for i,c in enumerate(present) if freq[c]<0.1]; abund=[i for i,c in enumerate(present) if freq[c]>=0.1]
    res['methods'][m].update({'per_class_recall':{c:float(v) for c,v in zip(present,rec)},'per_class_f1':{c:float(v) for c,v in zip(present,f1)},
        'macro_recall_all_classes':float(np.mean(rec)),'macro_f1_all_classes':float(np.mean(f1)),
        'accuracy_all_cells':float((yp==y).mean()),'accuracy_assigned_cells':float((yp[asg]==y[asg]).mean()) if asg.any() else 0.0,
        'rare':{'n_classes':len(rare),'macro_precision':float(np.mean(per[rare])) if rare else None,'macro_recall':float(np.mean(rec[rare])) if rare else None,'macro_f1':float(np.mean(f1[rare])) if rare else None},
        'abundant':{'n_classes':len(abund),'macro_precision':float(np.mean(per[abund])) if abund else None,'macro_recall':float(np.mean(rec[abund])) if abund else None,'macro_f1':float(np.mean(f1[abund])) if abund else None}})
    PRED_SAVE[m]=yp
    print(f"  {ds:5s} {m:9s} cov {asg.mean():.3f}  macro P (all {len(present)} classes) {np.mean(per):.3f}  (named {len(named)}) {res['methods'][m]['macro_precision_named_classes']:.3f}  R {np.mean(rec):.3f}  F1 {np.mean(f1):.3f}  acc all {(yp==y).mean():.3f}  runtime {times[m]}")
json.dump(res, open(f'{OUT}/fig4_data_{ds}.json','w'), indent=2); print('wrote', f'{OUT}/fig4_data_{ds}.json')
np.savez(f'{OUT}/fig4_preds_{ds}.npz', y=y, **{m:PRED_SAVE[m] for m in METHODS}); print('wrote', f'{OUT}/fig4_preds_{ds}.npz')
