"""R1 -- tier-1 artifacts on the codex.
Fixed library (phase3_repo: C1-C10, M17, M18), renamed labels (LabelEncoder fit on the new
names -- no legacy mapping), seeded RF with the published tuned hyperparameters (copied into
model_selection/ for provenance; re-tunes later).
Recipes = paper/04 cells 4/20/21/23/24/25/41-45, as tier1.py , minus the comparisons
against the published artifacts (they differ by design)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, time, json, shutil, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, CODE, RB, NAME, PROJ, MS, CLF, make_clf, best_params
NJ=int(sys.argv[1]) if len(sys.argv)>1 else 32
os.makedirs(MS, exist_ok=True)
sys.path.insert(0, CODE); os.chdir(CODE)
from quantcell import quantcell_project
from codex_project import codex_project
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
S={}
def rec(k,v):
    S[k]=v; json.dump(S, open(f'{PROJ}/r1_summary.json','w'), indent=2, default=float); log(f"  >>> {k}: {str(v)[:200]}")


log("STAGE A -- project + process_data (renamed labels)")
p=quantcell_project(); p.initialize(base_path=RB, project_name=NAME); p.process_data()
log(f"  X_train {p.X_train.shape}  X_test {p.X_test.shape}  unlabeled {p.unlabeled_features.shape}")
joblib.dump(p.target_encoder, f'{PROJ}/label_encoder.joblib')
rec('label_encoder_classes', list(p.target_encoder.classes_))
rec('sizes', {'X_train':list(p.X_train.shape),'X_test':list(p.X_test.shape),'unlabeled':list(p.unlabeled_features.shape)})

log("STAGE B -- fit seeded RF")
p.set_clf(make_clf(NJ, p.random_seed)); rec('classifier', {'name':CLF,'params':{k:(v if isinstance(v,(int,float,str,bool,type(None))) else str(v)) for k,v in best_params().items()}})
t=time.time(); p.fit(); log(f"  fit {time.time()-t:.1f}s")

log("STAGE C -- fraction_annotated, FDR sweep 0-10% (cell 21)")
counts=p.codex.cell_type.value_counts()
frac={'Conventional': float(counts[counts.index!='Other'].sum()/counts.sum())}
for i in range(0,11):
    p.relabel_unannotated(FDR_cutoff=i/100); rc=pd.Series(p.relabeled_labels).value_counts()
    frac[f'FDR < {i}%']=float(rc[rc.index!='Other'].sum()/counts.sum()); log(f"  FDR<{i}% -> {frac[f'FDR < {i}%']:.6f}")
joblib.dump(frac, f'{PROJ}/fraction_annotated.joblib'); rec('fraction_annotated', frac)
rec('headline_conventional', frac['Conventional']); rec('headline_conv_plus_fdr5', frac['Conventional']+frac['FDR < 5%'])

log("STAGE C2 -- relabel at 5% FDR, write codex_quantcell")
p.relabel_unannotated(FDR_cutoff=0.05); joblib.dump(list(p.thresholds), f'{PROJ}/thresholds_fdr5.joblib')
p.overwrite_codex(); p.save_codex(); log(f"  wrote {PROJ}/codex_quantcell_{NAME}.csv")

log("STAGE D -- ComparisonsTrainTest (cells 24/25)")
cx=codex_project(); cx.read_csv(f'{PROJ}/codex_rule_based_{NAME}.csv', project_name=NAME)
cx.codex=cx.codex.loc[cx.codex['cell_type']!='Other',:].reset_index(drop=True)
Xtr,Xte,ytr,yte=train_test_split(cx.codex, cx.codex['cell_type'], test_size=0.2, random_state=p.random_seed, stratify=cx.codex['cell_type'])
joblib.dump((Xtr,Xte,ytr,yte), f'{PROJ}/ComparisonsTrainTest.joblib'); rec('comparisons_split', {'train':len(Xtr),'test':len(Xte)})

log("STAGE E -- nested split -> quantcell_labels (cells 41-45)")
p2=quantcell_project(); p2.initialize(base_path=RB, project_name=NAME)
Xte_w=Xte.copy(); Xte_w['cell_type']='Other'
p2.codex=pd.concat([Xtr,Xte_w],axis=0,ignore_index=True); p2.process_data()
log(f"  nested X_train {p2.X_train.shape}  inner X_test {p2.X_test.shape}  held-out {p2.unlabeled_features.shape}")
p2.set_clf(make_clf(NJ)); t=time.time(); p2.fit(); log(f"  fit {time.time()-t:.1f}s")
p2.relabel_unannotated(FDR_cutoff=0.05)
ql=pd.Series(p2.relabeled_labels); ql.to_csv(f'{PROJ}/quantcell_labels.csv', index=False)
Xte['cell_type'].to_csv(f'{PROJ}/quantcell_true_labels.csv', index=True)
tru=Xte['cell_type'].values; lab=ql.values; keep=lab!='Other'
unthr=p2.target_encoder.inverse_transform(p2.clf.predict(p2.unlabeled_features))
rec('heldout_5pct', {'n':int(len(lab)),'coverage':float(keep.mean()),'accuracy':float(accuracy_score(tru[keep],lab[keep])),
    'precision_macro':float(precision_score(tru[keep],lab[keep],average='macro',zero_division=0)),
    'recall_macro_covered':float(recall_score(tru[keep],lab[keep],average='macro',zero_division=0)),
    'realised_FDR':float(1-accuracy_score(tru[keep],lab[keep]))})
rec('heldout_unthresholded', {'accuracy':float(accuracy_score(tru,unthr)),
    'precision_macro':float(precision_score(tru,unthr,average='macro',zero_division=0)),
    'recall_macro':float(recall_score(tru,unthr,average='macro',zero_division=0)),
    'f1_macro':float(f1_score(tru,unthr,average='macro',zero_division=0))})

log("STAGE F -- train_size_analysis (5-fold, cell 14)")
p3=quantcell_project(); p3.initialize(base_path=RB, project_name=NAME)
p3.set_clf(make_clf(NJ, p3.random_seed))
t=time.time(); ts=p3.train_size_analysis(train_sizes=[0.05,0.1,0.25,0.5,0.75,1.0], n_splits=5); log(f"  done {time.time()-t:.1f}s")
joblib.dump(ts, f'{MS}/train_size_results_5fold.joblib'); rec('train_size_mean_AP', {str(k):float(np.mean(v)) for k,v in ts.items()})
rec('total_seconds', round(time.time()-T0)); log("R1 DONE")
