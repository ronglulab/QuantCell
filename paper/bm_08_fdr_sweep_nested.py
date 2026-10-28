""" Dense FDR-cutoff sweep, bone marrow Fig 3b replaced by the sweep; linear axes, many points; panel b with
the rule-based baseline; panel c macro precision + recall). Same nested set-up as fdr_sweep_bm.py (A146): outer 80/20 of labelled
cells seed 42, inner 80/20 for thresholds, one LightGBM fit. Cutoffs 0..30 % in 0.1 % steps (find_thresholds_at_varying_FDR, the
pipeline's own function, == find_thresholds at each cutoff). Thresholds applied to (i) the 20,154 held-out labelled cells -> realised
FDR, coverage, macro P / R / F1 (unassigned = missed); (ii) the 144,950 cells WITHOUT a rule-based label -> fraction of ALL 245,719
cells annotated = rule-based 0.4101 + added (the Fig 3b accounting). Output retrain_lgbm/figures/fdr_sweep_bm_v2.json."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, time, json, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, CODE, RB, NAME, PROJ, make_clf, CLF
assert CLF=='lgbm', 'run with QC_CLF=lgbm'
sys.path.insert(0,CODE); os.chdir(CODE)
from quantcell import quantcell_project, find_thresholds, find_thresholds_at_varying_FDR
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
T0=time.time(); NJ=int(sys.argv[1]) if len(sys.argv)>1 else 16
def log(m): print(f"[{time.time()-T0:7.1f}s][fdr_sweep_v2 bm] {m}", flush=True)
Xtr,Xte,ytr,yte=joblib.load(f'{PROJ}/ComparisonsTrainTest.joblib')
full=pd.read_csv(f'{PROJ}/codex_rule_based_{NAME}.csv', low_memory=False); oth=full[full.cell_type=='Other'].copy(); n_all=len(full); rule_frac=float((full.cell_type!='Other').mean())
assert set(oth.columns)==set(Xtr.columns); oth=oth[Xtr.columns]
Xte_w=Xte.copy(); Xte_w['cell_type']='Other'
p=quantcell_project(); p.initialize(base_path=RB, project_name=NAME); p.codex=pd.concat([Xtr,Xte_w,oth],axis=0,ignore_index=True); p.process_data()
nte=len(Xte); assert p.unlabeled_features.shape[0]==nte+len(oth)
log(f"nested X_train {p.X_train.shape}  inner X_test {p.X_test.shape}  held-out {nte:,}  unlabelled (no rule) {len(oth):,}  all cells {n_all:,}  rule-based fraction {rule_frac:.4f}")
p.set_clf(make_clf(NJ)); t=time.time(); p.fit(); log(f"fit {time.time()-t:.0f}s")
enc=p.target_encoder; classes=list(enc.classes_); y=enc.transform(yte.values); n_cls=len(classes)
assert (pd.read_csv(f'{PROJ}/quantcell_true_labels.csv', index_col=0)['cell_type'].values==yte.values).all()
U=p.unlabeled_features; P_ho=p.clf.predict_proba(U[:nte]); P_ot=p.clf.predict_proba(U[nte:]); am_ho=P_ho.argmax(1); am_ot=P_ot.argmax(1)
top_ho=P_ho[np.arange(nte),am_ho]; top_ot=P_ot[np.arange(len(oth)),am_ot]
pred=am_ho; ref={'macro_f1':float(f1_score(y,pred,average='macro')),'accuracy':float(accuracy_score(y,pred)),'macro_precision':float(precision_score(y,pred,average='macro',zero_division=0)),'macro_recall':float(recall_score(y,pred,average='macro',zero_division=0))}
cuts,THR=find_thresholds_at_varying_FDR(p.clf,p.X_test,p.y_test,min_FDR=0.0,max_FDR=0.30,number_points=301)   # (n_cls, 301)
chk=np.array(find_thresholds(p.clf,p.X_test,p.y_test,FDR_cutoff=0.05)); assert np.allclose(chk, THR[:,50]), 'varying-FDR thresholds differ from find_thresholds at 5 %'
log(f"thresholds for {len(cuts)} cutoffs; 5 % column matches find_thresholds")
freq={c:float((y==i).mean()) for i,c in enumerate(classes)}
res={'classes':classes,'heldout_frequency':freq,'n_heldout':int(nte),'n_all_cells':int(n_all),'n_no_rule':int(len(oth)),'rule_based_fraction':rule_frac,'argmax':ref,'cutoffs':[]}
for j,c in enumerate(cuts):
    thr=THR[:,j]; asg=top_ho>=thr[am_ho]; lab=np.where(asg,am_ho,-1)
    per_p=precision_score(y[asg],lab[asg],average=None,labels=list(range(n_cls)),zero_division=0) if asg.any() else np.zeros(n_cls)
    rec=np.array([float(((lab==i)&(y==i)).sum()/max((y==i).sum(),1)) for i in range(n_cls)]); f1=np.where(per_p+rec>0,2*per_p*rec/np.maximum(per_p+rec,1e-12),0.0)
    added=float((top_ot>=thr[am_ot]).sum()/n_all)
    r={'cutoff':float(c),'coverage_heldout':float(asg.mean()),'realised_fdr_pooled':float(1-accuracy_score(y[asg],lab[asg])) if asg.any() else None,
       'macro_precision':float(per_p.mean()),'macro_recall':float(rec.mean()),'macro_f1':float(f1.mean()),'accuracy_all_heldout':float((lab==y).mean()),
       'added_fraction_all_cells':added,'annotated_fraction_all_cells':rule_frac+added,'realised_fdr_per_class':{cl:float(1-v) for cl,v in zip(classes,per_p)}}
    res['cutoffs'].append(r)
    if j%50==0 or abs(c-0.05)<1e-9: log(f"  cutoff {c*100:5.1f} %: cov(held-out) {r['coverage_heldout']:.4f}  realised FDR {r['realised_fdr_pooled'] if r['realised_fdr_pooled'] is not None else float('nan'):.4f}  P {r['macro_precision']:.4f}  R {r['macro_recall']:.4f}  F1 {r['macro_f1']:.4f}  annotated(all) {r['annotated_fraction_all_cells']:.4f}")
OUT=f'{QCP.OUT}/retrain_lgbm/figures'; SUFN='' if NAME=='QuantCellPaper' else f'_{NAME}';  # per-project output
json.dump(res, open(f'{OUT}/fdr_sweep_bm_v2{SUFN}.json','w'), indent=1); log(f"wrote {OUT}/fdr_sweep_bm_v2{SUFN}.json"); log("DONE")
