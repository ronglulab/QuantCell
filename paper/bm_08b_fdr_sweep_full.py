""" Dense FDR-cutoff sweep, bone marrow, FULL-DATA recipe for Fig 3 panel d panel d must use the method that
actually annotates the unannotated cells). Exactly the published Fig 3b / notebook cell 21 (`relabel_unannotated`) set-up: project
initialised on the whole codex, process_data() (80/20 stratified split of the rule-labelled cells), one LightGBM fit on the 80 %,
thresholds from the 20 % at every cutoff 0..30 % in 0.1 % steps (find_thresholds_at_varying_FDR == find_thresholds per cutoff),
applied to the 144,950 cells without a rule-based label. Annotated fraction of ALL cells = rule-based + added (r1_tier1.py STAGE C
accounting; the 5 % value is asserted against fraction_annotated.joblib). Panels c/e keep the nested sweep (fdr_sweep_bm_v2.json).
Output retrain_lgbm/figures/fdr_sweep_bm_full.json."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, time, json, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, CODE, RB, NAME, PROJ, make_clf, CLF
assert CLF=='lgbm', 'run with QC_CLF=lgbm'
sys.path.insert(0,CODE); os.chdir(CODE)
from quantcell import quantcell_project, find_thresholds, find_thresholds_at_varying_FDR, apply_thresholds
T0=time.time(); NJ=int(sys.argv[1]) if len(sys.argv)>1 else 16
def log(m): print(f"[{time.time()-T0:7.1f}s][fdr_sweep_full bm] {m}", flush=True)
p=quantcell_project(); p.initialize(base_path=RB, project_name=NAME); p.process_data()
counts=p.codex.cell_type.value_counts(); n_all=int(counts.sum()); rule_frac=float(counts[counts.index!='Other'].sum()/n_all); n_oth=int(p.unlabeled_features.shape[0])
log(f"X_train {p.X_train.shape}  X_test {p.X_test.shape}  unlabelled (no rule) {n_oth:,}  all cells {n_all:,}  rule-based fraction {rule_frac:.4f}")
p.set_clf(make_clf(NJ)); t=time.time(); p.fit(); log(f"fit {time.time()-t:.0f}s")
cuts,THR=find_thresholds_at_varying_FDR(p.clf,p.X_test,p.y_test,min_FDR=0.0,max_FDR=0.30,number_points=301)
chk=np.array(find_thresholds(p.clf,p.X_test,p.y_test,FDR_cutoff=0.05)); assert np.allclose(chk, THR[:,50]), 'varying-FDR thresholds differ from find_thresholds at 5 %'
P=p.clf.predict_proba(p.unlabeled_features); am=P.argmax(1); top=P[np.arange(len(am)),am]
# apply_thresholds equivalence check at 5 % (the pipeline's own function) on the cached probabilities
lab5=np.array(apply_thresholds(p.clf,p.unlabeled_features,list(THR[:,50]))); assert ((lab5!=-1)==(top>=THR[:,50][am])).all(), 'cached-proba gating differs from apply_thresholds'
fa=joblib.load(f'{PROJ}/fraction_annotated.joblib'); log(f"fraction_annotated.joblib: Conventional {fa['Conventional']:.4f}  FDR<5% {fa['FDR < 5%']:.4f}")
res={'protocol':'full-data (published Fig 3b / relabel_unannotated): fit on 80 % of rule-labelled cells, thresholds on the 20 %, applied to all unlabelled cells',
     'n_all_cells':n_all,'n_no_rule':n_oth,'n_train':int(p.X_train.shape[0]),'n_thresh':int(p.X_test.shape[0]),'rule_based_fraction':rule_frac,'cutoffs':[]}
for j,c in enumerate(cuts):
    added=float((top>=THR[:,j][am]).sum()/n_all); res['cutoffs'].append({'cutoff':float(c),'added_fraction_all_cells':added,'annotated_fraction_all_cells':rule_frac+added})
    if j%50==0 or abs(c-0.05)<1e-9: log(f"  cutoff {c*100:5.1f} %: added {added:.4f}  annotated(all) {rule_frac+added:.4f}")
a5=res['cutoffs'][50]; assert abs(a5['cutoff']-0.05)<1e-9 and abs(a5['added_fraction_all_cells']-fa['FDR < 5%'])<1e-6 and abs(rule_frac-fa['Conventional'])<1e-6, f"5 % mismatch vs fraction_annotated.joblib: {a5} vs {fa}"
log("5 % column matches fraction_annotated.joblib (Fig 3b)")
OUT=f'{QCP.OUT}/retrain_lgbm/figures'; SUFN='' if NAME=='QuantCellPaper' else f'_{NAME}';  # per-project output
json.dump(res, open(f'{OUT}/fdr_sweep_bm_full{SUFN}.json','w'), indent=1); log(f"wrote {OUT}/fdr_sweep_bm_full{SUFN}.json"); log("DONE")
