""" Matched baseline for the marker knockouts ). Identical set-up to r2_gene_ko.py --
same nested 80 % population, same process_data(), same make_clf(), same seeded StratifiedKFold via cv_fit_pred -- with NO
columns excluded. Result = (y_test, y_pred, y_proba) per fold, saved to model_selection/gene_ko_baseline_nested.joblib, so the
Fig 5b / Suppl 10 dF1 = knockout F1 - baseline F1 compares like with like (the published notebook used the full-population CV)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, time, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, CODE, RB, NAME, PROJ, MS, make_clf, CLF
assert CLF=='lgbm', 'run with QC_CLF=lgbm'
sys.path.insert(0,CODE); os.chdir(CODE)
from quantcell import quantcell_project
from codex_project import codex_project
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
T0=time.time(); NJ=int(sys.argv[1]) if len(sys.argv)>1 else 16
def log(m): print(f"[{time.time()-T0:7.1f}s][ko_baseline] {m}", flush=True)
cx=codex_project(); cx.read_csv(f'{PROJ}/codex_rule_based_{NAME}.csv', project_name=NAME)
cx.codex=cx.codex.loc[cx.codex['cell_type']!='Other',:].reset_index(drop=True)
Xtr,Xte,_,_=train_test_split(cx.codex, cx.codex['cell_type'], test_size=0.2, random_state=42, stratify=cx.codex['cell_type'])
Xte_w=Xte.copy(); Xte_w['cell_type']='Other'
p=quantcell_project(); p.initialize(base_path=RB, project_name=NAME)
p.codex=pd.concat([Xtr,Xte_w],axis=0,ignore_index=True); p.process_data()
log(f"nested X_train {p.X_train.shape}; held-out {p.unlabeled_features.shape}; columns {len(p.column_names)}")
p.set_clf(make_clf(NJ))
res=list(p.cv_fit_pred(name='LGBM_ko_baseline', excluded_cols=[]))
joblib.dump(res, f'{MS}/gene_ko_baseline_nested.joblib')
# sanity: fold structure identical to a knockout entry
gko=joblib.load(f'{MS}/gene_ko_dict.joblib'); k=sorted(gko)[0]
same=all(np.array_equal(res[0][i], gko[k][0][i]) for i in range(10)); log(f"fold y_test identical to knockout '{k}': {same}")
enc=joblib.load(f'{PROJ}/label_encoder.joblib'); f1=np.mean([f1_score(res[0][i],res[1][i],average=None) for i in range(10)],axis=0)
T=f'{QCP.OUT}/retrain/QuantCellPaper/base_models_tuned'; full=[joblib.load(f'{T}/LightGBM_y_test.joblib'),joblib.load(f'{T}/LightGBM_y_pred.joblib')]; f1full=np.mean([f1_score(full[0][i],full[1][i],average=None) for i in range(10)],axis=0)
for c,a,b in zip(enc.classes_,f1,f1full): log(f"  {c:<28} nested baseline F1 {a:.4f}   full-population baseline {b:.4f}   diff {a-b:+.4f}")
log(f"macro F1 nested {f1.mean():.4f} vs full {f1full.mean():.4f}"); log("DONE")
