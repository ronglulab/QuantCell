"""R2 -- base-model comparison on the codex (paper/04 cells 6-8; Fig 2a/2b). Ten classifiers
as published (incl. LightGBM, loaded from the side directory .envs/extra_lightgbm -- repro311 untouched),
SEEDED where the estimator accepts a seed, otherwise identical defaults to the notebook.
Writes the per-model y_test / y_pred / y_proba / time_elapsed joblibs and base_clf_performance_dict
exactly in the published layout so the Fig 2 driver runs unchanged."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, time, json, argparse, warnings, numpy as np, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, CODE, RB, NAME, PROJ, MS
sys.path.insert(0, f'{QCP.CODE}')
ap=argparse.ArgumentParser(); ap.add_argument('--njobs',type=int,default=12); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
sys.path.insert(0,CODE); os.chdir(CODE)
from quantcell import quantcell_project, interpolate_PRC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score
from lightgbm import LGBMClassifier
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s][base] {m}", flush=True)
NJ=a.njobs
names=['Nearest Neighbors','Multilayer Perceptron','Random Forest','Extra Trees','Decision Tree',
       'Gaussian Naive-Bayes','Ridge Classifier','Linear SVC','LightGBM','Logistic Regression']
classifiers=[KNeighborsClassifier(n_jobs=NJ), MLPClassifier(random_state=42), RandomForestClassifier(n_jobs=NJ, random_state=42),
             ExtraTreesClassifier(n_jobs=NJ, random_state=42), DecisionTreeClassifier(random_state=42), GaussianNB(),
             RidgeClassifier(random_state=42), OneVsRestClassifier(LinearSVC(random_state=42), n_jobs=NJ),
             LGBMClassifier(n_jobs=NJ, random_state=42, verbose=-1), LogisticRegression(n_jobs=NJ, random_state=42)]
p=quantcell_project(); p.initialize(base_path=RB, project_name=NAME); p.process_data()
log(f"X_train {p.X_train.shape}  classes {len(p.target_encoder.classes_)}")
if a.smoke: log("SMOKE OK"); sys.exit(0)
BM=f'{PROJ}/base_models'; os.makedirs(BM, exist_ok=True); summ={}
for name,clf in zip(names,classifiers):
    if os.path.exists(f'{BM}/{name}_y_proba.joblib'): log(f"  {name} cached"); continue
    p.set_clf(clf); t=time.time()
    try:
        yt,yp,pr=p.cv_fit_pred(name=name, n_splits=10); el=time.time()-t
    except Exception as e:
        log(f"  {name:<22} FAILED {type(e).__name__}: {str(e)[:120]}"); summ[name]={'error':f'{type(e).__name__}: {str(e)[:200]}'}; continue
    joblib.dump(yt,f'{BM}/{name}_y_test.joblib'); joblib.dump(yp,f'{BM}/{name}_y_pred.joblib')
    joblib.dump(pr,f'{BM}/{name}_y_proba.joblib'); joblib.dump(el,f'{BM}/{name}_time_elapsed.joblib')
    aps=[float(average_precision_score(yt[i],pr[i],average='macro')) for i in range(10)]
    log(f"  {name:<22} macroAP {np.mean(aps):.4f} +/- {np.std(aps):.4f}  ({el:.0f}s)  folds {[round(x,3) for x in aps]}")
# ---- cell 8: base_clf_performance_dict, verbatim recipe
d={}
for _name in names:
    if not os.path.exists(f'{BM}/{_name}_y_proba.joblib'): log(f"  {_name}: missing, left out of base_clf_performance_dict"); continue
    y_test=joblib.load(f'{BM}/{_name}_y_test.joblib'); y_pred=joblib.load(f'{BM}/{_name}_y_pred.joblib'); y_proba=joblib.load(f'{BM}/{_name}_y_proba.joblib')
    mAP,wAP,mPRC,wPRC=[],[],[],[]
    for fold in range(10):
        mAP.append(average_precision_score(y_test[fold], y_proba[fold], average='macro'))
        wAP.append(average_precision_score(y_test[fold], y_proba[fold], average='weighted'))
        mPRC.append(interpolate_PRC(y_test[fold], y_pred[fold], y_proba[fold], average='macro'))
        wPRC.append(interpolate_PRC(y_test[fold], y_pred[fold], y_proba[fold], average='weighted'))
    num_classes=len(np.unique(np.concatenate([y_test[x] for x in range(10)])))
    d[_name]=[mAP,wAP,mPRC,wPRC,num_classes]
    summ[_name]={'macro_AP_mean':float(np.mean(mAP)),'macro_AP_std':float(np.std(mAP)),'folds':[float(x) for x in mAP],
                 'seconds':float(joblib.load(f'{BM}/{_name}_time_elapsed.joblib'))}
joblib.dump(d, f'{BM}/base_clf_performance_dict.joblib'); json.dump(summ, open(f'{BM}/base_models_summary.json','w'), indent=2)
log("ranking:"); [log(f"    {k:<22} {v['macro_AP_mean']:.4f}") for k,v in sorted(summ.items(), key=lambda x:-x[1].get('macro_AP_mean',-1)) if 'macro_AP_mean' in v]
log("DONE")
