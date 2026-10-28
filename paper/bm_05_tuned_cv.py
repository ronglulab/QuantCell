"""R3b -- , second half: evaluate every model's NEW best setting with the same 10-fold CV on the
full training set that paper/04 cell 12 uses for the Fig 2c 'tuned' bars (and that r2_base_models.py used for
the default bars). Artifacts in the base_models layout under base_models_tuned/, so a tuned Fig 2b/2c can be
built the same way. Models whose search has not finished (LightGBM) are skipped and picked up on rerun."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, time, json, argparse, warnings, numpy as np, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, CODE, RB, NAME, PROJ, MS
sys.path.insert(0, f'{QCP.CODE}')
ap=argparse.ArgumentParser(); ap.add_argument('--njobs',type=int,default=12); a=ap.parse_args()
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
def log(m): print(f"[{time.time()-T0:7.1f}s][tunedcv] {m}", flush=True)
NJ=a.njobs; TUN=f'{MS}/tuning_all10'; BM=f'{PROJ}/base_models_tuned'; os.makedirs(BM, exist_ok=True)
# display name -> (search label, estimator factory)
MODELS=[('Nearest Neighbors','KNeighborsClassifier', lambda p: KNeighborsClassifier(n_jobs=NJ, **p)),
        ('Multilayer Perceptron','MLPClassifier', lambda p: MLPClassifier(random_state=42, **p)),
        ('Random Forest','RandomForestClassifier', lambda p: RandomForestClassifier(n_jobs=NJ, random_state=42, **p)),
        ('Extra Trees','ExtraTreesClassifier', lambda p: ExtraTreesClassifier(n_jobs=NJ, random_state=42, **p)),
        ('Decision Tree','DecisionTreeClassifier', lambda p: DecisionTreeClassifier(random_state=42, **p)),
        ('Gaussian Naive-Bayes','GaussianNB', lambda p: GaussianNB(**p)),
        ('Ridge Classifier','RidgeClassifier', lambda p: RidgeClassifier(random_state=42, **p)),
        ('Linear SVC','LinearSVC', lambda p: OneVsRestClassifier(LinearSVC(random_state=42, dual='auto', **{k.replace('estimator__',''):v for k,v in p.items()}), n_jobs=NJ)),
        ('LightGBM','LGBMClassifier', lambda p: LGBMClassifier(n_jobs=NJ, random_state=42, verbose=-1, **p)),
        ('Logistic Regression','LogisticRegression', lambda p: LogisticRegression(n_jobs=NJ, random_state=42, **p))]
p=quantcell_project(); p.initialize(base_path=RB, project_name=NAME); p.process_data()
log(f"X_train {p.X_train.shape}")
summ_path=f'{BM}/base_models_tuned_summary.json'; summ=json.load(open(summ_path)) if os.path.exists(summ_path) else {}
dflt=json.load(open(f'{PROJ}/base_models/base_models_summary.json'))
for name,label,make in MODELS:
    bp_path=f'{TUN}/{label}_best_params_macro.joblib'
    if not os.path.exists(bp_path): log(f"  {name}: search not finished, skipped"); continue
    if os.path.exists(f'{BM}/{name}_y_proba.joblib'): log(f"  {name}: cached"); continue
    bp=joblib.load(bp_path); p.set_clf(make(bp)); t=time.time()
    yt,yp,pr=p.cv_fit_pred(name=name, n_splits=10); el=time.time()-t
    for k,v in (('y_test',yt),('y_pred',yp),('y_proba',pr),('time_elapsed',el)): joblib.dump(v, f'{BM}/{name}_{k}.joblib')
    aps=[float(average_precision_score(yt[i],pr[i],average='macro')) for i in range(10)]
    summ[name]={'best_params':{k:(v if isinstance(v,(int,float,str,bool,type(None))) else str(v)) for k,v in bp.items()},
                'macro_AP_mean':float(np.mean(aps)),'macro_AP_std':float(np.std(aps)),'folds':aps,'seconds':round(el,1),
                'default_macro_AP_mean':dflt.get(name,{}).get('macro_AP_mean')}
    json.dump(summ, open(summ_path,'w'), indent=2)
    log(f"  {name:<22} tuned {np.mean(aps):.4f} +/- {np.std(aps):.4f}  (default {summ[name]['default_macro_AP_mean']})  {el:.0f}s")
log("DONE")
