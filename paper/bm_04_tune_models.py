"""R3 -- : hyperparameter-tune all ten base classifiers on the BM codex.
Protocol = paper/03 cell 9 exactly: HalvingGridSearchCV (factor 3, min_resources 'exhaust' -> starts at
n_classes*n_splits*2 = 420 for the big grids, last iteration 34,020 samples), StratifiedKFold(10, shuffle,
seed 42), scorer = macro average precision. Plain process_data() population (80,615 train cells).
Trees grid = the grid the PUBLISHED cv_results were actually run with (n_estimators 50/100/200/400 --
the notebook text says 100/300/600 and is stale). MLP grid = published. The other seven grids are new
(the paper tuned only the top three). Estimators single-threaded; parallelism is across fits.
Outputs: model_selection/tuning_all10/{label}_cv_results_macro.csv + _best_params_macro.joblib + summary.
"""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, time, json, argparse, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, CODE, RB, NAME, PROJ, MS
sys.path.insert(0, f'{QCP.CODE}')
ap=argparse.ArgumentParser(); ap.add_argument('--models', nargs='+', required=True); ap.add_argument('--njobs', type=int, default=36)
ap.add_argument('--lgbm-grid', default=None, help='JSON file with the LightGBM grid (from the M14 diagnosis)'); a=ap.parse_args()
sys.path.insert(0, CODE); os.chdir(CODE)
from quantcell import quantcell_project
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
T0=time.time()
def log(m): print(f"[{time.time()-T0:8.1f}s][tune] {m}", flush=True)
OUT=f'{MS}/tuning_all10'; os.makedirs(OUT, exist_ok=True)
p=quantcell_project(); p.initialize(base_path=RB, project_name=NAME); p.process_data()
log(f"X_train {p.X_train.shape}  classes {len(p.target_encoder.classes_)}")
def macro_ap_scorer(estimator, X, y):
    """Macro average precision that survives tiny halving subsamples.
    The published scorer (make_scorer(average_precision_score, average='macro', response_method='predict_proba'))
    raises IndexError whenever a test fold lacks a class the model knows (label_binarize on unique(y_true) gives fewer
    columns than predict_proba), so every score at the 420/1260/3780/11340-sample iterations was NaN and halving
    promoted candidates without evaluating them. Here y is binarised against estimator.classes_ and the mean
    is taken over the classes present in y (and known to the model). Identical to sklearn's macro AP when all
    classes are present."""
    if hasattr(estimator, 'predict_proba'):
        S=np.asarray(estimator.predict_proba(X))
    else:
        S=np.asarray(estimator.decision_function(X))
    cls=np.asarray(estimator.classes_)
    if S.ndim==1: S=np.column_stack([-S,S])
    keep=[i for i,c in enumerate(cls) if np.any(y==c)]
    if not keep: return np.nan
    return float(np.mean([average_precision_score((y==cls[i]).astype(int), S[:,i]) for i in keep]))
scorer=macro_ap_scorer
cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=p.random_seed)

TREES={'n_estimators':[50,100,200,400],'criterion':['gini','entropy'],'min_samples_split':[2,4,6,8],
       'min_samples_leaf':[1,2,4,6,8],'max_features':['sqrt','log2'],'max_depth':[10,20,40,60,80,None],
       'bootstrap':[False,True],'class_weight':[None,'balanced','balanced_subsample']}
MLP={'hidden_layer_sizes':[(50,),(100,),(150,),(200,),(300,),(400,),(500,),(100,50),(200,50),(50,50,50),(50,100,50)],
     'activation':['identity','logistic','tanh','relu'],'solver':['adam','lbfgs','sgd'],
     'alpha':[0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1],'learning_rate':['constant','invscaling','adaptive']}
LGBM=json.load(open(a.lgbm_grid)) if a.lgbm_grid else {'learning_rate':[0.02,0.05,0.1],'num_leaves':[15,31,63],
     'min_child_samples':[20,50,100],'min_child_weight':[1e-3,1.0],'n_estimators':[100,300],'reg_lambda':[0.0,1.0]}
MODELS={
 'RandomForestClassifier': (RandomForestClassifier(random_state=42), TREES),
 'ExtraTreesClassifier':   (ExtraTreesClassifier(random_state=42), TREES),
 'MLPClassifier':          (MLPClassifier(random_state=42), MLP),
 'KNeighborsClassifier':   (KNeighborsClassifier(), {'n_neighbors':[3,5,7,11,15,21,31],'weights':['uniform','distance'],'p':[1,2]}),
 'DecisionTreeClassifier': (DecisionTreeClassifier(random_state=42), {'criterion':['gini','entropy'],'max_depth':[10,20,40,60,80,None],
                            'min_samples_split':[2,4,6,8],'min_samples_leaf':[1,2,4,6,8],'max_features':['sqrt','log2',None],'class_weight':[None,'balanced']}),
 'GaussianNB':             (GaussianNB(), {'var_smoothing':list(np.logspace(-12,0,13))}),
 'RidgeClassifier':        (RidgeClassifier(random_state=42), {'alpha':[1e-3,1e-2,1e-1,1,10,100,1000],'fit_intercept':[True,False],'class_weight':[None,'balanced']}),
 'LinearSVC':              (OneVsRestClassifier(LinearSVC(random_state=42, dual='auto')), {'estimator__C':[0.001,0.01,0.1,1,10],
                            'estimator__loss':['hinge','squared_hinge'],'estimator__class_weight':[None,'balanced']}),
 'LogisticRegression':     (LogisticRegression(random_state=42), {'C':[0.001,0.01,0.1,1,10,100],'class_weight':[None,'balanced'],'max_iter':[100,1000]}),
 'LGBMClassifier':         (LGBMClassifier(random_state=42, verbose=-1, n_jobs=1), LGBM),
}
PUBMS=f'{QCP.OUT}/published_model_selection_unused'
summ_path=f'{OUT}/tuning_summary.json'; summ=json.load(open(summ_path)) if os.path.exists(summ_path) else {}
for label in a.models:
    est,grid=MODELS[label]
    if os.path.exists(f'{OUT}/{label}_best_params_macro.joblib'): log(f"{label}: already tuned, skipping"); continue
    ncand=int(np.prod([len(v) for v in grid.values()])); log(f"=== {label}: {ncand} candidates, grid keys {list(grid)}")
    t=time.time()
    search=HalvingGridSearchCV(est, grid, n_jobs=a.njobs, cv=cv, verbose=1, random_state=p.random_seed, scoring=scorer)
    search.fit(p.X_train, p.y_train)
    res=pd.DataFrame(search.cv_results_); res.to_csv(f'{OUT}/{label}_cv_results_macro.csv', index=False)
    joblib.dump(search.best_params_, f'{OUT}/{label}_best_params_macro.joblib')
    last=res[res['iter']==res['iter'].max()]
    entry={'best_params':{k:(v if isinstance(v,(int,float,str,bool,type(None))) else str(v)) for k,v in search.best_params_.items()},
           'best_score':float(search.best_score_),'n_candidates':ncand,'n_fits':int(len(res)*10),'iterations':int(res['iter'].max()+1),
           'n_resources':[int(x) for x in sorted(res['n_resources'].unique())],'last_iter_candidates':int(len(last)),
           'wall_hours':round((time.time()-t)/3600,2),'fit_cpu_hours':round(float(res['mean_fit_time'].sum()*10/3600),1)}
    pub=f'{PUBMS}/{label}_best_params_macro.joblib'
    if os.path.exists(pub):
        pb=joblib.load(pub); entry['published_best']={k:(v if isinstance(v,(int,float,str,bool,type(None))) else str(v)) for k,v in pb.items()}
        entry['same_as_published']=bool(pb==search.best_params_)
        pubrow=res[res['params'].astype(str)==str(pb)]
        entry['published_params_score_now']=float(pubrow['mean_test_score'].max()) if len(pubrow) else None
    summ[label]=entry; json.dump(summ, open(summ_path,'w'), indent=2)
    log(f"{label}: best {search.best_score_:.4f}  {search.best_params_}  wall {entry['wall_hours']} h  "
        f"{'SAME as published' if entry.get('same_as_published') else ('DIFFERS from published '+str(entry.get('published_best')) if 'published_best' in entry else '')}")
log("DONE")
