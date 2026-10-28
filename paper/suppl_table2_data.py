#!/usr/bin/env python
"""Supplementary Table 2 on the corrected basis: hyperparameter search space and selected value for all ten base
classifiers (tuning_all10, HalvingGridSearchCV, macro AP scorer that survives rare-class folds ), plus base
(default) and tuned 10-fold CV metrics (macro average precision, balanced accuracy) from base_models/ and
base_models_tuned/ (the Fig 2a inputs). Outputs: suppl_table2_params.csv (long), suppl_table2_metrics.csv, and a
markdown rendering on stdout."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import json, joblib, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score
P=f'{QCP.OUT}/retrain/QuantCellPaper'; OUT=f'{QCP.OUT}/retrain_lgbm/figures'
summ=json.load(open(f'{P}/model_selection/tuning_all10/tuning_summary.json'))
TREES={'n_estimators':[50,100,200,400],'criterion':['gini','entropy'],'min_samples_split':[2,4,6,8],
       'min_samples_leaf':[1,2,4,6,8],'max_features':['sqrt','log2'],'max_depth':[10,20,40,60,80,None],
       'bootstrap':[False,True],'class_weight':[None,'balanced','balanced_subsample']}
MLP={'hidden_layer_sizes':['(50,)','(100,)','(150,)','(200,)','(300,)','(400,)','(500,)','(100, 50)','(200, 50)','(50, 50, 50)','(50, 100, 50)'],
     'activation':['identity','logistic','tanh','relu'],'solver':['adam','lbfgs','sgd'],
     'alpha':[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1],'learning_rate':['constant','invscaling','adaptive']}
GRIDS={'RandomForestClassifier':TREES,'ExtraTreesClassifier':TREES,'MLPClassifier':MLP,
 'KNeighborsClassifier':{'n_neighbors':[3,5,7,11,15,21,31],'weights':['uniform','distance'],'p':[1,2]},
 'DecisionTreeClassifier':{'criterion':['gini','entropy'],'max_depth':[10,20,40,60,80,None],'min_samples_split':[2,4,6,8],
                           'min_samples_leaf':[1,2,4,6,8],'max_features':['sqrt','log2',None],'class_weight':[None,'balanced']},
 'GaussianNB':{'var_smoothing':[f'1e{e}' for e in range(-12,1)]},
 'RidgeClassifier':{'alpha':[1e-3,1e-2,1e-1,1,10,100,1000],'fit_intercept':[True,False],'class_weight':[None,'balanced']},
 'LinearSVC':{'estimator__C':[0.001,0.01,0.1,1,10],'estimator__loss':['hinge','squared_hinge'],'estimator__class_weight':[None,'balanced']},
 'LogisticRegression':{'C':[0.001,0.01,0.1,1,10,100],'class_weight':[None,'balanced'],'max_iter':[100,1000]},
 'LGBMClassifier':json.load(open(f'{P}/m14/lgbm_grid.json'))}
LABEL={'RandomForestClassifier':'Random Forest','ExtraTreesClassifier':'Extra Trees','MLPClassifier':'Multilayer Perceptron',
 'KNeighborsClassifier':'Nearest Neighbors','DecisionTreeClassifier':'Decision Tree','GaussianNB':'Gaussian Naive-Bayes',
 'RidgeClassifier':'Ridge Classifier','LinearSVC':'Linear SVC','LogisticRegression':'Logistic Regression','LGBMClassifier':'LightGBM'}
def fmt(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return 'None'
    if isinstance(v,float) and v!=0 and (abs(v)<1e-3 or abs(v)>=1e4): return f'{v:g}'
    return str(v)
rows=[]
for m,grid in GRIDS.items():
    best=summ[m]['best_params']
    for k,vals in grid.items():
        sel=best[k]; sel='None' if sel is None else str(sel).replace("'",'')
        if k=='var_smoothing': sel=f'{float(sel):g}'
        rows.append({'model':LABEL[m],'parameter':k.replace('estimator__',''),'search_space':', '.join(fmt(v) for v in vals),'selected':sel,
                     'cv_macro_AP_best':round(summ[m]['best_score'],4),'n_candidates':summ[m]['n_candidates']})
params=pd.DataFrame(rows); params.to_csv(f'{OUT}/suppl_table2_params.csv',index=False)
# metrics
met=[]
for m in GRIDS:
    lab=LABEL[m]; r={'model':lab}
    for d,tag in (('base_models','base'),('base_models_tuned','tuned')):
        yt=joblib.load(f'{P}/{d}/{lab}_y_test.joblib'); yp=joblib.load(f'{P}/{d}/{lab}_y_pred.joblib'); pr=joblib.load(f'{P}/{d}/{lab}_y_proba.joblib')
        ap=[average_precision_score(yt[f],pr[f],average='macro') for f in yt]; ba=[balanced_accuracy_score(yt[f],yp[f]) for f in yt]
        r[f'{tag}_macro_AP']=np.mean(ap); r[f'{tag}_macro_AP_sd']=np.std(ap); r[f'{tag}_balanced_accuracy']=np.mean(ba); r[f'{tag}_balanced_accuracy_sd']=np.std(ba)
    met.append(r)
met=pd.DataFrame(met).sort_values('tuned_macro_AP',ascending=False); met.to_csv(f'{OUT}/suppl_table2_metrics.csv',index=False)
print(met.round(3).to_string(index=False)); print()
for m in met['model']:
    sub=params[params.model==m]
    print(f'### {m}  (search candidates {sub.n_candidates.iloc[0]:,}; CV macro AP of selected {sub.cv_macro_AP_best.iloc[0]:.3f})')
    print('| parameter | search space | selected |'); print('|---|---|---|')
    for _,q in sub.iterrows(): print(f'| {q.parameter} | {q.search_space} | **{q.selected}** |')
    print()
