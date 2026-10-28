""" Ten-model comparison on the external datasets ): every classifier with its BM-tuned
hyperparameters (retrain/QuantCellPaper/model_selection/tuning_all10), scored by 10-fold stratified CV (quantcell.cv_fit_pred,
seed 42) on the SAME training split QuantCell uses in Fig 4 (process_data: 80 % of labelled cells, stratified, seed 42), macro AP
exactly as the BM tuned bars (r3_tuned_cv.py). Held-out test cells are not touched. Usage: external_tuned_cv.py colon|chl1|chl2|codex [njobs]
Resumable: per-model checkpoint {OUT}/{ds}/{model}.json. Models run cheapest first."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, re, time, json, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
R=f'{QCP.REPO}'; sys.path.insert(0,f'{QCP.CODE}'); CODE=f'{QCP.CODE}'; sys.path.insert(0,CODE); os.chdir(CODE)
from quantcell import quantcell_project
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
ds=sys.argv[1]; NJ=int(sys.argv[2]) if len(sys.argv)>2 else 12
OUT=f'{QCP.OUT}/external_tuned_cv/{ds}'; os.makedirs(OUT, exist_ok=True)
TUN=f'{QCP.OUT}/retrain/QuantCellPaper/model_selection/tuning_all10'
T0=time.time()
def log(m): print(f"[{time.time()-T0:8.1f}s][{ds}] {m}", flush=True)
# ---- data: identical loaders to the Fig 4 QuantCell runs (colon_b1.py; fig4_v2_data.py cHL block)
if ds=='colon':
    SRC=f'{QCP.EXT}/colon/CRC_clusters_neighborhoods_markers.csv'; hdr=list(pd.read_csv(SRC,nrows=0).columns)
    mk=[c for c in hdr if ':Cyc_' in c and not any(x in c.upper() for x in ('HOECHST','DRAQ5'))]
    d=pd.read_csv(SRC, usecols=mk+['ClusterName','X:X','Y:Y'], low_memory=False)
    ren={c: f"{c.split(':')[0].split(' - ')[0].strip()}: Cell: Mean" for c in mk}; d=d.rename(columns=ren); feat=list(ren.values())
    d['cell_type']=d['ClusterName'].where(~d['ClusterName'].isin({'dirt','undefined'}),'Other'); d['x']=d['X:X']; d['y']=d['Y:Y']
    codex=d[['cell_type','x','y']+feat].copy()
else:
    RAW=f'{QCP.EXT}'
    CFG={'chl2':dict(files=[f'{RAW}/cHL_2_MIBI/annotation_csv/{i}.csv' for i in range(1,7)], label='Annotation', drop={'cellLabel','X_cent','Y_cent','cellSize','Annotation','Annotation_num'}, unlab={'Other'}),
         'chl1':dict(files=[f'{RAW}/cHL_1_MIBI/annotation_csv/cHL1_MIBI.csv'], label='Annotation', drop={'cellLabel','Annotation','centroidX','centroidY','cellSize','identifier'}, unlab={'Other'}),
         'codex':dict(files=[f'{RAW}/cHL_CODEX/annotation_csv/cHL_CODEX_annotation.csv'], label='cellType', drop={'cellLabel','X_cent','Y_cent','cellSize','cellType','DAPI.01'}, unlab={'Other','Seg Artifact'})}[ds]
    full=pd.concat([pd.read_csv(f, low_memory=False) for f in CFG['files']], ignore_index=True); full.columns=[c.strip('"') for c in full.columns]
    y_all=full[CFG['label']].astype(str).where(~full[CFG['label']].astype(str).isin(CFG['unlab']),'Other').values.astype(object)
    markers=[c for c in full.columns if c not in CFG['drop']]; safe=lambda m: re.sub(r'[^A-Za-z0-9_]','_',m)
    codex=full[markers].copy(); codex.columns=[f'{safe(m)}: Cell: Mean' for m in markers]; codex.insert(0,'cell_type',y_all)
p=quantcell_project(); p.codex=codex; p.process_data()
log(f"X_train {p.X_train.shape}  X_test {len(p.X_test)}  classes {len(p.target_encoder.classes_)}  njobs {NJ}")
MODELS=[('Gaussian Naive-Bayes','GaussianNB', lambda q: GaussianNB(**q)),
        ('Ridge Classifier','RidgeClassifier', lambda q: RidgeClassifier(random_state=42, **q)),
        ('Logistic Regression','LogisticRegression', lambda q: LogisticRegression(n_jobs=NJ, random_state=42, **q)),
        ('Extra Trees','ExtraTreesClassifier', lambda q: ExtraTreesClassifier(n_jobs=NJ, random_state=42, **q)),
        ('LightGBM','LGBMClassifier', lambda q: LGBMClassifier(n_jobs=NJ, random_state=42, verbose=-1, **q)),
        ('Decision Tree','DecisionTreeClassifier', lambda q: DecisionTreeClassifier(random_state=42, **q)),
        ('Random Forest','RandomForestClassifier', lambda q: RandomForestClassifier(n_jobs=NJ, random_state=42, **q)),
        ('Linear SVC','LinearSVC', lambda q: OneVsRestClassifier(LinearSVC(random_state=42, dual='auto', **{k.replace('estimator__',''):v for k,v in q.items()}), n_jobs=NJ)),
        ('Multilayer Perceptron','MLPClassifier', lambda q: MLPClassifier(random_state=42, **q)),
        ('Nearest Neighbors','KNeighborsClassifier', lambda q: KNeighborsClassifier(n_jobs=NJ, **q))]
only=os.environ.get('ONLY'); skip=set(filter(None,os.environ.get('SKIP','').split(',')))
for name,label,make in MODELS:
    if only and name!=only: continue
    if name in skip: log(f"  {name}: skipped (SKIP)"); continue
    ck=f'{OUT}/{label}.json'
    if os.path.exists(ck): log(f"  {name}: cached {json.load(open(ck))['macro_AP_mean']:.4f}"); continue
    bp=joblib.load(f'{TUN}/{label}_best_params_macro.joblib'); p.set_clf(make(bp)); t=time.time()
    yt,yp,pr=p.cv_fit_pred(name=f'{ds}:{name}', n_splits=10); el=time.time()-t
    aps=[float(average_precision_score(yt[i],pr[i],average='macro')) for i in range(10)]
    rec={'model':name,'macro_AP_mean':float(np.mean(aps)),'macro_AP_std':float(np.std(aps)),'folds':aps,'seconds':round(el,1),
         'n_train':int(p.X_train.shape[0]),'n_features':int(p.X_train.shape[1]),'params':{k:(v if isinstance(v,(int,float,str,bool,type(None))) else str(v)) for k,v in bp.items()}}
    json.dump(rec, open(ck,'w'), indent=1); log(f"  {name:<22} macro AP {rec['macro_AP_mean']:.4f} +/- {rec['macro_AP_std']:.4f}  {el:.0f}s")
res={json.load(open(f'{OUT}/{f}'))['model']:json.load(open(f'{OUT}/{f}'))['macro_AP_mean'] for f in os.listdir(OUT) if f.endswith('.json') and f!='summary.json'}
json.dump(dict(sorted(res.items(), key=lambda kv:-kv[1])), open(f'{OUT}/summary.json','w'), indent=1); log(f"summary {json.dumps(dict(sorted(res.items(), key=lambda kv:-kv[1])))}"); log("DONE")
