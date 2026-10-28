""" Cell-centric spatial features added to QuantCell's inputs, bone marrow and colon.
Neighbourhoods: k nearest cells within the same region (BM section / colon Region), k in {10, 20}, self excluded.
Families: F1 neighbour-averaged marker means (one column per marker); F2 composition of rule-based conventional
labels among neighbours (BM only -- colon has no marker-combination labels; noted); F3 density = mean distance to
the k neighbours + number of cells within 30 um. Same held-out cells as the retrain / B1 (outer 80/20 stratified
seed 42 of labelled cells), inner 80/20 of the training side for FDR thresholds, published-tuned RF, standardised
features (scaler fit on the inner training set), exactly as process_data() does. Variants: base, +F1, +F2, +F3, +all."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, time, json, argparse, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _common import R, CODE, PROJ, NAME, MS, SUF, make_clf
ap=argparse.ArgumentParser(); ap.add_argument('dataset', choices=['bm','colon','chl1']); ap.add_argument('--njobs',type=int,default=12); a=ap.parse_args()
sys.path.insert(0,CODE); os.chdir(CODE)
from quantcell import find_thresholds, apply_thresholds
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score, balanced_accuracy_score
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s][spatial {a.dataset}] {m}", flush=True)
OUT=f'{QCP.OUT}/spatial{SUF}/{a.dataset}'; os.makedirs(OUT, exist_ok=True)
V=f'{QCP.OUT}'
if a.dataset=='bm':
    d=pd.read_csv(f'{PROJ}/codex_rule_based_{NAME}.csv', low_memory=False)
    import re; base=[c for c in d.columns if re.search(r"\w+: (Nucleus|Cytoplasm|Membrane|Cell): (Mean|Median|Min|Max|Std\.Dev\.)", c)]
    markers=sorted({c.split(':')[0] for c in base}); mcols=[f'{m}: Cell: Mean' for m in markers]; region='section'; xy=['x','y']; um_per_unit=1.0; has_rules=True
elif a.dataset=='chl1':
    # Hodgkin lymphoma cHL1 (MIBI), same loader as fig4_v2_data.py: 41 marker means, 32 stitched regions ('identifier'),
    # centroids in pixels; pixel size 0.78125 um (400 um / 512 px MIBI fields, every region a multiple of 512 px -- ).
    import re
    full=pd.read_csv(f'{QCP.EXT}/cHL_1_MIBI/annotation_csv/cHL1_MIBI.csv', low_memory=False); full.columns=[c.strip('"') for c in full.columns]
    drop={'cellLabel','Annotation','centroidX','centroidY','cellSize','identifier'}; mk=[c for c in full.columns if c not in drop]; safe=lambda m: re.sub(r'[^A-Za-z0-9_]','_',m)
    ren={c: f"{safe(c)}: Cell: Mean" for c in mk}; d=full.rename(columns=ren); base=[ren[c] for c in mk]; mcols=base
    d['cell_type']=d['Annotation'].astype(str).where(~d['Annotation'].astype(str).isin({'Other'}),'Other'); region='identifier'; xy=['centroidX','centroidY']; um_per_unit=0.78125; has_rules=False
    HELD=pd.read_csv(f'{V}/chl/chl1_maps/heldout_rows.csv').iloc[:,0].values
else:
    src=f'{V}/colon/CRC_clusters_neighborhoods_markers.csv'; hdr=list(pd.read_csv(src,nrows=0).columns)
    mk=[c for c in hdr if ':Cyc_' in c and not any(x in c.upper() for x in ('HOECHST','DRAQ5'))]
    d=pd.read_csv(src, usecols=mk+['ClusterName','Region','X:X','Y:Y'], low_memory=False)
    ren={c: f"{c.split(':')[0].split(' - ')[0].strip()}: Cell: Mean" for c in mk}; d=d.rename(columns=ren); base=sorted(ren.values()); mcols=base
    d['cell_type']=d['ClusterName'].where(~d['ClusterName'].isin({'dirt','undefined'}),'Other'); region='Region'; xy=['X:X','Y:Y']; um_per_unit=0.377; has_rules=False
d=d.reset_index(drop=True); n=len(d); log(f"{n:,} cells, {len(base)} base features, {len(mcols)} markers, regions {d[region].nunique()}, rule-based labels {has_rules}")
types=sorted(d['cell_type'].unique())
# ---- neighbour features
M=d[mcols].to_numpy(float); XY=d[xy].to_numpy(float)*um_per_unit; onehot=(d['cell_type'].to_numpy()[:,None]==np.array(types)[None,:]).astype(float)
feats={}
for k in (10,20):
    F1=np.zeros((n,len(mcols))); F2=np.zeros((n,len(types))); F3=np.zeros((n,2))
    for reg,idx in d.groupby(region).indices.items():
        if len(idx)<=k+1: F1[idx]=M[idx].mean(0); F2[idx]=onehot[idx].mean(0); F3[idx]=[np.nan, len(idx)]; continue
        tree=cKDTree(XY[idx]); dist,nb=tree.query(XY[idx], k=k+1); dist,nb=dist[:,1:],nb[:,1:]; g=idx[nb]
        F1[idx]=M[g].mean(1); F2[idx]=onehot[g].mean(1); F3[idx,0]=dist.mean(1); F3[idx,1]=[len(x)-1 for x in tree.query_ball_point(XY[idx], r=30.0)]
    F3=np.nan_to_num(F3, nan=np.nanmax(F3[:,0]))
    feats[k]={'F1':(F1,[f'nbr{k}: {m}: Cell: Mean' for m in markers] if a.dataset=='bm' else [f'nbr{k}: {c}' for c in mcols]),'F2':(F2,[f'nbr{k}: frac {t}' for t in types]),'F3':(F3,[f'nbr{k}: mean dist um', f'n within 30um'])}
    log(f"k={k}: neighbour features built; mean kNN distance {np.median(F3[:,0]):.1f} um (median), cells within 30 um median {np.median(F3[:,1]):.0f}")
# ---- split (identical to retrain / B1): outer 80/20 of labelled cells, then inner 80/20 for thresholds
lab_idx=np.where(d['cell_type'].values!='Other')[0]; enc=LabelEncoder().fit(d.loc[lab_idx,'cell_type']); y_all=enc.transform(d.loc[lab_idx,'cell_type'])
tr,te=train_test_split(np.arange(len(lab_idx)), train_size=0.8, stratify=y_all, random_state=42); itr,ite=train_test_split(tr, train_size=0.8, stratify=y_all[tr], random_state=42)
rows_itr,rows_ite,rows_te=lab_idx[itr],lab_idx[ite],lab_idx[te]; y_itr,y_ite,y_te=y_all[itr],y_all[ite],y_all[te]; n_cls=len(enc.classes_)
if a.dataset=='chl1': assert set(rows_te)==set(HELD) and len(rows_te)==len(HELD), 'held-out split differs from Fig 4 (chl1_maps/heldout_rows.csv)'; log('held-out rows identical to the Fig 4 / MAPS split')
log(f"inner train {len(rows_itr):,}  inner test {len(rows_ite):,}  held-out {len(rows_te):,}  classes {n_cls}")
Xbase=d[base].to_numpy(float); Yb=label_binarize(y_te, classes=np.arange(n_cls)); freq=pd.Series(y_te).value_counts(normalize=True)
rare=[c for c in range(n_cls) if freq.get(c,0)<0.1]; abund=[c for c in range(n_cls) if freq.get(c,0)>=0.1]
fams=['F1','F2','F3'] if has_rules else ['F1','F3']
VARIANTS=[('base',None,[])]+[(f'{f}_k{k}',k,[f]) for k in (10,20) for f in fams]+[(f'all_k{k}',k,fams) for k in (10,20)]
res_path=f'{OUT}/spatial_results.json'; res=json.load(open(res_path)) if os.path.exists(res_path) else {}
for name,k,fl in VARIANTS:
    if name in res: log(f"  {name}: cached"); continue
    X=Xbase if not fl else np.hstack([Xbase]+[feats[k][f][0] for f in fl]); cols=list(base)+sum([feats[k][f][1] for f in fl],[]) if fl else list(base)
    sc=StandardScaler().fit(X[rows_itr]); Xtr_,Xin_,Xho_=sc.transform(X[rows_itr]),sc.transform(X[rows_ite]),sc.transform(X[rows_te])
    t=time.time(); clf=make_clf(a.njobs).fit(Xtr_, y_itr); fit_s=time.time()-t
    pred=clf.predict(Xho_); proba=clf.predict_proba(Xho_); per=precision_score(y_te,pred,average=None,labels=list(range(n_cls)),zero_division=0)
    r={'n_features':int(X.shape[1]),'fit_s':round(fit_s),'accuracy':float(accuracy_score(y_te,pred)),'balanced_accuracy':float(balanced_accuracy_score(y_te,pred)),
       'precision_macro':float(precision_score(y_te,pred,average='macro',zero_division=0)),'recall_macro':float(recall_score(y_te,pred,average='macro',zero_division=0)),
       'f1_macro':float(f1_score(y_te,pred,average='macro',zero_division=0)),'average_precision_macro':float(average_precision_score(Yb,proba,average='macro')),
       'precision_macro_rare':float(np.mean(per[rare])) if rare else None,'precision_macro_abundant':float(np.mean(per[abund])) if abund else None,
       'per_class_precision':{enc.classes_[i]:float(per[i]) for i in range(n_cls)}}
    thr=np.array(find_thresholds(clf,Xin_,y_ite,FDR_cutoff=0.05)); lab=apply_thresholds(clf,Xho_,thr); keep=lab!=-1
    r['fdr_5']={'coverage':float(keep.mean()),'precision_macro':float(precision_score(y_te[keep],lab[keep],average='macro',zero_division=0)),'realised_FDR':float(1-accuracy_score(y_te[keep],lab[keep]))}
    imp=pd.Series(clf.feature_importances_, index=cols); r['importance_share_spatial']=float(imp[[c for c in cols if c.startswith('nbr') or c.startswith('n within')]].sum()) if fl else 0.0
    np.savez(f'{OUT}/{name}.npz', y=y_te, pred=pred, lab5=lab, classes=np.array(list(enc.classes_),dtype=object))
    res[name]=r; json.dump(res, open(res_path,'w'), indent=2)
    log(f"  {name:<10} feats {X.shape[1]:>4}  fit {fit_s:4.0f}s  acc {r['accuracy']:.4f}  macroP {r['precision_macro']:.4f}  F1 {r['f1_macro']:.4f}  AP {r['average_precision_macro']:.4f}  rareP {r['precision_macro_rare']:.4f} | @5%: cov {r['fdr_5']['coverage']:.4f} P {r['fdr_5']['precision_macro']:.4f} FDR {r['fdr_5']['realised_FDR']:.4f} | spatial importance share {r['importance_share_spatial']:.3f}")
log("DONE")
