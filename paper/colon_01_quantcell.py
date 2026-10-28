"""colon_b1.py re-run that also SAVES per-cell held-out predictions with original row ids (needed for the five-method
colon table). Same code path (process_data split seed 42, BM-tuned classifier via QC_CLF, thresholds at 5% FDR on the
held-out set as in B1); """
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import sys, os, warnings, time, json, joblib, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
KEEP16={'tumor cells','CD68+CD163+ macrophages','smooth muscle','granulocytes','CD8+ T cells','CD4+ T cells CD45RO+','B cells','vasculature','plasma cells','immune cells','Tregs','CD4+ T cells','CD11c+ DCs','lymphatics','NK cells','CD3+ T cells'}   # the 16 published classes with a CELESTA signature type (smooth muscle <- aSMA+ stroma)
CODE=f'{QCP.CODE}'; sys.path.insert(0,CODE); os.chdir(CODE)
from quantcell import quantcell_project, find_thresholds, apply_thresholds
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
SRC=f'{QCP.EXT}/colon/CRC_clusters_neighborhoods_markers.csv'
_S=os.environ.get('QC_CLF','rf'); _S='' if _S=='rf' else '_'+_S; C16=os.environ.get('QC_COLON16')=='1'; OUT=f'{QCP.OUT}/workspace_out/colon'+('16' if C16 else '')+_S; os.makedirs(OUT, exist_ok=True)
hdr=list(pd.read_csv(SRC,nrows=0).columns); mk=[c for c in hdr if ':Cyc_' in c and not any(x in c.upper() for x in ('HOECHST','DRAQ5'))]
d=pd.read_csv(SRC, usecols=mk+['ClusterName','X:X','Y:Y','CellID','Region'], low_memory=False)
ren={c: f"{c.split(':')[0].split(' - ')[0].strip()}: Cell: Mean" for c in mk}; d=d.rename(columns=ren); feat=list(ren.values())
d['cell_type']=d['ClusterName'].where(~d['ClusterName'].isin({'dirt','undefined'}),'Other'); d['x']=d['X:X']; d['y']=d['Y:Y']
if C16: d['cell_type']=d['cell_type'].where(d['cell_type'].isin(KEEP16),'Other')
p=quantcell_project(); p.codex=d[['cell_type','x','y']+feat].copy(); p.process_data()
sys.path.insert(0,f'{QCP.PAPER}'); from _common import make_clf
p.set_clf(make_clf(12)); t=time.time(); p.fit(); print(f'fit {time.time()-t:.0f}s', flush=True)
enc=p.target_encoder; classes=list(enc.classes_); y_true=p.y_test; y_pred=p.clf.predict(p.X_test)
mP=precision_score(y_true,y_pred,average='macro',zero_division=0); print(f'unthresholded macro precision {mP:.6f}')
# held-out row ids: same split as process_data (labelled rows in table order, train_size 0.8, stratify, seed 42)
lab_rows=np.where(d.cell_type.values!='Other')[0]; y_all=enc.transform(d.cell_type.values[lab_rows])
_,pos,_,yte=train_test_split(np.arange(len(lab_rows)), y_all, train_size=0.8, stratify=y_all, random_state=42); assert (yte==y_true).all()
rows=lab_rows[pos]
thr=np.array(find_thresholds(p.clf,p.X_test,p.y_test,FDR_cutoff=0.05)); lab5=apply_thresholds(p.clf,p.X_test,thr)
out=pd.DataFrame({'row':rows,'CellID':d.CellID.values[rows],'Region':d.Region.values[rows],'true':enc.inverse_transform(y_true),'pred':enc.inverse_transform(y_pred),
                  'pred_5pct':np.where(lab5==-1,'Other',enc.inverse_transform(np.where(lab5==-1,0,lab5)))})
out.to_csv(f'{OUT}/heldout_predictions.csv', index=False); print('wrote', f'{OUT}/heldout_predictions.csv', out.shape, 'coverage@5%', float((out.pred_5pct!='Other').mean()))
