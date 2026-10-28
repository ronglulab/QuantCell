"""MAPS inputs for cHL2 on the FIXED ground-truth split used by chl2_native.py (labelled = Annotation != Other; 80/20
stratified on the string labels, seed 42). Features = 46 markers + cellSize ('Cell'), as the paper's MAPS runs."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
SRC=f'{QCP.EXT}/cHL_2_MIBI/annotation_csv'; B=f'{QCP.OUT}/chl/chl2_maps'; D=f'{B}/data/cell_phenotyping'; os.makedirs(D, exist_ok=True); os.makedirs(f'{B}/results/cell_phenotyping', exist_ok=True)
d=pd.concat([pd.read_csv(f'{SRC}/{i}.csv').assign(fov=i) for i in range(1,7)], ignore_index=True); markers=list(d.columns[6:52]); gt=d.Annotation.astype(str).values
lab_rows=np.where(gt!='Other')[0]; tr,ho=train_test_split(np.arange(len(lab_rows)), train_size=0.8, stratify=gt[lab_rows], random_state=42)
enc=LabelEncoder().fit(gt[lab_rows]); X=d[markers].copy(); X['Cell']=d['cellSize'].values
Xtr=X.iloc[lab_rows[tr]].reset_index(drop=True); ytr=enc.transform(gt[lab_rows[tr]]); Xte=X.iloc[lab_rows[ho]].reset_index(drop=True); yte=enc.transform(gt[lab_rows[ho]])
Xt,Xv,yt,yv=train_test_split(Xtr,ytr,test_size=0.2,stratify=ytr,random_state=42)
for nm,(A,b) in {'train':(Xt,yt),'valid':(Xv,yv),'test':(Xte,yte)}.items():
    f=A.reset_index(drop=True).copy(); f['cell_label']=b; f.to_csv(f'{D}/{nm}_features.csv', index=False); print(f'  {nm:6s} {f.shape}', flush=True)
json.dump({'classes':list(enc.classes_),'n_features':X.shape[1],'n_classes':len(enc.classes_),'n_test':int(len(yte))}, open(f'{B}/maps_meta.json','w'), indent=2)
pd.Series(lab_rows[ho]).to_csv(f'{B}/heldout_rows.csv', index=False); print('held-out rows saved', len(ho))
