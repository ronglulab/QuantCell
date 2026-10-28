"""Prepare MAPS inputs for colon on EXACTLY the split QuantCell B1 used."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import sys, os, warnings, json, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
KEEP16={'tumor cells','CD68+CD163+ macrophages','smooth muscle','granulocytes','CD8+ T cells','CD4+ T cells CD45RO+','B cells','vasculature','plasma cells','immune cells','Tregs','CD4+ T cells','CD11c+ DCs','lymphatics','NK cells','CD3+ T cells'}   # the 16 published classes with a CELESTA signature type (smooth muscle <- aSMA+ stroma)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
SRC=f'{QCP.EXT}/colon/CRC_clusters_neighborhoods_markers.csv'
C16=os.environ.get('QC_COLON16')=='1'; OUT=f'{QCP.OUT}/workspace_out/colon'+('16' if C16 else '')+'/MAPS/data/cell_phenotyping'
os.makedirs(OUT, exist_ok=True)
UNLAB={'dirt','undefined'}
hdr=list(pd.read_csv(SRC,nrows=0).columns)
mk=[c for c in hdr if ':Cyc_' in c and not any(x in c.upper() for x in ('HOECHST','DRAQ5'))]
d=pd.read_csv(SRC, usecols=mk+['ClusterName','size:size'], low_memory=False)
names=[c.split(':')[0].split(' - ')[0].strip() for c in mk]
d=d.rename(columns=dict(zip(mk,names)))
d['cell_type']=d['ClusterName'].where(~d['ClusterName'].isin(UNLAB),'Other')
if C16: d['cell_type']=d['cell_type'].where(d['cell_type'].isin(KEEP16),'Other')
lab=d[d.cell_type!='Other'].reset_index(drop=True)
enc=LabelEncoder(); y=enc.fit_transform(lab['cell_type'])
X=lab[names+['size:size']].rename(columns={'size:size':'Cell'})
# identical to quantcell_project.process_data(): train_size=0.8, stratify, seed 42
Xtr,Xte,ytr,yte=train_test_split(X,y,train_size=0.8,stratify=y,random_state=42)
# MAPS splits its training data again into train/valid (paper used 80/20, seed 42)
Xt,Xv,yt,yv=train_test_split(Xtr,ytr,test_size=0.2,stratify=ytr,random_state=42)
for nm,(A,b) in {'train':(Xt,yt),'valid':(Xv,yv),'test':(Xte,yte)}.items():
    f=A.reset_index(drop=True).copy(); f['cell_label']=b
    f.to_csv(f'{OUT}/{nm}_features.csv', index=False)
    print(f'  {nm:6s} {f.shape}  -> {OUT}/{nm}_features.csv', flush=True)
json.dump({'classes':list(enc.classes_),'n_features':len(names)+1,'n_classes':len(enc.classes_),
           'n_test':int(len(yte))}, open(f'{OUT}/../../maps_meta.json','w'), indent=2)
np.save(f'{OUT}/../../y_test.npy', yte)
print(f"  {len(enc.classes_)} classes, {len(names)+1} features, test {len(yte):,}")
