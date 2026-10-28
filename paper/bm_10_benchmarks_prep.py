""" Inputs for Astir / AnnoSpat / MAPS on the bone-marrow labels, in the published formats.
Astir + AnnoSpat: every labelled cell (cell_type != Other), 25 'Cell: Mean' columns named by marker, truth = cell_type.
Marker-combination files = the published ones with the five renamed cell types (combos unchanged, ).
AnnoSpat's ROI column was the row index in the published run (every cell its own ROI); reproduced as such.
MAPS: the retrain's nested split -- train/valid = ComparisonsTrainTest Xtr split 80/20 stratified seed 42,
test = Xte (20,154) -- 25 means + area, as published (num_features 26)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, json, re, numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _common import PROJ, NAME
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
P=f'{QCP.INPUTS}/bm_published'; B=f'{QCP.OUT}/retrain/benchmarks'
OLD2NEW={'Arteries':'Arterial ECs','Capillaries/Sinusoids':'Capillary ECs','Endothelial Niche':'Other BM ECs','Perivascular Niche':'Lepr+ Perivascular Cells','Preplatelets':'Megakaryocytes'}
cx=pd.read_csv(f'{PROJ}/codex_rule_based_{NAME}.csv', low_memory=False)
mk=sorted({c.split(':')[0] for c in cx.columns if c.endswith(': Cell: Mean')}); assert len(mk)==25
lab=cx[cx.cell_type!='Other'].reset_index(drop=True); X=lab[[f'{m}: Cell: Mean' for m in mk]].copy(); X.columns=mk
types=sorted(lab.cell_type.unique()); print(f"labelled {len(lab):,} cells, {len(types)} types, markers {mk}")
# Astir
X.reset_index().to_csv(f'{B}/Astir/marker_data.csv', index=True)      # published layout: unnamed index + 'index' column
lab[['cell_type']].to_csv(f'{B}/Astir/astir_true_labels.csv', index=False)
# published yaml is 'cell_types:' then ' Name:' / ' - marker' lines; parse and rewrite without a yaml library (not in repro311)
ct={}; cur=None
for line in open(f'{P}/Astir/marker_combos.yaml'):
    if re.match(r'^  \S.*:\s*$', line): cur=line.strip()[:-1]; ct[cur]=[]
    elif line.strip().startswith('- ') and cur: ct[cur].append(line.strip()[2:])
ct={OLD2NEW.get(k,k):v for k,v in ct.items()}; assert set(ct)==set(types), (set(ct)^set(types))
with open(f'{B}/Astir/marker_combos.yaml','w') as f:
    f.write('cell_types:\n')
    for k in sorted(ct): f.write(f'  {k}:\n'+''.join(f'  - {m}\n' for m in ct[k]))
# AnnoSpat
A=X.copy(); A['ROI']=np.arange(len(A)); A.to_csv(f'{B}/AnnoSpat/marker_data.csv', index=True)
lab[['cell_type']].to_csv(f'{B}/AnnoSpat/annospat_true_labels.csv', index=True)
mc=pd.read_csv(f'{P}/AnnoSpat/marker_combos.csv', index_col=0); mc.columns=[OLD2NEW.get(c,c) for c in mc.columns]
assert set(mc.columns)==set(types)
mc.columns=[c.replace('+','') for c in mc.columns]   # AnnoSpat parses '+' in cell type names as a marker sign; mapped back when scoring
mc.to_csv(f'{B}/AnnoSpat/marker_combos.csv')
# MAPS
Xtr,Xte,ytr,yte=joblib.load(f'{PROJ}/ComparisonsTrainTest.joblib'); enc=LabelEncoder().fit(sorted(types))
def feats(df): f=df[[f'{m}: Cell: Mean' for m in mk]].copy(); f.columns=mk; f['Cell']=df['Cell: Area µm^2'].values; return f.reset_index(drop=True)
Ftr,Fte=feats(Xtr),feats(Xte); ytr_e=enc.transform(ytr.values); yte_e=enc.transform(yte.values)
Ft,Fv,yt,yv=train_test_split(Ftr,ytr_e,train_size=0.8,stratify=ytr_e,random_state=42)
for nm,(F,yy) in {'train':(Ft,yt),'valid':(Fv,yv),'test':(Fte,yte_e)}.items():
    f=F.reset_index(drop=True).copy(); f['cell_label']=yy; f.to_csv(f'{B}/MAPS/data/cell_phenotyping/{nm}_features.csv', index=False); print(f'  MAPS {nm} {f.shape}')
json.dump({'classes':list(enc.classes_),'n_features':Ftr.shape[1],'n_classes':len(enc.classes_),'n_test':int(len(yte_e))}, open(f'{B}/MAPS/maps_meta.json','w'), indent=2)
joblib.dump(enc, f'{B}/MAPS/label_encoder.joblib'); print("prep done")
