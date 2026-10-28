"""Generic prep for the cHL1 MIBI and cHL CODEX five-method runs (same recipe as cHL2, ). Usage: chl_prep.py chl1|codex
Writes <C>/<ds>_priors/{truth.csv, Astir/, AnnoSpat/, CELESTA/} from the MAPS positive-only list, and <C>/<ds>_maps/ (fixed
GT-stratified 20% hold-out, seed 42; features = all markers + cellSize)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, re, sys, json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
ds=sys.argv[1]; R=f'{QCP.REPO}'; RAW=f'{QCP.EXT}'; C=f'{QCP.OUT}/chl'
CFG={'chl1':dict(src=f'{RAW}/cHL_1_MIBI/annotation_csv/cHL1_MIBI.csv', yml='maps_cHL1_MIBI_marker.yml', label='Annotation', x='centroidX', y='centroidY', size='cellSize', sample='identifier', unlab={'Other'}, drop={'cellLabel','Annotation','centroidX','centroidY','cellSize','identifier'}),
     'codex':dict(src=f'{RAW}/cHL_CODEX/annotation_csv/cHL_CODEX_annotation.csv', yml='maps_cHL_CODEX_marker.yml', label='cellType', x='X_cent', y='Y_cent', size='cellSize', sample=None, unlab={'Other','Seg Artifact'}, drop={'cellLabel','X_cent','Y_cent','cellSize','cellType','DAPI.01'})}[ds]
O=f'{C}/{ds}_priors'; M=f'{C}/{ds}_maps'; [os.makedirs(f'{O}/{s}', exist_ok=True) for s in ('Astir','AnnoSpat/outputdir','CELESTA/per_sample')]; os.makedirs(f'{M}/data/cell_phenotyping', exist_ok=True); os.makedirs(f'{M}/results/cell_phenotyping', exist_ok=True)
def read_marker_yml(path):
    ct={}; cur=None
    for line in open(path):
        if line.strip().startswith('cell_types'): continue
        if line.startswith('  ') and not line.strip().startswith('-') and line.strip().endswith(':'): cur=line.strip()[:-1]; ct[cur]=[]
        elif line.strip().startswith('- ') and cur: ct[cur].append(line.strip()[2:].strip())
    return ct
rules=read_marker_yml(f'{QCP.INPUTS}/{CFG["yml"]}'); rm=sorted({m for v in rules.values() for m in v})
d=pd.read_csv(CFG['src'], low_memory=False); d.columns=[c.strip('"') for c in d.columns]; assert all(m in d.columns for m in rm), [m for m in rm if m not in d.columns]
markers=[c for c in d.columns if c not in CFG['drop']]; gt=d[CFG['label']].astype(str).where(~d[CFG['label']].astype(str).isin(CFG['unlab']),'Other').values
sample=d[CFG['sample']].astype(str).values if CFG['sample'] else np.array(['fov1']*len(d))
print(f'{ds}: {len(d):,} cells, {len(markers)} markers, {len(np.unique(sample))} samples, rule markers {len(rm)}, types {len(rules)}, labelled {(gt!="Other").sum():,}', flush=True)
safe={m:re.sub(r'[^A-Za-z0-9_]','_',m) for m in rm}
pd.DataFrame({'row':np.arange(len(d)),'sample':sample,'cellLabel':d.cellLabel,'X':d[CFG['x']],'Y':d[CFG['y']],'cell_type':gt}).to_csv(f'{O}/truth.csv', index=False)
X=d[rm].rename(columns=safe); X.reset_index().to_csv(f'{O}/Astir/marker_data.csv', index=True)
with open(f'{O}/Astir/marker_combos.yaml','w') as f: f.write('cell_types:\n'); [f.write(f'  {ct}:\n'+''.join(f'  - {safe[m]}\n' for m in ms)) for ct,ms in rules.items()]
A=X.copy(); A['ROI']=sample if len(np.unique(sample))>1 else np.arange(len(A)); A.to_csv(f'{O}/AnnoSpat/marker_data.csv', index=True)   # single-field sets: ROI = row index, as the published BM run
n=max(len(v) for v in rules.values()); pd.DataFrame({ct.replace('+',''):[safe[m] for m in ms]+['']*(n-len(ms)) for ct,ms in rules.items()}, index=range(1,n+1)).to_csv(f'{O}/AnnoSpat/marker_combos.csv')
json.dump({ct.replace('+',''):ct for ct in rules}, open(f'{O}/AnnoSpat/name_map.json','w'))
prior=pd.DataFrame(index=list(rules), columns=[safe[m] for m in rm], dtype=object)
for k,(ct,ms) in enumerate(rules.items(), start=1):
    for m in ms: prior.loc[ct, safe[m]]=1
prior.insert(0,'Lineage_level',[f'1_0_{k}' for k in range(1,len(rules)+1)]); prior.index.name=''; prior.to_csv(f'{O}/CELESTA/prior_marker_info.csv', na_rep='NA')
img=pd.DataFrame({'CellID':np.arange(len(d)),'sample':sample,'X':d[CFG['x']].values,'Y':d[CFG['y']].values}); [img.__setitem__(safe[m], d[m].values) for m in rm]; img.to_csv(f'{O}/CELESTA/imaging_data.csv', index=False)
# MAPS on the fixed split
lab_rows=np.where(gt!='Other')[0]; tr,ho=train_test_split(np.arange(len(lab_rows)), train_size=0.8, stratify=gt[lab_rows], random_state=42); enc=LabelEncoder().fit(gt[lab_rows])
F=d[markers].copy(); F['Cell']=d[CFG['size']].values
Xtr=F.iloc[lab_rows[tr]].reset_index(drop=True); ytr=enc.transform(gt[lab_rows[tr]]); Xte=F.iloc[lab_rows[ho]].reset_index(drop=True); yte=enc.transform(gt[lab_rows[ho]])
Xt,Xv,yt,yv=train_test_split(Xtr,ytr,test_size=0.2,stratify=ytr,random_state=42)
for nm,(Aa,b) in {'train':(Xt,yt),'valid':(Xv,yv),'test':(Xte,yte)}.items(): f=Aa.reset_index(drop=True).copy(); f['cell_label']=b; f.to_csv(f'{M}/data/cell_phenotyping/{nm}_features.csv', index=False)
json.dump({'classes':list(enc.classes_),'n_features':F.shape[1],'n_classes':len(enc.classes_),'n_test':int(len(yte))}, open(f'{M}/maps_meta.json','w'), indent=2); pd.Series(lab_rows[ho]).to_csv(f'{M}/heldout_rows.csv', index=False)
print(f'  AnnoSpat first/last: {X.columns[0]} {X.columns[-1]} | n types {len(rules)} | hold-out {len(ho):,} | MAPS features {F.shape[1]}', flush=True)
