"""Astir / AnnoSpat / CELESTA inputs for cHL2 MIBI from the MAPS authors' positive-only list (11 types, 15 markers)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, re, numpy as np, pandas as pd
R=f'{QCP.REPO}'; SRC=f'{QCP.EXT}/cHL_2_MIBI/annotation_csv'; O=f'{QCP.OUT}/chl/chl2_priors'
for sub in ('Astir','AnnoSpat/outputdir','CELESTA/per_fov'): os.makedirs(f'{O}/{sub}', exist_ok=True)
def read_marker_yml(path):
    ct={}; cur=None
    for line in open(path):
        if line.strip().startswith('cell_types'): continue
        if line.startswith('  ') and not line.strip().startswith('-') and line.strip().endswith(':'): cur=line.strip()[:-1]; ct[cur]=[]
        elif line.strip().startswith('- ') and cur: ct[cur].append(line.strip()[2:].strip())
    return ct
rules=read_marker_yml(f'{QCP.INPUTS}/maps_cHL2_MIBI_marker.yml'); rm=sorted({m for v in rules.values() for m in v})
d=pd.concat([pd.read_csv(f'{SRC}/{i}.csv').assign(fov=i) for i in range(1,7)], ignore_index=True); assert all(m in d.columns for m in rm)
safe={m:re.sub(r'[^A-Za-z0-9_]','_',m) for m in rm}            # AnnoSpat/Astir-safe column names (no spaces/brackets/'-')
X=d[rm].rename(columns=safe); truth=pd.DataFrame({'row':np.arange(len(d)),'fov':d.fov,'cellLabel':d.cellLabel,'X':d.X_cent,'Y':d.Y_cent,'cell_type':d.Annotation.astype(str)}); truth.to_csv(f'{O}/truth.csv', index=False)
# Astir
X.reset_index().to_csv(f'{O}/Astir/marker_data.csv', index=True)
with open(f'{O}/Astir/marker_combos.yaml','w') as f:
    f.write('cell_types:\n'); [f.write(f'  {ct}:\n'+''.join(f'  - {safe[m]}\n' for m in ms)) for ct,ms in rules.items()]
# AnnoSpat (positives only; type names have no '+'; ROI = fov)
A=X.copy(); A['ROI']=d.fov.values; A.to_csv(f'{O}/AnnoSpat/marker_data.csv', index=True)
n=max(len(v) for v in rules.values()); pd.DataFrame({ct:[safe[m] for m in ms]+['']*(n-len(ms)) for ct,ms in rules.items()}, index=range(1,n+1)).to_csv(f'{O}/AnnoSpat/marker_combos.csv')
# CELESTA: prior 1 / NA (no negatives), flat lineage; imaging data per cell with X, Y, fov
prior=pd.DataFrame(index=list(rules), columns=[safe[m] for m in rm], dtype=object)
for k,(ct,ms) in enumerate(rules.items(), start=1):
    for m in ms: prior.loc[ct, safe[m]]=1
prior.insert(0,'Lineage_level',[f'1_0_{k}' for k in range(1,len(rules)+1)]); prior.index.name=''; prior.to_csv(f'{O}/CELESTA/prior_marker_info.csv', na_rep='NA')
img=pd.DataFrame({'CellID':np.arange(len(d)),'fov':d.fov.values,'X':d.X_cent.values,'Y':d.Y_cent.values}); 
for m in rm: img[safe[m]]=d[m].values
img.to_csv(f'{O}/CELESTA/imaging_data.csv', index=False)
print('cells', len(d), '| markers', [safe[m] for m in rm], '| first/last for AnnoSpat:', X.columns[0], X.columns[-1], '| types', len(rules))
