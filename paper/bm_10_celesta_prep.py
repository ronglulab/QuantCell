"""CELESTA on bone marrow prior = the paper's 21 RENAMED marker combinations (1 = '+', 0 = '-', NA
otherwise), flat lineage (every type in round 1, '1_0_k'); imaging data = every cell of the codex (245,719) with
X, Y, section and the 25 'Cell: Mean' intensities under short marker names. CELESTA runs per section (its per-image design)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from _common import PROJ, NAME
B=f'{QCP.OUT}/retrain/benchmarks/CELESTA'; os.makedirs(f'{B}/per_section', exist_ok=True)
combos=json.load(open(f'{QCP.COMBOS}'))
mk=sorted({m[:-1] for v in combos.values() for m in v}); assert len(mk)==25
prior=pd.DataFrame(index=list(combos), columns=mk, dtype=object)
for k,(ct,rules) in enumerate(combos.items(), start=1):
    for r in rules: prior.loc[ct, r[:-1]] = 1 if r.endswith('+') else 0
prior.insert(0,'Lineage_level',[f'1_0_{k}' for k in range(1,len(combos)+1)])
prior.index.name=''; prior.to_csv(f'{B}/prior_marker_info.csv', na_rep='NA')
print('prior', prior.shape, 'positives/type', prior[mk].eq(1).sum(1).tolist(), 'negatives/type', prior[mk].eq(0).sum(1).tolist())
cx=pd.read_csv(f'{PROJ}/codex_rule_based_{NAME}.csv', low_memory=False)
img=pd.DataFrame({'CellID':np.arange(len(cx)),'section':cx['section'].values,'X':cx['x'].values,'Y':cx['y'].values,'cell_type':cx['cell_type'].values})
for m in mk: img[m]=cx[f'{m}: Cell: Mean'].values
assert img[mk].isna().sum().sum()==0; img.to_csv(f'{B}/imaging_data.csv', index=False)
print('imaging data', img.shape, 'sections', img.section.value_counts().to_dict())
