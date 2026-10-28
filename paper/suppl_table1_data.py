#!/usr/bin/env python
"""Supplementary Table 1 from the annotation strategy actually used (server JSON, names): cell type, positive
markers, negative markers, in the published display conventions (CD8 -> CD8a, MHCII -> MHC II, F4_80 -> F4/80,
full names in parentheses, alphabetical by display name). Output: suppl_table1.csv + markdown on stdout."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import json, pandas as pd
J=f'{QCP.COMBOS}'
OUT=f'{QCP.OUT}/retrain_lgbm/figures/suppl_table1.csv'
FULL={'Arterial ECs':'Arterial Endothelial Cells (Arterial ECs)','Capillary ECs':'Capillary Endothelial Cells (Capillary ECs)',
 'cDC':'Conventional Dendritic Cells (cDC)','CFU-E':'Colony Forming Units - Erythroid (CFU-E)','CLP':'Common Lymphoid Progenitors (CLP)',
 'CMP':'Common Myeloid Progenitors (CMP)','GMP':'Granulocyte-Macrophage Progenitors (GMP)','KLS':'cKit+Lineage-Sca1+ Cells (KLS)',
 'MEP':'Megakaryocyte-Erythroid Progenitors (MEP)','MkP':'Megakaryocyte Progenitors (MkP)','Monocytes':'Monocytes*',
 'Other BM ECs':'Other Bone Marrow Endothelial Cells (Other BM ECs)'}
DISP={'CD8':'CD8α','MHCII':'MHC II','F4_80':'F4/80'}
combos=json.load(open(J)); rows=[]
for ct,ms in combos.items():
    pos=sorted(DISP.get(m[:-1],m[:-1]) for m in ms if m.endswith('+')); neg=sorted(DISP.get(m[:-1],m[:-1]) for m in ms if m.endswith('-'))
    rows.append({'cell_type':FULL.get(ct,ct),'json_key':ct,'positive_markers':', '.join(pos),'negative_markers':', '.join(neg)})
t=pd.DataFrame(rows).sort_values('cell_type',key=lambda s:s.str.lower()).reset_index(drop=True)
t.to_csv(OUT,index=False); print(f'{len(t)} cell types -> {OUT}')
print('| Cell type | Positive markers | Negative markers |'); print('|---|---|---|')
for _,r in t.iterrows(): print(f'| {r.cell_type} | {r.positive_markers} | {r.negative_markers} |')
