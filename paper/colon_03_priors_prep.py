"""Astir + AnnoSpat on colon driven by CELESTA's published prior . Inputs: the 18 prior markers of the
published colon table (258,385 cells, all regions) under short names; Astir YAML = markers marked 1 per type; AnnoSpat
signature = markers marked 1 plus 'marker-' for markers marked 0, type names without '+' (AnnoSpat parses +/-), ROI = Region.
Truth = published ClusterName (27 types; dirt/undefined -> Other)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, re, numpy as np, pandas as pd
C=f'{QCP.OUT}/celesta'; O=f'{C}/colon/priors'; os.makedirs(f'{O}/Astir', exist_ok=True); os.makedirs(f'{O}/AnnoSpat/outputdir', exist_ok=True)
prior=pd.read_csv(f'{QCP.INPUTS}/celesta_colon_prior_marker_info.csv', index_col=0)   # CELESTA's published colon signature; mcols=list(prior.columns[1:]); short={c:c.split(' - ')[0].strip() for c in mcols}; assert len(set(short.values()))==18
df=pd.read_csv(f'{C}/../colon/CRC_clusters_neighborhoods_markers.csv', usecols=['CellID','Region','patients','ClusterName']+mcols, low_memory=False)
X=df[mcols].rename(columns=short); truth=df['ClusterName'].where(~df['ClusterName'].isin({'dirt','undefined'}),'Other')
pd.DataFrame({'CellID':df.CellID,'Region':df.Region,'patients':df.patients,'cell_type':truth}).to_csv(f'{O}/truth.csv', index=False)
# Astir
X.reset_index().to_csv(f'{O}/Astir/marker_data.csv', index=True)
with open(f'{O}/Astir/marker_combos.yaml','w') as f:
    f.write('cell_types:\n')
    for ct,row in prior[mcols].iterrows():
        posm=[short[m] for m in mcols if row[m]==1]; assert posm, ct
        f.write(f'  {ct}:\n'+''.join(f'  - {m}\n' for m in posm))
# AnnoSpat
A=X.copy(); A['ROI']=df['Region'].values; A.to_csv(f'{O}/AnnoSpat/marker_data.csv', index=True)
cols={}
for ct,row in prior[mcols].iterrows():
    entries=[short[m] for m in mcols if row[m]==1]+[short[m]+'-' for m in mcols if row[m]==0]; cols[ct.replace('+','')]=entries
n=max(len(v) for v in cols.values()); mc=pd.DataFrame({k:v+['']*(n-len(v)) for k,v in cols.items()}, index=range(1,n+1)); mc.to_csv(f'{O}/AnnoSpat/marker_combos.csv')
pd.Series({ct.replace('+',''):ct for ct in prior.index}).to_json(f'{O}/AnnoSpat/name_map.json')
print('cells', len(df), 'markers', list(X.columns)); print('types', list(prior.index)); print('AnnoSpat first/last protein:', X.columns[0], X.columns[-1], '| n types', len(cols))
