""" Suppl Table 1 (marker combinations, 21 cell types) rendered as a PNG page in the shared table style (24 pt).
Reads suppl_table1.csv (A134; identical markers to the published table, display conventions CD8α / MHC II / F4/80)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import textwrap, pandas as pd
from _table_png import render
D=f'{QCP.OUT}/retrain_lgbm/figures'
t=pd.read_csv(f'{D}/suppl_table1.csv').fillna('')
# references 's Cell_Annotation_References.xlsx, 2026-10-18) -> numbered by first appearance; list on page 2
R=pd.read_csv(f'{QCP.INPUTS}/suppl_table1_refs.csv').set_index('json_key'); assert set(R.index)==set(t.json_key), set(t.json_key)^set(R.index)
t['pmids']=t.json_key.map(R['pmids']); t['reference_urls']=t.json_key.map(R['reference_urls'])   # PubMed IDs instead of links
t.to_csv(f'{D}/suppl_table1_with_refs.csv', index=False)
w=lambda s,n: '\n'.join(textwrap.wrap(s, n)) if s else '–'
rows=[[w(r.cell_type,34), w(r.positive_markers,22), w(r.negative_markers,40), w(r.pmids,20)] for r in t.itertuples()]
render(['Cell type','Positive markers','Negative markers','PubMed ID'], [8.6,5.6,9.6,5.4], rows, f'{D}/suppl_table1.png')
