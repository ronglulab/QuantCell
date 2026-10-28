""" Suppl Table 2 rendered as PNG pages , same typeface/size as the figures (24 pt). Two pages of
hyper-parameters (model order = tuned macro AP best -> worst; page 1 = LightGBM/RF/MLP/ET). The metrics page was DROPPED
 duplicates Fig 2a); suppl_table2_metrics*.csv remain as source data."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import textwrap, pandas as pd
from _table_png import render
D=f'{QCP.OUT}/retrain_lgbm/figures'
# ---- params
p=pd.read_csv(f'{D}/suppl_table2_params.csv', dtype=str)
# model order = tuned macro AP, best to worst (same as the metrics page); page 1 = top four, page 2 = the rest
order=list(pd.read_csv(f'{D}/suppl_table2_metrics.csv')['model']); assert set(order)==set(p['model']); pages=[order]   # one long image instead of two pages
for pi,ms in enumerate(pages, start=1):
    rows=[]; breaks=[]
    for m in ms:
        g=p[p['model']==m]; breaks.append(len(rows))
        for j,(_,r) in enumerate(g.iterrows()):
            rows.append([m if j==0 else '', r['parameter'].replace('_',' '), '\n'.join(textwrap.wrap(r['search_space'], 42)), r['selected'],
                         (f"{float(r['cv_macro_AP_best']):.3f}", m==order[0]) if j==0 else '', f"{int(r['n_candidates']):,}" if j==0 else ''])
    render(['Model','Hyper-parameter','Search space','Selected','Best CV\nmacro AP','Candidates'], [5.6,4.6,9.4,3.8,2.9,2.9], rows, f'{D}/suppl_table2_params.png', group_breaks=breaks[1:])
