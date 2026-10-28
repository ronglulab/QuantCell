""" Suppl Table 3 ): the A138 benchmark metrics table as PNG pages, 24 pt.
Page 1: per dataset and method, coverage and overall metrics on the Fig 4 held-out cells (precision on assigned cells; recall /
F1 / accuracy over all held-out cells with unassigned = missed; accuracy on assigned cells). Page 2: the same precision / recall /
F1 split into rare (<10 % of held-out cells) and abundant (>=10 %) classes. Methods alphabetical; QuantCell's classifier in the name."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import sys, pandas as pd
SUF=('_'+sys.argv[1]) if len(sys.argv)>1 else ''   
from _table_png import render
D=f'{QCP.OUT}/retrain_lgbm/figures'
t=pd.read_csv(f'{D}/suppl_table_benchmarks{SUF}.csv'); f=lambda v: f'{v:.3f}'
name=lambda r: f"QuantCell ({'LightGBM' if r.quantcell_classifier=='LightGBM' else 'random forest'})" if r.method=='QuantCell' else r.method
DS=list(dict.fromkeys(t['dataset'])); rows1=[]; rows2=[]; breaks=[]
# bold the best score in every metric column within each dataset (ties at 3 dp all bold)
def B(g, col, v): return (f(v), round(v,3)>=round(g[col].max(),3))
for ds in DS:
    g=t[t['dataset']==ds]; breaks.append(len(rows1))
    for j,r in enumerate(g.itertuples()):
        first=(j==0)
        rows1.append([ds.replace(' (', '\n(') if first else '', name(r), f'{r.n_heldout:,}' if first else '', str(r.n_classes) if first else '', B(g,'coverage',r.coverage), B(g,'precision_macro',r.precision_macro), B(g,'recall_macro',r.recall_macro), B(g,'f1_macro',r.f1_macro), B(g,'accuracy_all_cells',r.accuracy_all_cells), B(g,'accuracy_assigned_cells',r.accuracy_assigned_cells)])
        rows2.append([ds.replace(' (', '\n(') if first else '', name(r), str(r.n_rare_classes) if first else '', B(g,'rare_precision',r.rare_precision), B(g,'rare_recall',r.rare_recall), B(g,'rare_f1',r.rare_f1), str(r.n_abundant_classes) if first else '', B(g,'abundant_precision',r.abundant_precision), B(g,'abundant_recall',r.abundant_recall), B(g,'abundant_f1',r.abundant_f1)])
render(['Dataset','Method','Held-out\ncells','Classes','Coverage','Precision','Recall','F1','Accuracy\n(all cells)','Accuracy\n(assigned)'],
       [5.2,6.2,2.9,2.4,2.6,2.7,2.3,2.0,2.9,2.9], rows1, f'{D}/suppl_table3{SUF}_p1.png', group_breaks=breaks[1:])
render(['Dataset','Method','Cell types\nbelow 0.1\nfrequency','Precision\n(<0.1)','Recall\n(<0.1)','F1\n(<0.1)','Cell types\nat or above\n0.1 frequency','Precision\n(≥0.1)','Recall\n(≥0.1)','F1\n(≥0.1)'],   # frequency wording instead of rare / abundant
       [5.2,6.2,2.6,2.9,2.6,2.3,2.9,3.2,2.9,2.9], rows2, f'{D}/suppl_table3{SUF}_p2.png', group_breaks=breaks[1:])
