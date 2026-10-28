""" Main-text Table 1 (comparison of QuantCell with existing annotation frameworks) rendered in the shared table style
(24 pt, _table_png.render). Descriptive columns as published submission, reference/figures_main/image4.png);
CELESTA row added (in the revised Fig 4a); 'Macro average precision' = Fig 4a bone marrow held-out macro precision over all 21
classes (retrain/figures/fig4_v2/fig4_data_bm.json, QuantCell = LightGBM @ 5 % FDR), best value bold. Row order AnnoSpat, Astir, CELESTA, MAPS, QuantCell ."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import json
from _table_png import render
D=f'{QCP.OUT}/retrain_lgbm/figures'
bm=json.load(open(f'{QCP.OUT}/retrain/figures/fig4_v2/fig4_data_bm.json'))['methods']
P={m:v['macro_precision_all_classes'] for m,v in bm.items()}; best=max(P.values())
desc={  # model: (method, input data, manual annotation required, marker combos, uncertainty / allows unlabelled)
 'AnnoSpat': ('Clustering',                         'Expression-by-cell matrix\n& marker combos',             'No',  'Positive &\nnegative', 'Yes'),
 'Astir':    ('Clustering',                         'Expression-by-cell matrix\n& marker combos',             'No',  'Positive',            'Yes'),
 'CELESTA':  ('Iterative scoring with\nspatial neighbours', 'Expression-by-cell matrix,\ncell coordinates & marker combos', 'No', 'Positive &\nnegative', 'Yes'),
 'MAPS':     ('Feed-forward\nneural network',       'Expression-by-cell matrix\n& cell size',                 'Yes', 'No',                  'No'),
 'QuantCell':('Variable\nclassifiers',              'Expression-by-cell matrix\n& marker combos',             'No',  'Positive &\nnegative', 'Yes'),
}
rows=[[(m,True)]+list(desc[m])+[(f'{P[m]:.3f}', P[m]==best)] for m in desc]
render(['Model','Method','Input data','Manual cell type\nannotation required','Marker\ncombos','Uncertainty\n(allows unlabelled)','Macro average\nprecision'],
       [3.3,5.6,7.3,4.6,3.2,4.9,4.0], rows, f'{D}/table1.png')
json.dump({m:round(P[m],4) for m in desc}, open(f'{D}/table1_precision.json','w'), indent=1)
