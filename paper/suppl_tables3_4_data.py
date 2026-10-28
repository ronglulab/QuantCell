""" Supplementary benchmark table ). For every Fig 4 dataset and method,
on the same held-out cells as Fig 4 v2: coverage, macro precision (assigned cells), macro recall and macro F1 (all held-out cells,
unassigned = missed), accuracy over all cells and over assigned cells, and the rare (<10 % held-out frequency) vs abundant split.
Reads retrain/figures/fig4_v2/fig4_data_*.json written by fig4_v2_data.py. Methods alphabetical."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import json, sys, pandas as pd
D=f'{QCP.OUT}/retrain/figures/fig4_v2'; OUT=f'{QCP.OUT}/retrain_lgbm/figures'
DS=[('bm','Bone marrow'),('colon','Colon'),('chl1','Lymph node (cHL1)'),('codex','Lymph node (cHL CODEX)'),('chl2','Lymph node (cHL2)')]
SUF=''
if len(sys.argv)>2: DS=[(sys.argv[1], sys.argv[2])]; SUF='_'+sys.argv[1]   # e.g. colon16 'Colon (16 CELESTA-signature types)'
METHODS=['AnnoSpat','Astir','CELESTA','MAPS','QuantCell']; rows=[]
for k,name in DS:
    j=json.load(open(f'{D}/fig4_data_{k}.json'))
    for m in METHODS:
        r=j['methods'][m]; ra,ab=r['rare'],r['abundant']
        rows.append({'dataset':name,'method':m,'quantcell_classifier':j['quantcell_classifier'] if m=='QuantCell' else '','n_heldout':j['n_heldout'],'n_classes':len(j['classes']),
            'coverage':r['coverage'],'precision_macro':r['macro_precision_all_classes'],'recall_macro':r['macro_recall_all_classes'],'f1_macro':r['macro_f1_all_classes'],
            'accuracy_all_cells':r['accuracy_all_cells'],'accuracy_assigned_cells':r['accuracy_assigned_cells'],
            'n_rare_classes':ra['n_classes'],'rare_precision':ra['macro_precision'],'rare_recall':ra['macro_recall'],'rare_f1':ra['macro_f1'],
            'n_abundant_classes':ab['n_classes'],'abundant_precision':ab['macro_precision'],'abundant_recall':ab['macro_recall'],'abundant_f1':ab['macro_f1']})
t=pd.DataFrame(rows); t.to_csv(f'{OUT}/suppl_table_benchmarks{SUF}.csv', index=False, float_format='%.3f'); print('wrote', f'{OUT}/suppl_table_benchmarks{SUF}.csv')
f=lambda v: '' if v is None or v!=v else f'{v:.3f}'
print('\n| Dataset | Method | Cov. | P | R | F1 | Acc (all) | Acc (assigned) | Rare P / R / F1 (n) | Abundant P / R / F1 (n) |\n|---|---|---|---|---|---|---|---|---|---|')
for r in rows: print(f"| {r['dataset']} | {r['method']}{' ('+r['quantcell_classifier']+')' if r['quantcell_classifier'] else ''} | {f(r['coverage'])} | {f(r['precision_macro'])} | {f(r['recall_macro'])} | {f(r['f1_macro'])} | {f(r['accuracy_all_cells'])} | {f(r['accuracy_assigned_cells'])} | {f(r['rare_precision'])} / {f(r['rare_recall'])} / {f(r['rare_f1'])} ({r['n_rare_classes']}) | {f(r['abundant_precision'])} / {f(r['abundant_recall'])} / {f(r['abundant_f1'])} ({r['n_abundant_classes']}) |")
