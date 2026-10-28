"""R2 -- gene_ko_dict on the codex (paper/04 cell 47; Fig 5b, Suppl 10).
NESTED split (cells 24/25 -> 41-43), SEEDED, and with the M15 fix: each marker's knockout removes
exactly its own 20 columns (exact 'marker:' prefix), not every column containing the substring.
Resumable: checkpoints after every marker."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, time, json, argparse, warnings, numpy as np, pandas as pd, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, CODE, RB, NAME, PROJ, MS, make_clf
ap=argparse.ArgumentParser(); ap.add_argument('--njobs',type=int,default=16); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
sys.path.insert(0,CODE); os.chdir(CODE)
from quantcell import quantcell_project
from codex_project import codex_project
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s][gene_ko] {m}", flush=True)
cx=codex_project(); cx.read_csv(f'{PROJ}/codex_rule_based_{NAME}.csv', project_name=NAME)
cx.codex=cx.codex.loc[cx.codex['cell_type']!='Other',:].reset_index(drop=True)
Xtr,Xte,_,_=train_test_split(cx.codex, cx.codex['cell_type'], test_size=0.2, random_state=42, stratify=cx.codex['cell_type'])
Xte_w=Xte.copy(); Xte_w['cell_type']='Other'
p=quantcell_project(); p.initialize(base_path=RB, project_name=NAME)
p.codex=pd.concat([Xtr,Xte_w],axis=0,ignore_index=True); p.process_data()
log(f"nested X_train {p.X_train.shape} (80% of labelled); held-out {p.unlabeled_features.shape}")
gene_order=sorted({c.split(': Cell: Mean')[0] for c in p.column_names if ': Cell: Mean' in c}, key=str.lower)
log(f"{len(gene_order)} markers: {gene_order}")
sizes={m:len([c for c in p.column_names if c.startswith(m+':')]) for m in gene_order}
assert all(v==20 for v in sizes.values()), f"M15 selection wrong: {sizes}"
if a.smoke: log("SMOKE OK"); sys.exit(0)
p.set_clf(make_clf(a.njobs))
ck=f'{MS}/gene_ko_dict.joblib'; gko=joblib.load(ck) if os.path.exists(ck) else {}
for n,marker in enumerate(gene_order):
    if marker in gko: log(f"  [{n+1:>2}/25] {marker} cached"); continue
    t=time.time(); sel=[c for c in p.column_names if c.startswith(marker+':')]
    gko[marker]=list(p.cv_fit_pred(name=f'RF_ko_{marker}', excluded_cols=sel)); joblib.dump(gko, ck)
    log(f"  [{n+1:>2}/25] {marker:<10} {len(sel):>3} cols excluded  {time.time()-t:6.1f}s")
log("DONE")
