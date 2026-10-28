"""R2 -- stats_loc_dict_{included,excluded} on the codex (paper/04 cells 16/18; Fig 2e/2f).
SEEDED. Five locations incl. Membrane (the shipped cells omit it; the published artifact has it ).
Resumable: checkpoints after every (location, statistic) cell."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys, time, json, argparse, warnings, numpy as np, joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import R, CODE, RB, NAME, PROJ, MS, make_clf
ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['included','excluded'],required=True)
ap.add_argument('--njobs',type=int,default=12); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
LOC=['Membrane','Cytoplasm','Nucleus','Cell','All']; STAT=['Min','Max','Std.Dev.','Median','Mean','All']
sys.path.insert(0,CODE); os.chdir(CODE)
from quantcell import quantcell_project
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s][{a.mode}] {m}", flush=True)
p=quantcell_project(); p.initialize(base_path=RB, project_name=NAME); p.process_data()
log(f"X_train {p.X_train.shape}")
if a.smoke: log("SMOKE OK"); sys.exit(0)
p.set_clf(make_clf(a.njobs))
ck=f'{MS}/stats_loc_dict_{a.mode}.joblib'; d=joblib.load(ck) if os.path.exists(ck) else {}
for L in LOC:
    d.setdefault(L,{})
    inc=[c for c in p.column_names if L in c] if L!='All' else list(p.column_names)
    for S in STAT:
        if S in d[L]: log(f"  {L}/{S} cached"); continue
        sel=inc if S=='All' else [c for c in inc if S in c]
        if len(sel)==0:  # location absent from this codex (Cellpose codex has no Membrane compartment)
            d[L][S]=[None,None,None]; joblib.dump(d,ck); log(f"  {L}/{S} skipped (no columns)"); continue
        if a.mode=='excluded' and L=='All' and S=='All':
            d[L][S]=[None,None,None]; joblib.dump(d,ck); log(f"  {L}/{S} skipped (excluding everything)"); continue
        t=time.time(); kw={'included_cols':sel} if a.mode=='included' else {'excluded_cols':sel}
        d[L][S]=list(p.cv_fit_pred(name=f'RF_{L}_{S}', **kw)); joblib.dump(d,ck)
        log(f"  {L}/{S:<9} {len(sel):>3} cols  {time.time()-t:6.1f}s")
grid={}
for L in LOC:
    grid[L]={}
    for S in STAT:
        yt,pr=d[L][S][0],d[L][S][2]
        grid[L][S]=None if yt is None else float(np.mean([average_precision_score(yt[n],pr[n],average='macro') for n in yt]))
json.dump(grid, open(f'{MS}/stats_loc_{a.mode}_grid.json','w'), indent=2)
for L in LOC: log("  "+L.ljust(10)+"  ".join(f"{S}={grid[L][S]:.4f}" if grid[L][S] is not None else f"{S}=--" for S in STAT))
log("DONE")
