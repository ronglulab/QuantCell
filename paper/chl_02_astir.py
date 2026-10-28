import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import sys, time, torch, numpy as np, pandas as pd
from astir.data import from_csv_yaml
torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(8)
B=f'{QCP.OUT}/chl/{sys.argv[1]}_priors/Astir'
ast=from_csv_yaml(f'{B}/marker_data.csv', f'{B}/marker_combos.yaml'); t=time.time(); ast.fit_type()
ast.get_celltypes().to_csv(f'{B}/astir_labels.csv'); pd.Series(time.time()-t).to_csv(f'{B}/astir_time.csv'); print(f'astir done {time.time()-t:.0f}s', flush=True)
