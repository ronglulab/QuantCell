import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, time, json, pandas as pd
from maps.cell_phenotyping import Trainer, Predictor
B=f'{QCP.OUT}/workspace_out/colon'+('16' if os.environ.get('QC_COLON16')=='1' else '')+'/MAPS'
D=f'{B}/data/cell_phenotyping'; R=f'{B}/results/cell_phenotyping'
os.makedirs(R, exist_ok=True)
meta=json.load(open(f'{B}/maps_meta.json'))
nf, nc = meta['n_features'], meta['n_classes']
print(f'  num_features={nf} num_classes={nc}', flush=True)
t=time.time()
model=Trainer(results_dir=R, num_features=nf, num_classes=nc, verbose=1, min_epochs=5, patience=25)
model.fit(f'{D}/train_features.csv', f'{D}/valid_features.csv')
pred=Predictor(model_checkpoint_path=f'{R}/best_checkpoint.pt', num_features=nf, num_classes=nc)
pl, pp = pred.predict(f'{D}/test_features.csv')
el=time.time()-t
pd.Series(pl).to_csv(f'{R}/pred_labels.csv', index=True)
pd.DataFrame(pp).to_csv(f'{R}/pred_probs.csv', index=True)
pd.Series([el]).to_csv(f'{R}/time_elapsed_maps.csv', index=True)
print(f'  done in {el:.1f}s -> {R}/pred_labels.csv', flush=True)
