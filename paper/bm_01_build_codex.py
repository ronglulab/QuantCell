"""Build the QuantCell input table for the bone marrow dataset from the QuPath exports in the Zenodo deposit:
initialize() -> set_sample_labels() -> annotate() (binary marker rule-based annotation) -> drop_missing() -> section_samples().
Writes codex_rule_based_QuantCellPaper.csv to the random forest and LightGBM project folders. On the deposit this reproduces
quantcell/codex_rule_based_QuantCellPaper.csv (same cells, marker calls, cell types, features and sections; see
qupath_exports/full_panel_rebuild_validation.json)."""
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config as QCP
sys.path.insert(0, QCP.CODE); os.chdir(QCP.CODE)
from codex_project import codex_project
NAME='QuantCellPaper'; parent, folder = os.path.split(QCP.QUPATH.rstrip('/'))
cp=codex_project()
cp.initialize(data_path=parent, folders=[folder], project_name=NAME, annotation_strategy_path=QCP.COMBOS)
cp.set_sample_labels({folder: NAME})
cp.annotate()
cp.drop_missing(max_missing_per_col=1000000, max_missing_per_row=2)
cp.section_samples(eps=500)
print('cells', cp.codex.shape[0], '| rule-based annotated', round((cp.codex.cell_type != 'Other').mean(), 4), '| sections', cp.codex.section.nunique())
for d in ('retrain', 'retrain_lgbm'):
    out=f'{QCP.OUT}/{d}/{NAME}'; os.makedirs(out, exist_ok=True)
    if d == 'retrain': cp.save_csv(f'{out}/codex_rule_based_{NAME}.csv'); first=f'{out}/codex_rule_based_{NAME}.csv'
    else: shutil.copy2(first, f'{out}/codex_rule_based_{NAME}.csv')
    print('written', f'{out}/codex_rule_based_{NAME}.csv')
