"""Shared constants for the retrain scripts. QC_CLF=rf (default) or lgbm selects the classifier and routes
outputs to parallel directories (suffix _lgbm) so the two sets never overwrite each other."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os, sys
R=f'{QCP.REPO}'; CODE=f'{QCP.CODE}'
CLF=os.environ.get('QC_CLF','rf'); assert CLF in ('rf','lgbm'), CLF
SUF='' if CLF=='rf' else f'_{CLF}'
RB=f'{QCP.OUT}/retrain{SUF}'; NAME=os.environ.get('QC_NAME','QuantCellPaper'); PROJ=f'{RB}/{NAME}'; MS=f'{PROJ}/model_selection'  # QC_NAME selects the project (Cellpose variant)
RF_PARAMS=f'{QCP.OUT}/retrain/QuantCellPaper/model_selection/tuning_all10/RandomForestClassifier_best_params_macro.joblib'  # corrected-search optimum use everywhere); published setting = ../RandomForestClassifier_best_params_macro.joblib
LGBM_PARAMS=f'{QCP.OUT}/retrain/QuantCellPaper/model_selection/tuning_all10/LGBMClassifier_best_params_macro.joblib'
def best_params():
    import joblib; return joblib.load(RF_PARAMS if CLF=='rf' else LGBM_PARAMS)
def make_clf(n_jobs, random_state=42):
    """The tuned classifier in use: published-setting RF, or the M20-corrected-search LightGBM (hessian floor 1)."""
    if CLF=='rf':
        from sklearn.ensemble import RandomForestClassifier; return RandomForestClassifier(**best_params(), n_jobs=n_jobs, random_state=random_state)
    sys.path.insert(0, f'{QCP.CODE}'); from lightgbm import LGBMClassifier
    return LGBMClassifier(**best_params(), n_jobs=n_jobs, random_state=random_state, verbose=-1)
