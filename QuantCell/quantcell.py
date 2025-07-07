import re
import json
import os
import joblib
import numpy as np
from tqdm import tqdm
import time

from codex_project import codex_project

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve



class quantcell_project:
    def __init__(self):
        self.clf = None
        self.base_path = None
        self.project_name = None
        self.codex = None
        self.codex_project = None
        self.random_seed = 42
        self.train_size = 0.8
        self.target_encoder = None
        self.cell_type_exclusion_threshold=None
        self.excluded_cell_types = ['Other']
        self.is_fitted = False
        self.relabeled = False


    def initialize(self, base_path, project_name):
        self.base_path = base_path
        self.project_name = project_name

        print(f"Initializing project {project_name} at {base_path}")
        self._load_codex()


    def _load_codex(self):
        self.codex = codex_project()
        codex_path = f"{self.base_path}/{self.project_name}/codex_conventional_{self.project_name}.csv"
        self.codex.read_csv(codex_path, self.project_name)

        self.codex_project = self.codex
        self.codex = self.codex.codex

    def _filter_metadata(self):
        mask=list(self.codex.columns.map(self._regex_marker_cols))
        return self.codex.loc[:, mask]

    def _regex_marker_cols(self, col):
        return re.search("\w+: (Nucleus|Cytoplasm|Membrane|Cell): (Mean|Median|Min|Max|Std.Dev.)", col) != None

   
    def process_data(self, cell_type_exclusion_threshold=0, excluded_cell_types=None):
        self.target_encoder = LabelEncoder()
        scaler = StandardScaler()

        features_df = self._filter_metadata()
        for x in features_df.columns:
            features_df.loc[:, x] = features_df.loc[:, x].astype(float)


        self.excluded_cell_types +=  excluded_cell_types if excluded_cell_types is not None else []
        self.cell_type_exclusion_threshold = cell_type_exclusion_threshold
        
        for cell_type in self.codex.loc[:, 'cell_type'].unique():
            if self.codex.loc[self.codex.loc[:, 'cell_type'] == cell_type].shape[0] < self.cell_type_exclusion_threshold:
                self.excluded_cell_types.append(cell_type)

        unlabeled_mask = self.codex.loc[:, 'cell_type'].isin(self.excluded_cell_types)
        target = self.target_encoder.fit_transform(self.codex.loc[~unlabeled_mask, 'cell_type'])
        labeled_features = features_df.loc[~unlabeled_mask, :]
        unlabeled_features = features_df.loc[unlabeled_mask, :]

        self.column_names = labeled_features.columns.tolist()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(labeled_features, target, train_size = self.train_size, stratify=target, random_state=self.random_seed)

        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        self.unlabeled_features = scaler.transform(unlabeled_features)

    def cv_fit_pred(self, name=None, excluded_cols=None, included_cols=None):
        if self.clf == None:
            raise ValueError("No classifier set")
        if excluded_cols and included_cols:
            raise ValueError("Cannot specify both excluded_cols and included_cols. Please choose one.")
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_seed)
        y_test={}
        y_pred={}
        y_proba={}
        i=0
        for train_index, test_index in tqdm(cv.split(self.X_train, self.y_train), total=cv.n_splits, desc=name):
            if excluded_cols is not None:
                col_mask = [True if col not in excluded_cols else False for col in self.column_names]
            elif included_cols is not None:
                col_mask = [True if col in included_cols else False for col in self.column_names]
            else:
                col_mask = [True] * len(self.column_names)
            
            X_train, X_test = self.X_train[train_index][:, col_mask], self.X_train[test_index][:, col_mask] 
            y_train, y_test[i] = self.y_train[train_index], self.y_train[test_index]
            self.clf.fit(X_train, y_train)
            y_pred[i] = self.clf.predict(X_test)
            try:
                y_proba[i] = self.clf.predict_proba(X_test)
            except:
                y_proba[i] = self.clf.decision_function(X_test)
            i+=1
        return y_test, y_pred, y_proba


    def fit(self, excluded_cols=None, included_cols=None):
        if self.clf == None:
            raise ValueError("No classifier set")
        
        self.clf.fit(self.X_train, self.y_train)
        self.is_fitted = True

    def find_fdr_threshold(self, min_FDR=0, max_FDR=0.25, number_points=100):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        fdr_cutoffs, thresholds = find_thresholds_at_varying_FDR(self.clf, self.X_test, self.y_test, min_FDR=min_FDR, max_FDR=max_FDR, number_points=number_points)
        unlabeled_features_proba = self.clf.predict_proba(self.unlabeled_features)

        for n, fdr in enumerate(fdr_cutoffs):
            annotated_mask = (unlabeled_features_proba >= thresholds)[np.arange(len(unlabeled_features_proba)), np.argmax(unlabeled_features_proba, axis=1)]
            fraction_annotated = np.sum(mask) / len(mask)

        fig, ax = plt.subplots(figsize=(10, 6))
        

    def relabel_unannotated(self, FDR_cutoff=0.05):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        thresholds = find_thresholds(self.clf, self.X_test, self.y_test, FDR_cutoff=FDR_cutoff)
        labels = apply_thresholds(self.clf, self.unlabeled_features, thresholds)
        self.thresholds = thresholds
        labels=[self.target_encoder.inverse_transform([label])[0] if label != -1 else 'Other' for label in labels]
        self.relabeled_labels = labels

    def overwrite_codex(self):
        self.codex.loc[self.codex.loc[:, 'cell_type'].isin(self.excluded_cell_types), 'cell_type'] = self.relabeled_labels
        self.relabeled=True

    def save_codex(self):
        if not self.relabeled:
            raise ValueError("Codex not relabeled yet")
        self.codex.to_csv(f"{self.base_path}/{self.project_name}/codex_quantcell_{self.project_name}.csv", index=False)

    def set_clf(self, clf):
        self.clf = clf

    def test_model_list(self, model_list, model_names=None):
        for n, model in enumerate(model_list):
            self.set_clf(model)

            time_start = time.time()
            y_test, y_pred, y_proba = self.cv_fit_pred(model_names[n] if model_names is not None else f"Model_{n}")
            time_end = time.time()
            time_elapsed = time_end - time_start

            os.makedirs(f"{self.base_path}/{self.project_name}/base_models", exist_ok=True)
            joblib.dump(y_test, f"{self.base_path}/{self.project_name}/base_models/{model_names[n]}_y_test.joblib")
            joblib.dump(y_pred, f"{self.base_path}/{self.project_name}/base_models/{model_names[n]}_y_pred.joblib")
            joblib.dump(y_proba, f"{self.base_path}/{self.project_name}/base_models/{model_names[n]}_y_proba.joblib")

            joblib.dump(time_elapsed, f"{self.base_path}/{self.project_name}/base_models/{model_names[n]}_time_elapsed.joblib")
    
        print("Testing complete")


def find_thresholds_at_varying_FDR(clf, X_test, y_test, min_FDR=0, max_FDR=0.25, number_points=100):
    FDR_cutoffs = np.linspace(min_FDR, max_FDR, number_points)
    thresholds = np.zeros((clf.n_classes_, number_points))
    pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    precision, recall, thresholds_array = interpolate_PRC(y_test, pred, probs, average=None)

    for _class in range(clf.n_classes_):
        precision_class = precision[_class, :]
        recall_class = recall[::-1]
        # find the threshold where precision is above the goal
        for i, FDR_cutoff in enumerate(FDR_cutoffs):
            precision_goal = 1 - FDR_cutoff
            threshold_index = np.where(precision_class >= precision_goal)[0]
            if len(threshold_index) == 0:
                thresholds[_class, i] = 1
            else:
                thresholds[_class, i] = thresholds_array[_class, threshold_index[0]]
    return FDR_cutoffs, thresholds



def find_thresholds(clf, X_test, y_test, FDR_cutoff=0.05):
    if FDR_cutoff == None:
        return [0] * clf.n_classes_

    precision_goal = 1 - FDR_cutoff

    thresholds=[]
    pred =  clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    precision, recall, thresholds_array = interpolate_PRC(y_test, pred, probs, average=None)

    for _class in range(clf.n_classes_):
        precision_class = precision[_class, :]
        recall_class = recall[::-1]
        # find the threshold where precision is above the goal
        threshold_index = np.where(precision_class >= precision_goal)[0]
        if len(threshold_index) == 0:
            thresholds.append(1)
        else:
            thresholds.append(thresholds_array[_class, threshold_index[0]])  # take the first threshold that meets the goal
    return thresholds


def apply_thresholds(clf, X, thresholds):
    labels = clf.predict(X)
    probs = clf.predict_proba(X)

    # create a mask if cells highest scoring cell type is above threshold
    mask = (probs >= thresholds)[np.arange(len(probs)), np.argmax(probs, axis=1)]

    # relabel cells that are below threshold as -1
    labels[~mask] = -1
    
    return labels


def interpolate_PRC(y_true, y_pred, proba, average='macro'):
    precision_list=[]
    recall_list=[]
    thresholds_list=[]
    
    for _class in sorted(np.unique(y_true)):
        mask=y_pred==_class
        if sum(mask)==0:
            precision_list.append([0])
            recall_list.append([0])
            thresholds_list.append([0])
        else:
            precision, recall, thresholds = precision_recall_curve(y_true[mask]==_class, proba[mask, _class])
            precision_list.append(precision)
            recall_list.append(recall)
            thresholds = np.concatenate(([0], thresholds))  # prepend 0 to thresholds to match precision and recall lengths
            thresholds_list.append(thresholds)

    precision_array=np.zeros((len(precision_list), 1000))
    thresholds_array = np.zeros((len(thresholds_list), 1000))

    recall_pts = np.linspace(0, 1, 1000)
    for _class in range(len(precision_list)):
        precision=precision_list[_class]
        recall=recall_list[_class]
        precision_array[_class, :] = np.interp(recall_pts, recall[::-1], precision[::-1])[::-1]
        thresholds_array[_class, :] = np.interp(recall_pts, recall[::-1], thresholds_list[_class][::-1])[::-1]
    if average=='macro':
        precision_return = np.mean(precision_array, axis=0)
        recall_return = recall_pts[::-1]
    elif average=='weighted':
        weights=[np.sum(y_true==_class)/len(y_true) for _class in sorted(np.unique(y_true))]
        precision_return = np.sum(precision_array*np.array(weights)[:, np.newaxis], axis=0)
        recall_return = recall_pts[::-1]
    elif average == None:
        precision_return = precision_array
        recall_return = recall_pts[::-1]
    return precision_return, recall_return, thresholds_array