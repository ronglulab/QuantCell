import pandas as pd
import numpy as np
import glob as gb
import os
import json
import re
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class codex_project:
    def __init__(self):
        self.codex = None
        self.marker_list = None
        self.project_name = None
        self.annotation_strategy = None

        self._marker_combos = None
        self._data_folders = None
        self._sectioned = False
        self._sample_named = False
    
    def save_csv(self, save_path: str, force: bool = False) -> None:
        if self.codex is None:
            raise ValueError('No codex found')
        
        verified = self.verify_codex(verbose=False)
        if not verified and not force:
            raise ValueError('Inconsistiencies found in codex, verify_codex() to verify construction or use force=True to save anyway')

        try:
            self.codex.to_csv(save_path, index=False)
        except OSError as e:
            raise ValueError(f'Could not write codex to {save_path}: {e}') from e

    def read_csv(self, csv_path: str, project_name: str, force: bool = False) -> None:
        try:
            self.codex = pd.read_csv(csv_path)
        except FileNotFoundError as e:
            raise ValueError(f'Codex file not found: {csv_path}') from e
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            raise ValueError(f'Could not parse {csv_path} as a codex CSV: {e}') from e

        # Deliberately outside the try above: a failed verification is a real,
        # specific result and must not be re-reported as a file/path problem.
        verified = self.verify_codex(verbose=False)
        if not verified and not force:
            self.codex = None
            raise ValueError('Inconsistiencies found in codex, verify_codex() to verify construction or use force=True to save anyway')

        try:
            self.annotation_strategy = self.codex.loc[0, 'annotation_strategy']
        except KeyError as e:
            self.codex = None
            raise ValueError(f"{csv_path} has no 'annotation_strategy' column; not a codex CSV") from e
        if 'section' in self.codex.columns:
            self._sectioned = True
        if 'sample' in self.codex.columns:
            self._sample_named = True
    
    def read_annotation_strategy(self, annotation_strategy_path: str) -> None:
        # expects .json format from python script
        try:
            self._marker_combos = json.loads(open(annotation_strategy_path).read())
        except FileNotFoundError as e:
            raise ValueError(f'Annotation strategy not found: {annotation_strategy_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Annotation strategy {annotation_strategy_path} is not valid JSON: {e}') from e
        _, self.annotation_strategy = os.path.split(annotation_strategy_path)
        self.annotation_strategy = self.annotation_strategy.split('.')[0]
        self._get_marker_list()

    def _get_marker_list(self) -> None:
        if self._marker_combos is None:
            raise ValueError('No annotation strategy found')
        else:
            self.marker_list = []
            for cell_type in self._marker_combos:
                for marker in self._marker_combos[cell_type]:
                    if len(marker) > 0: # not empty string
                        self.marker_list.append(marker[:-1]) # remove the '+' or '-' from the end
        self.marker_list = sorted(list(set(self.marker_list)), key=lambda x: x.lower()) # sort alphabetically and only unique values

    def convert_annotation_strategy_to_df(self) -> pd.DataFrame:
        if self._marker_combos is None:
            raise ValueError('No annotation strategy found')
        else:
            marker_df = pd.DataFrame(columns=self.marker_list)
            for cell_type in self._marker_combos:
                temp = []
                for marker in self.marker_list:
                    if marker + '+' in self._marker_combos[cell_type]:
                        temp.append(marker + '+')
                    elif marker + '-' in self._marker_combos[cell_type]:
                        temp.append(marker + '-')
                    else:
                        temp.append('')
                marker_df.loc[cell_type] = temp
            return marker_df

    def verify_codex(self, verbose: bool = True) -> bool:
        verified=True
        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
        def check(condition):
            nonlocal verified
            if condition:
                vprint('Passed')
            else:
                verified=False
                vprint('Failed')

        vprint('~~~Verifying codex~~~')

        if self.codex is None:
            raise ValueError('No codex found')

        # Required columns are checked first: the strategy check below dereferences
        # one of them, so the original order raised KeyError on exactly the codices
        # this function exists to reject.
        vprint('Checking for required columns...', end='')
        required_columns = ['cell_type', 'marker_string', 'sample', 'x', 'y', 'original_folder', 'annotation_strategy']
        check(all(col in self.codex.columns for col in required_columns))

        vprint('Checking for consistent annotation strategy...', end='')
        check('annotation_strategy' in self.codex.columns
              and self.codex.loc[:, 'annotation_strategy'].nunique() == 1)

        vprint('Checking for missing values...', end='')
        check(self.codex.isna().sum().sum() == 0)



        return verified

    def verify_marker_annotations(self, verbose: bool = True) -> bool:
        verified=True
        if self._data_folders is None:
            raise ValueError('No data folders found')

        if self._marker_combos is None:
            raise ValueError('No annotation strategy found')

        for folder in self._data_folders:
            files = gb.glob(folder + '/*.csv')
            if len(files) == 0:
                print('No .csv files found in ' + folder)
                continue
            markers_present = []
            for f in files:
                _, f = os.path.split(f)
                if " (" in f:
                    marker = f.split(' (')[0]
                else:
                    marker = f.split('.')[0]
                markers_present.append(marker)
            
            missing_markers = [m for m in self.marker_list if m not in markers_present] # markers which are in the annotation strategy but not in the folder
            unused_markers = [m for m in markers_present if m not in self.marker_list] # markers which are in the folder but not in the annotation strategy

            if verbose:
                print('~~~Verifying markers in ' + folder + '~~~')
                print('Missing markers:', missing_markers)
                print('Unused markers:', unused_markers)
            
            if len(missing_markers) > 0:
                verified=False
            print()
    
        return verified

    def initialize(self, data_path: str, folders: list, annotation_strategy_path: str, project_name: str) -> None:
        self._data_folders = [data_path + '/' + x for x in folders]
        self.project_name = project_name
        self.read_annotation_strategy(annotation_strategy_path)
        multi_compartment = False

        codex_list = []
        for folder in folders:
            template_file = gb.glob(data_path+'/'+folder+'/*')[0] # use first file in folder as template
            #template_file = gb.glob(data_path+'/'+folder+'/' + self.marker_list[0]+'*')[0] # use first file in folder as template
            codex = pd.read_csv(template_file)

            # drop marker data because it will be added back in later w/ correct format name
            for col in codex.columns:
                if ':' in col and col.split(':')[0] not in ['Nucleus', 'Cell', 'Membrane', 'Cytoplasm']:
                    codex = codex.drop(col, axis=1)

            codex = codex.assign(Class = '')
            codex = self._add_markers(codex, data_path+'/'+folder)
            codex.loc[:, 'original_folder'] = folder
            codex.loc[:, 'annotation_strategy'] = self.annotation_strategy
            codex_list.append(codex)

        self.codex = pd.concat(codex_list, axis=0, ignore_index=True)
        self.codex = self.codex.drop('Name', axis=1)
        self.codex = self.codex.rename(columns={'Centroid X µm': 'x', 'Centroid Y µm': 'y', 'Class' : 'marker_string'})
        self.codex.reset_index(drop=True, inplace=True)

    @staticmethod
    def _norm_marker(s) -> str:
        """Normalize a marker/channel name for comparison: uppercase, alphanumeric only."""
        return re.sub(r'[^A-Z0-9]', '', str(s).upper())

    def _resolve_marker_prefix(self, columns, marker, marker_name, file):
        """Return the column prefix holding this file's antigen measurements.

        QuPath channel names do not reliably match either the filename or the
        annotation class. They carry fluorophore suffixes ('CD117 (Cy5)'), clone
        numbers ('CD90.2'), spacing ('MHC II') or separator variants ('F4_80' vs
        'F4/80'). Some exports also contain a second, unrelated antigen block --
        'CD41.csv' carries a 'dapi1' block ahead of its own -- which must not be
        adopted as this marker's features.

        Exact match against the annotation class wins. Otherwise a prefix is a
        candidate when it is normalization-compatible with the class OR with the
        filename marker, and exactly one candidate must remain.
        """
        morphology = {'Nucleus', 'Cell', 'Cytoplasm', 'Membrane', 'ROI', 'Image'}
        prefixes = []
        for col in columns:
            if ':' in col:
                p = col.split(':')[0].strip()
                if p not in morphology and p not in prefixes and self._norm_marker(p):
                    prefixes.append(p)

        if not prefixes:
            # Export formats such as 'ROI: 2.00 um per pixel: CD3: Mean' carry no
            # leading antigen prefix. They yield no per-compartment features under
            # any naming (the spatial filter drops them), so the file contributes
            # its marker calls only. Not an error.
            return None

        n_cls = self._norm_marker(marker_name)
        n_mark = self._norm_marker(marker)
        if not n_cls and not n_mark:
            raise ValueError(f"Cannot resolve marker channel in {file}: empty class and marker name")

        for p in prefixes:
            if self._norm_marker(p) == n_cls:
                return p

        candidates = []
        for p in prefixes:
            n_p = self._norm_marker(p)
            if ((n_cls and (n_p.startswith(n_cls) or n_cls.startswith(n_p))) or
                    (n_mark and (n_p.startswith(n_mark) or n_mark.startswith(n_p)))):
                candidates.append(p)

        if len(candidates) == 1:
            print(f"Warning: {os.path.basename(file)} - using channel '{candidates[0]}' for "
                  f"marker '{marker}' (annotation class is '{marker_name}')")
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous marker channel for '{marker}' in {file}: {candidates} "
                             f"are all compatible with class '{marker_name}'")
        raise ValueError(f"No marker channel found for '{marker}' in {file}: class "
                         f"'{marker_name}', available prefixes {prefixes}")

    def _add_markers(self, codex: pd.DataFrame, folder_path: str) -> pd.DataFrame:
        csv_files = gb.glob(folder_path + '/*.csv')
        spatial_words = ['Area', 'Volume', 'Centroid', 'µm', 'Diameter', 'Circularity', 'Detection', 'Solidity' ,'Length']
        for marker in self.marker_list:
            found=False
            for file in csv_files:
                if marker +'.csv' in file or marker + ' (' in file:
                    found=True
                    df = pd.read_csv(file)
                    break
            if found:
                a = list(df.Class.unique())
                a = {x for x in a if x==x}
                if "Other" in a:
                    a.remove("Other")

                if 'other' in a:
                    a.remove("other")
                
                if len(a) > 1:
                    raise ValueError('Multiple classes found in ' + file)
                marker_name = a.pop()

                coords = codex.loc[:, ['Centroid X µm', 'Centroid Y µm']]
                if 'Centroid X µm' not in df.columns or 'Centroid Y µm' not in df.columns:
                    print(f'Warning: Missing centroid data for {marker} in {folder_path}')
                    if df.shape[0] == codex.shape[0]:
                        print('~ Resolved: Row shapes match, reusing centroid data from other files')
                    else:
                        raise ValueError('Fatal Error: Mismatched shape and no centroid data for ' + marker + ' in ' + folder_path)
                else:
                    if not coords.equals(df.loc[:, ['Centroid X µm', 'Centroid Y µm']]):
                        raise ValueError('Fatal Error: Centroid mismatch for ' + marker + ' in ' + folder_path)
        
                # Change the labels so its either marker_name+, marker_name?, or marker_name-
                df.loc[:, 'Name'] = df.loc[:, 'Name'].str.upper()
                df.loc[df.Name == marker_name.upper(), "Class"] = marker + "+"
                df.loc[df.Name == "PATHCELLOBJECT", "Class"] = marker + "?"
                df.loc[df.Name == "OTHER", "Class"] = marker + "-"

                marker_prefix = self._resolve_marker_prefix(df.columns, marker, marker_name, file)

                # Add new label to current labels
                codex[marker] = df['Class']
                for col in df.columns:
                    is_spatial=False
                    if ':' in col:
                        # Only this marker's own antigen block is renamed onto the
                        # filename marker. Other blocks (a stray 'dapi1', QuPath
                        # morphology) are left alone rather than silently adopted.
                        if col.split(':')[0].strip() != marker_prefix:
                            continue
                        new_cname=marker + ':' + ':'.join(col.split(':')[1:])
                    else:
                        new_cname=col.replace(marker_name, marker) # replaces marker_name in file w/ name of file
                    for x in spatial_words:
                        if x in new_cname:
                            is_spatial=True

                    if new_cname not in codex.columns and not is_spatial:
                        codex[new_cname] = df.loc[:, col]

                codex['Class'] = codex['Class'] + ":" + df["Class"]
            else:
                print('Warning: Missing marker data for ' + marker + ' in ' + folder_path)
        return codex

    def annotate(self, force:bool = False) -> None:
        if self.codex is None:
            raise ValueError('No codex found')

        if self._marker_combos is None:
            raise ValueError('No annotation strategy found')


        if self._sample_named == False and not force:
            raise ValueError('No sample labels found. Use set_sample_labels() to set sample labels or use force=True to annotate all cells together')

        if self._sample_named == False:
            self.codex.loc[:, 'cell_type'] = self.rule_based_annotation(self.codex.loc[:, 'marker_string'], self._marker_combos)
        else:
            for sample in self.codex.loc[:, 'sample'].unique():
                mask = self.codex.loc[:, 'sample'] == sample
                self.codex.loc[mask, 'cell_type'] = self.rule_based_annotation(self.codex.loc[mask, 'marker_string'], self._marker_combos)


    def set_sample_labels(self, labels: dict) -> None:
        if self.codex is None:
            raise ValueError('No codex found')

        for sample in labels:
            self.codex.loc[self.codex['original_folder'] == sample, 'sample'] = labels[sample]
            self._sample_named = True


    
    def drop_missing(self, max_missing_per_row: int = 2, max_missing_per_col: int = 100, drop_axis=0) -> None:
        if self.codex is None:
            raise ValueError('No codex found')

        print("OLD DIMENSIONS:", self.codex.shape)
        self.codex = self.codex.loc[:, self.codex.isna().sum() < max_missing_per_col] # drop columns with more than X NAs
        self.codex = self.codex.loc[self.codex.isna().T.sum() < max_missing_per_row, :] # drop rows with more than X NAs
        self.codex = self.codex.dropna(axis=drop_axis) # 0 to drop cells (rows), 1 to drop features (columns), dropping cells should be used when many columns have a small number of NAs
        # dropping columns should be used when a small number of columns have many NAs
        self.codex = self.codex.reset_index(drop=True)
        print("NEW DIMENSIONS:", self.codex.shape)


    def section_samples(self, eps) -> None:
        if self.codex is None:
            raise ValueError('No codex found')

        clusterer = DBSCAN(eps = eps)
        for collection in self.codex.loc[:, 'original_folder'].unique():
            mask = self.codex.loc[:, 'original_folder'] == collection
            section_labels = clusterer.fit_predict(self.codex.loc[mask, ['x', 'y']])
            self.codex.loc[mask, 'section'] = [f'{collection}_{str(x)}' for x in section_labels]
        self._sectioned = True
        
    def override_sectioning(self) -> None:
        if self.codex is None:
            raise ValueError('No codex found')

        self.codex.loc[:, 'section'] = self.codex.loc[:, 'original_folder']+'_0'
        self._sectioned = True

    
    def visualize_sectioning(self) -> None:
        if not self._sectioned:
            raise ValueError('No sectioning found')

        for collection in self.codex.loc[:, 'original_folder'].unique():
            mask = self.codex.loc[:, 'original_folder'] == collection
            x = self.codex.loc[mask, 'x']
            y = self.codex.loc[mask, 'y']
            section_labels = self.codex.loc[mask, 'section'].str.split('_').str[-1].astype(int)
            fig, ax = plt.subplots()
            plt.scatter(x, y, s=1, c=section_labels)
            plt.title(collection)

    def verify_sectioning(self, verbose=True) -> bool:
        if not self._sectioned:
            raise ValueError('No sectioning found')
        
        verified=True
        if verbose:
            print('~~~Verifying sectioning~~~')

        for collection in self.codex.loc[:, 'original_folder'].unique():
            if -1 in self.codex.loc[self.codex.loc[:, 'original_folder'] == collection, 'section'].str.split('_').str[-1].astype(int).unique():
                verified=False
                if verbose:
                    print('Failed: Unassigned cells in ' + collection)
            else:
                if verbose:
                    print('Passed: All cells assigned in ' + collection)

        return verified




    """
    marker_strings: a Series of strings in the form of marker names and +/-/? status separated by colons
        e.g. "marker+:marker2-:marker3?"

    marker_combos: a dict of cell types and their corresponding marker combinations
        e.g. {'CD8 T Cells' : ['CD4-', 'CD3+', 'CD8+'], 'CD4 T Cells' : ['CD4+', 'CD3+', 'CD8-']}

    """
    def rule_based_annotation(self, marker_strings: pd.Series, marker_combos: dict) -> list:
        orig_labs = []
        for _str in marker_strings:
            temp_list=[]
            marker_pos_neg_statuses = _str.split(':')

            for ct in marker_combos:
                if all(m in marker_pos_neg_statuses for m in marker_combos[ct]):
                    temp_list.append(ct)

            if len(temp_list) == 0:
                temp_list.append('Other')
            orig_labs.append(temp_list)
        new_labs = self._resolve_labels(orig_labs)
        return new_labs


    conventional_annotation = rule_based_annotation   # backward-compatible alias
    def _resolve_labels(self, orig_labs: list) -> list:
        return [x[0] if len(x) == 1 else 'Other' for x in orig_labs]


"""
Function to replace columns in main codex with columns from another codex based on matching centroids

Useful when the samples need to be resegmented and you need to match up the new segmentations with the old ones
"""
def replace_marker_token(marker_string, marker, new_call):
    """Swap the ':<marker><+|-|?>:' token inside a marker_string.

    A marker_string is a ':'-delimited run of '<marker><call>' tokens, e.g.
    ':B220-:CD105-:CD41+:CD45-:'. Matching whole tokens means 'CD4' cannot hit
    'CD41'/'CD45', and a pre-existing '?' elsewhere in the string cannot shift the
    result -- both of which defeated the previous substring/split('?') approach.

    Returns (new_string, matched).
    """
    if not isinstance(marker_string, str):
        return marker_string, False
    parts = marker_string.split(':')
    matched = False
    for i, part in enumerate(parts):
        if part[:-1] == marker and part[-1:] in ('+', '-', '?'):
            parts[i] = new_call
            matched = True
    return ':'.join(parts), matched


def replace_marker_match_centroids(main_codex, marker_codex, marker):
    """Replace one marker's data in `main_codex` with `marker_codex`'s, matched on
    cell centroid. Used when samples are resegmented and the new segmentation has to
    be matched back onto the existing codex.

    All three representations of the marker are replaced together:
      1. the quantitative '<marker>: <compartment>: <stat>' columns,
      2. the binary call column '<marker>',
      3. the corresponding token inside 'marker_string'.

    (3) is not optional. annotate() derives every cell_type from marker_string, so
    replacing only (1) and (2) leaves the codex internally inconsistent -- the
    quantitative data and binary calls describe the new segmentation while every
    cell_type still describes the old one.

    Cells in `main_codex` with no centroid match get NaN in the replaced columns and
    are expected to be removed by drop_missing(). Inputs are not modified.
    """
    main_codex = main_codex.copy()
    marker_codex = marker_codex.copy()

    # Exact-prefix match: the binary call column plus this marker's measurements.
    # A substring test would make marker='CD4' also select CD41 and CD45.
    cols_to_replace = [c for c in main_codex.columns
                       if c == marker or c.split(':')[0].strip() == marker]
    if marker not in cols_to_replace:
        raise ValueError(f"No binary call column '{marker}' in main_codex")
    missing = [c for c in cols_to_replace if c not in marker_codex.columns]
    if missing:
        raise ValueError(f"marker_codex is missing columns for '{marker}': {missing[:5]}")

    main_key = (main_codex['x'].astype(str) + '_' + main_codex['y'].astype(str)).values
    mk_key = (marker_codex['x'].astype(str) + '_' + marker_codex['y'].astype(str)).values
    if pd.Index(main_key).has_duplicates:
        raise ValueError('main_codex has duplicate centroids; cannot match unambiguously')
    if pd.Index(mk_key).has_duplicates:
        raise ValueError('marker_codex has duplicate centroids; cannot match unambiguously')

    keep = pd.Index(mk_key).isin(set(main_key))
    marker_codex = marker_codex.loc[keep]
    marker_codex.index = pd.Index(mk_key[keep])

    original_index = main_codex.index
    main_codex.index = pd.Index(main_key)
    target = marker_codex.index

    for col in cols_to_replace:
        main_codex[col] = np.nan
        main_codex.loc[target, col] = marker_codex[col].values

    # rewrite the marker's token in marker_string for the matched cells
    old_strings = main_codex.loc[target, 'marker_string'].values
    new_calls = marker_codex[marker].values
    rebuilt, unmatched = [], 0
    for old, call in zip(old_strings, new_calls):
        text, ok = replace_marker_token(old, marker, call)
        rebuilt.append(text)
        if not ok:
            unmatched += 1
    if unmatched:
        raise ValueError(f"marker_string held no '{marker}' token for {unmatched} of "
                         f"{len(rebuilt)} matched cells; refusing to leave marker_string "
                         f"inconsistent with the replaced calls")
    main_codex.loc[target, 'marker_string'] = rebuilt

    main_codex.index = original_index
    return main_codex



def read_marker_combos(annotations_path):
    # expects .json format from python script
    try:
        marker_combos = json.loads(open(annotations_path).read())
    except FileNotFoundError as e:
        raise ValueError(f'Annotation strategy not found: {annotations_path}') from e
    except json.JSONDecodeError as e:
        raise ValueError(f'Annotation strategy {annotations_path} is not valid JSON: {e}') from e
    return marker_combos


def get_marker_list(marker_combos):
    marker_list = []
    for cell_type in marker_combos:
        for marker in marker_combos[cell_type]:
            if len(marker) > 0: # not empty string
                marker_list.append(marker[:-1]) # remove the '+' or '-' from the end
    marker_list = sorted(list(set(marker_list)), key=lambda x: x.lower()) # sort alphabetically and only unique values
    return marker_list