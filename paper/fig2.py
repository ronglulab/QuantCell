#!/usr/bin/env python
"""Figure 2 v3 (2026-09-28, panels a-c of the published layout replaced by
 a) default vs hyperparameter-tuned macro AP for all ten base classifiers ( , corrected search),
 b) macro precision-recall curves of the ten TUNED classifiers (mean over the 10 CV folds);
panels d/e/f of the published figure become c/d/e unchanged (train size, feature inclusion, feature exclusion, M13 fix).
Panels a/b are classifier comparisons shared by both sets (RF base_models / base_models_tuned artifacts);
c/d/e come from the chosen set's model_selection. Usage: python fig2_render_v3.py rf|lgbm
Data/style from fig2_render.py (paper/05 cells 15-28) and fig_tuned_all10.py; nothing recomputed except the tuned PR curves."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import sys, os, json, joblib, numpy as np, pandas as pd, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import average_precision_score
warnings.filterwarnings('ignore')
SET = sys.argv[1] if len(sys.argv) > 1 else 'rf'; assert SET in ('rf', 'lgbm')
V = f'{QCP.OUT}'; SUF = '' if SET == 'rf' else '_lgbm'
RF_PROJ = f'{V}/retrain/QuantCellPaper'                  # panels a/b: shared classifier comparisons
project_path = f'{V}/retrain{SUF}/QuantCellPaper'        # panels c/d/e: the chosen set
OUT = f'{V}/retrain{SUF}/figures/fig2_v3_{SET}.png'
sys.path.insert(0, f'{QCP.CODE}'); from quantcell import interpolate_PRC
np.random.seed(42)

# ---- panel a: default vs tuned, all ten (fig_tuned_all10 data) ----
base_classifier_names = ['Nearest Neighbors','Multilayer Perceptron','Random Forest','Extra Trees',
    'Decision Tree','Gaussian Naive-Bayes','Ridge Classifier','Linear SVC','LightGBM','Logistic Regression']
dsum = json.load(open(f'{RF_PROJ}/base_models/base_models_summary.json'))
tsum = json.load(open(f'{RF_PROJ}/base_models_tuned/base_models_tuned_summary.json'))
dflt = {n: dsum[n]['macro_AP_mean'] for n in base_classifier_names}
tuned = {n: tsum[n]['macro_AP_mean'] for n in base_classifier_names}
order = sorted(base_classifier_names, key=lambda n: tuned[n])          # barh draws bottom-up: best on top
num_classes = 21; min_macro = 1/num_classes

# ---- panel b: PR curves of the TUNED models, same recipe as base_clf_performance_dict[..][2] ----
tuned_prc = {}
for n in base_classifier_names:
    yt = joblib.load(f'{RF_PROJ}/base_models_tuned/{n}_y_test.joblib'); yp = joblib.load(f'{RF_PROJ}/base_models_tuned/{n}_y_pred.joblib')
    pr = joblib.load(f'{RF_PROJ}/base_models_tuned/{n}_y_proba.joblib')
    curves = [interpolate_PRC(yt[f], yp[f], pr[f], average='macro') for f in range(10)]
    tuned_prc[n] = (np.mean([c[0] for c in curves], axis=0), np.mean([c[1] for c in curves], axis=0))  # precision, recall
    chk = np.mean([average_precision_score(yt[f], pr[f], average='macro') for f in range(10)])
    assert abs(chk - tuned[n]) < 1e-6, (n, chk, tuned[n])

# ---- panel c: train size (cell 21) ----
train_size_data = joblib.load(f'{project_path}/model_selection/train_size_results_5fold.joblib')
train_sizes = [0.05,0.1,0.25,0.5,0.75,1.0]
average_precisions = [np.mean(train_size_data[s]) for s in train_sizes]

# ---- panels d/e (cells 23/25, M13 fix) ----
location_order = ['Membrane','Cytoplasm','Nucleus','Cell','All']
stat_order = ['Min','Max','Std.Dev.','Median','Mean','All']
def build(path, skip=False):
    D = joblib.load(path); m={}; s={}
    for L in location_order:
        m[L]={}; s[L]={}
        for S in stat_order:
            if skip and L=='All' and S=='All': m[L][S]=None; s[L][S]=None; continue
            yt, pr = D[L][S][0], D[L][S][2]
            ap=[average_precision_score(yt[n],pr[n],average='macro') for n in yt]
            m[L][S]=np.mean(ap); s[L][S]=np.std(ap)
    return pd.DataFrame(m).T, pd.DataFrame(s).T
included_df, included_std_df = build(f'{project_path}/model_selection/stats_loc_dict_included.joblib')
included_text_df = included_df.copy()
for r in included_text_df.index:
    for c in included_text_df.columns:
        included_text_df.loc[r,c] = f'{included_df.loc[r,c]:.3f}\n±{included_std_df.loc[r,c]:.3f}'
baseline = included_df.loc['All','All']
exc_df, exc_std_df = build(f'{project_path}/model_selection/stats_loc_dict_excluded.joblib', skip=True)
excluded_df = baseline - exc_df; excluded_std_df = exc_std_df
excluded_text_df = excluded_df.copy()
for r in excluded_text_df.index:
    for c in excluded_text_df.columns:
        excluded_text_df.loc[r,c] = f'{excluded_df.loc[r,c]:.3f}\n±{excluded_std_df.loc[r,c]:.3f}'

# ---- style (cell 27) ----
default_fontsize, small_fontsize, big_fontsize = 24, 21, 30
plt.rcParams['font.size'] = default_fontsize
color_dict = {'Nearest Neighbors':'#1f77b4','Multilayer Perceptron':'mediumslateblue',
    'Random Forest':'#2ca02c','Extra Trees':'#d62728','Decision Tree':'darkgoldenrod',
    'Gaussian Naive-Bayes':'#8c564b','Ridge Classifier':'darkcyan','Linear SVC':'#e377c2',
    'LightGBM':'#7f7f7f','Logistic Regression':'#bcbd22'}
pretty = {'Nearest Neighbors':'Nearest neighbors','Multilayer Perceptron':'Multilayer perceptron','Random Forest':'Random forest',
    'Extra Trees':'Extra trees','Decision Tree':'Decision tree','Gaussian Naive-Bayes':'Gaussian naive-Bayes',
    'Ridge Classifier':'Ridge classifier','Linear SVC':'Linear SVC','LightGBM':'LightGBM','Logistic Regression':'Logistic regression'}
max_val, min_val = np.nanmax(excluded_df.values), np.nanmin(excluded_df.values)
strength = 0.5; pivot = int(np.abs(min_val)/np.abs(max_val-min_val)*256)
colors = np.zeros((256,3))
colors[:pivot,0]=1; colors[pivot:,0]=np.linspace(1,1-strength,256-pivot)
colors[:pivot,1]=np.linspace(1-strength,1,pivot); colors[pivot:,1]=np.linspace(1,1-strength,256-pivot)
colors[:pivot,2]=np.linspace(1-strength,1,pivot); colors[pivot:,2]=1
custom_coolwarm_cmap = LinearSegmentedColormap.from_list('Custom_Coolwarm', colors)

# ---- layout: row 1 = a | b | c ; row 2 = d | e ----
fig = plt.figure(figsize=(21,15)); axd = {}
pos = {'A':[0.15,0.60,0.20,0.36], 'B':[0.43,0.60,0.24,0.36], 'C':[0.74,0.60,0.24,0.36],
       'D':[0.05,0.05,0.30,0.40], 'X':[0.37,0.05,0.02,0.40], 'E':[0.58,0.05,0.30,0.40], 'Y':[0.90,0.05,0.02,0.40]}
for k,p in pos.items(): axd[k] = fig.add_axes(p)

# a) paired horizontal bars: light = default, full = tuned (Fig 2b/2c conventions)
y = np.arange(len(order)); h = 0.38
for i,n in enumerate(order):
    c = color_dict[n]
    axd['A'].barh(y[i]-h/2, dflt[n], height=h, color='white', edgecolor='black')
    axd['A'].barh(y[i]-h/2, dflt[n], height=h, color=c, alpha=0.45, edgecolor='black')
    axd['A'].barh(y[i]+h/2, tuned[n], height=h, color='white', edgecolor='black')
    axd['A'].barh(y[i]+h/2, tuned[n], height=h, color=c, alpha=1.0, edgecolor='black')
axd['A'].axvline(min_macro, color='black', linestyle='--', linewidth=1.5)
axd['A'].set_yticks(y, [pretty[n] for n in order], fontsize=default_fontsize)
axd['A'].set_xlim(0,1); axd['A'].set_xticks([0,0.2,0.4,0.6,0.8,1.0])
axd['A'].set_xlabel('Average precision', fontsize=default_fontsize)
axd['A'].tick_params(axis='both', which='major', labelsize=default_fontsize)
axd['A'].spines[['top','right']].set_visible(False)
axd['A'].legend(handles=[mpatches.Patch(facecolor='lightgray',alpha=0.45,edgecolor='black',label='Default'),
                          mpatches.Patch(facecolor='dimgray',edgecolor='black',label='Hyperparameter tuned')],
                 fontsize=small_fontsize, frameon=False, loc='upper center', bbox_to_anchor=(0.5,-0.16), ncol=2, handletextpad=0.5, columnspacing=1.2)

# b) tuned PR curves (old 2a styling)
for n in base_classifier_names:
    p, r = tuned_prc[n]
    axd['B'].plot(r, p, color=color_dict[n], label=n, linewidth=4, alpha=0.6)
axd['B'].set_xlim(0,1); axd['B'].set_ylim(0.45,1)
axd['B'].spines[['top']].set_visible(False)   # right edge drawn as a solid spine like left/bottom (dashed recall=1 line removed)
axd['B'].set_xlabel('Recall', fontsize=default_fontsize); axd['B'].set_ylabel('Precision', fontsize=default_fontsize)
axd['B'].tick_params(axis='both', which='major', labelsize=default_fontsize)

# c) train size (old 2d)
axd['C'].bar(range(0,6), average_precisions, width=0.7, color='gray', edgecolor='black', alpha=0.6)
for i,size in enumerate(train_sizes):
    axd['C'].scatter([i-(np.random.rand(5)-0.5)/2], train_size_data[size], color='black', zorder=5, s=40)
axd['C'].set_ylim([0.85,1]); axd['C'].set_yticks([0.85,0.9,0.95,1])
axd['C'].spines[['top','right']].set_visible(False); axd['C'].set_ylabel('Average precision')
axd['C'].set_xticks(range(0,6), [f'{s:.0%}' for s in train_sizes], fontsize=default_fontsize)
axd['C'].set_xlabel('Proportion of training data', fontsize=default_fontsize)

# d) inclusion (old 2e)
sns.heatmap(included_df.loc[location_order,stat_order], annot=included_text_df.loc[location_order,stat_order],
            fmt='', vmin=0, vmax=1, cmap='Reds', annot_kws={"size":small_fontsize}, ax=axd['D'], cbar_ax=axd['X'])
axd['D'].tick_params(labelsize=default_fontsize)
axd['D'].set_xticks(axd['D'].get_xticks(), axd['D'].get_xticklabels(), rotation=90)
axd['D'].set_yticks(axd['D'].get_yticks(), axd['D'].get_yticklabels(), rotation=0)
axd['X'].set_ylabel('Average precision', rotation=270, fontsize=default_fontsize, labelpad=30)
axd['X'].set_yticks([0,0.2,0.4,0.6,0.8,1], ['0.0','0.2','0.4','0.6','0.8','1.0'], fontsize=default_fontsize)

# e) exclusion (old 2f, M13-corrected SDs)
sns.heatmap(excluded_df.loc[location_order,stat_order], vmin=min_val, vmax=max_val, cmap=custom_coolwarm_cmap,
            annot=excluded_text_df.loc[location_order,stat_order], fmt='', annot_kws={"size":small_fontsize},
            ax=axd['E'], cbar_ax=axd['Y'])
axd['E'].tick_params(labelsize=default_fontsize)
axd['E'].set_xticks(axd['E'].get_xticks(), axd['E'].get_xticklabels(), rotation=90)
axd['E'].set_yticks(axd['E'].get_yticks(), axd['E'].get_yticklabels(), rotation=0)
axd['Y'].set_ylabel('Δ Average precision', rotation=270, fontsize=default_fontsize, labelpad=30)
# the fixed 0.015 tick lay above vmax and set_yticks stretched the colour-bar axis past the coloured range
# (bar looked cut off) -> ticks only inside [min_val, max_val] and the axis limits pinned to that range
_t=[v for v in (-0.005,0,0.005,0.01,0.015) if min_val-1e-12<=v<=max_val+1e-12]
axd['Y'].set_yticks(_t, [f'{v:.3f}' for v in _t], fontsize=default_fontsize); axd['Y'].set_ylim(min_val, max_val)

# panel letters in figure coordinates
for k,l,dx in [('A','a',0.11),('B','b',0.06),('C','c',0.06),('D','d',0.03),('E','e',0.03)]:
    x0,y0,w,hh = pos[k]; fig.text(x0-dx, y0+hh+0.005, l.upper(), fontsize=big_fontsize, fontweight='bold')

fig.savefig(OUT, dpi=100, bbox_inches='tight'); print(f"written -> {OUT}")
print("panel a (default -> tuned):"); [print(f"  {n:<22} {dflt[n]:.4f} -> {tuned[n]:.4f}") for n in order[::-1]]
print(f"panel c train-size AP: {[round(v,4) for v in average_precisions]}")
print(f"panel d All/All baseline AP = {baseline:.4f}")
