#!/usr/bin/env python
""" Figure 3 v2 a confusion matrix (unchanged), b = old 3c frequency agreement, c-e = the dense BM FDR sweep
(fdr_sweep_bm_v2.json: realised FDR vs target, fraction of all cells annotated with the rule-based baseline, macro precision + recall).
Old 3b (stacked fraction bars) dropped. 24 pt text, 30 pt letters. Data loaders copied from fig3_render_lgbm.py."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import joblib, numpy as np, pandas as pd, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix, precision_score
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

project_path = f'{QCP.OUT}/retrain_lgbm/QuantCellPaper'
project_name = 'QuantCellPaper'
OUT = f'{QCP.OUT}/retrain_lgbm/figures/fig3_v2_lgbm.png'

cell_order = ['KLS','MEP','CMP','GMP','CLP','CFU-E','MkP','Erythroblasts','Erythrocytes',
    'Megakaryocytes','B Cells','CD4 T Cells','CD8 T Cells','cDC','Double Negative T Cells',
    'Monocytes','Neutrophils','Arterial ECs','Capillary ECs','Other BM ECs','Lepr+ Perivascular Cells']

def plot_confusion_matrix(conf_matrix, name, normalization='row',
                          cbar_space=np.linspace(0,1,100), cbar_ticks=None, **kw):
    if normalization=='row':   conf_matrix = conf_matrix.divide(conf_matrix.sum(axis=1), axis=0)
    elif normalization=='col': conf_matrix = conf_matrix.divide(conf_matrix.sum(axis=0), axis=1)
    elif normalization=='all': conf_matrix = conf_matrix/conf_matrix.sum().sum()
    ax = sns.heatmap(conf_matrix, linecolor='grey', linewidth=0.01,
        xticklabels=cell_order, yticklabels=cell_order,
        cmap=sns.color_palette("flare", as_cmap=True),
        cbar_kws={'format': FuncFormatter(lambda x,_: f'{x:.0f}%'), 'boundaries': cbar_space}, **kw)
    cb = ax.collections[0].colorbar
    if cbar_ticks is None:
        cb.set_ticks([0,0.2,0.4,0.6,0.8,1.0]); cb.set_ticklabels(['0%','20%','40%','60%','80%','100%'])
    else:
        cb.set_ticks(cbar_ticks); cb.set_ticklabels([f'{int(t*100)}%' for t in cbar_ticks])
    ax.tick_params(length=6); ax.set_xticklabels(cell_order, rotation=90)
    return ax

# ---- cell 32 ----
encoder = joblib.load(f'{project_path}/label_encoder.joblib')
quantcell_labels = pd.read_csv(f'{project_path}/quantcell_labels.csv').iloc[:,0]
quantcell_true   = pd.read_csv(f'{project_path}/quantcell_true_labels.csv', index_col=0).loc[:,'cell_type']
RENAME = {'Arteries':'Arterial ECs','Capillaries/Sinusoids':'Capillary ECs',
          'Endothelial Niche':'Other BM ECs','Perivascular Niche':'Lepr+ Perivascular Cells',
          'Preplatelets':'Megakaryocytes'}
print("labels present BEFORE the hardcoded rename:")
print("  true  :", sorted(set(quantcell_true) & set(RENAME)))
print("  pred  :", sorted(set(quantcell_labels) & set(RENAME)))
for old,new in RENAME.items():
    quantcell_true[quantcell_true==old]   = new
    quantcell_labels[quantcell_labels==old] = new

cm = confusion_matrix(quantcell_true, quantcell_labels, labels=encoder.classes_)
confusion_matrix_quantcell = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
confusion_matrix_quantcell = confusion_matrix_quantcell.loc[cell_order, cell_order]
other_mask = (quantcell_labels != 'Other').values
scores = precision_score(quantcell_true[other_mask], quantcell_labels[other_mask], average=None)
print(f"\nMinimum cell-type precision score for QuantCell: {min(scores)}")

diag = np.diag(confusion_matrix_quantcell.divide(confusion_matrix_quantcell.sum(axis=0), axis=1).values)
print(f"3a diagonal (=precision): min {np.nanmin(diag):.4f}  mean {np.nanmean(diag):.4f}")
overall = np.trace(confusion_matrix_quantcell.values)/confusion_matrix_quantcell.values.sum()
print(f"overall agreement (trace/total) = {overall:.4f}   <- paper claims 96.5%")

# ---- cell 34 ----
fraction_annotated = joblib.load(f'{project_path}/fraction_annotated.joblib')
print(f"\n3b: Conventional {fraction_annotated['Conventional']:.4f}"
      f"  +FDR<5% {fraction_annotated['FDR < 5%']:.4f}"
      f"  = {fraction_annotated['Conventional']+fraction_annotated['FDR < 5%']:.4f}")
print(f"    percent increase = {fraction_annotated['FDR < 5%']/fraction_annotated['Conventional']*100:.2f}%")

# ---- cell 37 ----
original_codex  = pd.read_csv(f'{project_path}/codex_rule_based_{project_name}.csv', usecols=['cell_type'])
relabeled_codex = pd.read_csv(f'{project_path}/codex_quantcell_{project_name}.csv',   usecols=['cell_type'])
original_frequencies  = original_codex['cell_type'].value_counts()/len(original_codex)
relabeled_frequencies = relabeled_codex['cell_type'].value_counts()/len(relabeled_codex)

# ---- layout: top row a (confusion) + b (frequency agreement); bottom row c, d, e (FDR sweep) ----
import json
FS, LETTER = 24, 30; plt.rcParams['font.size'] = FS
sw = json.load(open(f'{QCP.OUT}/retrain_lgbm/figures/fdr_sweep_bm_v2.json'))
# panel d uses the full-data recipe (fit 80 % of labelled, thresholds on 20 %, applied to all unlabelled) = relabel_unannotated
swf = json.load(open(f'{QCP.OUT}/retrain_lgbm/figures/fdr_sweep_bm_full.json')); assert abs(swf['rule_based_fraction']-sw['rule_based_fraction'])<1e-9
pcf = np.array([c['cutoff']*100 for c in swf['cutoffs']]); annot_full = np.array([c['annotated_fraction_all_cells'] for c in swf['cutoffs']])
SC = sw['cutoffs']; pc = np.array([c['cutoff']*100 for c in SC]); g = lambda k: np.array([c[k] if c[k] is not None else np.nan for c in SC]); XMAX = 10.0
fig = plt.figure(figsize=(19, 19)); axd = {}   # ~19 in wide so 24 pt text is the same relative size as the old 17-in Fig 3
axd['A'] = fig.add_axes([0.07, 0.65, 0.33, 0.33]); axd['X'] = fig.add_axes([0.41, 0.65, 0.015, 0.33])   # square canvas
axd['B'] = fig.add_axes([0.60, 0.705, 0.275, 0.275])   # all four 0.275 square
axd['C'] = fig.add_axes([0.60, 0.36, 0.275, 0.275]); axd['D'] = fig.add_axes([-0.114, 0.03, 0.275, 0.275]); axd['E'] = fig.add_axes([0.256, 0.03, 0.275, 0.275])   # b/c closer to a; d y-label flush with a's y-label # layout 4 b over c on the right spanning a's height; d, e bottom row left-aligned # c square; letter c under letter a (fig x -0.049), e right edge == b right edge

plot_confusion_matrix(confusion_matrix_quantcell, 'QuantCell', normalization='col', ax=axd['A'], cbar_ax=axd['X'])
axd['A'].set_xticklabels(axd['A'].get_xticklabels(), fontsize=FS, rotation=90)
axd['A'].set_yticklabels(axd['A'].get_yticklabels(), fontsize=FS, rotation=0)
axd['A'].set_ylabel('Rule-based annotation', fontsize=FS); axd['A'].set_xlabel('QuantCell', fontsize=FS); axd['X'].tick_params(labelsize=FS)

# panel b = QuantCell-only frequencies (rule-unannotated cells labelled by QuantCell at 5 % FDR), both axes / all cells
assert len(original_codex)==len(relabeled_codex)
qc_only = relabeled_codex['cell_type'][(original_codex['cell_type']=='Other').values & (relabeled_codex['cell_type']!='Other').values]
# composition within each method -- x = fraction of rule-based annotated cells, y = fraction of QuantCell-only cells
xvals = original_codex['cell_type'][original_codex['cell_type']!='Other'].value_counts(normalize=True)
yvals = qc_only.value_counts(normalize=True).reindex(xvals.index).fillna(0)
ax = axd['B']; m = max(xvals.max(), yvals.max())*1.05
ax.plot([0, m], [0, m], color='black', ls='--', lw=1.5, zorder=1)   # fit removed; y = x in the reference-line style
ax.scatter(xvals, yvals, s=40, color='black', zorder=3)
ax.set_xlabel('Fraction of rule-based cells', fontsize=FS); ax.set_ylabel('Fraction of QuantCell cells', fontsize=FS); ax.tick_params(labelsize=FS)
r, p = spearmanr(xvals, yvals); print(f"\n3b within-method: spearman rho={r:.6f}  p={p:.3e}")   # Spearman
ax.set_title(f'$r_s$={r:.3f}   $p$-value={p:.2g}', fontsize=FS); ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim([-0.005, m]); ax.set_ylim([-0.005, m])

ax = axd['C']; ax.plot(pc, g('realised_fdr_pooled')*100, color='black', lw=3.5); ax.plot([0, XMAX], [0, XMAX], ls='--', color='black', lw=1.5)
ax.set_ylabel('Realized FDR (%)'); ax.set_ylim(0, XMAX)   # square axes by size
ax = axd['D']; ax.plot(pcf, annot_full, color='black', lw=3.5); ax.axhline(sw['rule_based_fraction'], color='black', ls='--', lw=1.5, label='Rule-based annotation')
ax.set_ylabel('Fraction annotated'); ax.set_ylim(0, 1.02)
ax = axd['E']; ax.plot(pc, g('macro_precision'), color='#1f77b4', lw=3.5, label='Macro precision'); ax.plot(pc, g('macro_recall'), color='#d62728', lw=3.5, label='Macro recall')
ax.set_ylabel('Score'); ax.set_ylim(0, 1.02)
# both legends to the right of e (d's dotted-line label becomes a legend entry)
axd['D'].legend(frameon=False, loc='lower left', bbox_to_anchor=(0.0, 1.02), fontsize=FS)   # above d, in the white space under a
axd['E'].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.08, 0.80), bbox_transform=axd['E'].transAxes, fontsize=FS)
for k in 'CDE':
    ax = axd[k]; ax.set_xlim(0, XMAX); ax.set_xticks([0, 2, 4, 6, 8, 10]); ax.set_xlabel('FDR target (%)'); ax.axvline(5, color='lightgray', ls='-', lw=1.5, zorder=0); ax.spines[['top', 'right']].set_visible(False)
for k, l, x, y in [('A', 'a', -0.33, 1.03), ('B', 'b', -0.22, 1.04), ('C', 'c', -0.22, 1.04), ('D', 'd', -0.22, 1.04), ('E', 'e', -0.22, 1.04)]:
    axd[k].text(x, y, l.upper(), fontsize=LETTER, fontweight='bold', transform=axd[k].transAxes)
fig.savefig(OUT, dpi=70, bbox_inches='tight'); print(f"\nwritten -> {OUT}")
