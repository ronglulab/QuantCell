#!/usr/bin/env python
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import os
"""Figure 5 from published artifacts (cells 100/101/103/104/105/106)."""
import joblib, numpy as np, pandas as pd, json, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

P=f'{QCP.OUT}/retrain_lgbm/QuantCellPaper'
ANN=f'{QCP.COMBOS}'
OUT=f'{QCP.OUT}/retrain_lgbm/figures/fig5_lgbm.png'
cell_order=['KLS','MEP','CMP','GMP','CLP','CFU-E','MkP','Erythroblasts','Erythrocytes',
 'Megakaryocytes','B Cells','CD4 T Cells','CD8 T Cells','cDC','Double Negative T Cells',
 'Monocytes','Neutrophils','Arterial ECs','Capillary ECs','Other BM ECs','Lepr+ Perivascular Cells']

T=f'{QCP.OUT}/retrain/QuantCellPaper/base_models_tuned'   # 10-fold CV of the tuned LightGBM (bm_05_tuned_cv.py)
enc=joblib.load(f'{P}/label_encoder.joblib')
y_test,y_pred=joblib.load(f'{T}/LightGBM_y_test.joblib'),joblib.load(f'{T}/LightGBM_y_pred.joblib')
sm=np.zeros((len(enc.classes_),10))
for n in range(10): sm[:,n]=f1_score(y_test[n],y_pred[n],average=None)
f1_scores={l:np.mean(sm[i,:]) for i,l in enumerate(enc.classes_)}

codex=pd.read_csv(f'{P}/codex_rule_based_QuantCellPaper.csv',
                  usecols=['cell_type','Cell: Area µm^2','Cell: Circularity'])
combos=json.load(open(ANN))
vc=codex['cell_type'].value_counts(); total=vc.sum()-vc['Other']
frequency_dict={ct:vc[ct]/total for ct in cell_order}
size_dict={ct:codex.loc[codex.cell_type==ct,'Cell: Area µm^2'].mean() for ct in cell_order}
circularity_dict={ct:codex.loc[codex.cell_type==ct,'Cell: Circularity'].mean() for ct in cell_order}
number_marker_dict={ct:len(combos[ct]) for ct in cell_order}

gko=joblib.load(f'{P}/model_selection/gene_ko_dict.joblib')
gene_order=sorted(gko.keys(), key=str.lower)
# matched baseline 'Recompute'): same nested population + folds as the knockouts, nothing excluded
bl=joblib.load(f'{P}/model_selection/gene_ko_baseline_nested.joblib') if os.environ.get('QC_KO_BASELINE','nested')=='nested' else (y_test,y_pred)
f1_baseline=np.mean([f1_score(bl[0][n],bl[1][n],average=None) for n in range(10)],axis=0)
f1_matrix=np.zeros((len(gene_order),10,len(enc.classes_)))
for n,g in enumerate(gene_order):
    for fold in range(10):
        f1_matrix[n,fold,:]=f1_score(gko[g][0][fold],gko[g][1][fold],average=None)
rows=[]
for n,g in enumerate(gene_order):
    for fold in range(10):
        for ct in enc.classes_:
            ci=list(enc.classes_).index(ct)
            rows.append((g,fold,ct,f1_matrix[n,fold,ci]-f1_baseline[ci]))
long_df=pd.DataFrame(rows, columns=['Gene','Fold','Cell Type','F1 Score'])

default_fontsize, big_fontsize = 24, 30
plt.rcParams['font.size']=default_fontsize
fig,axs=plt.subplots(4,2,figsize=(17,22)); axs=axs.flatten()
for i,pos in enumerate([[0.05,0.75,0.4,0.13],[0.55,0.75,0.4,0.13],[0.05,0.53,0.4,0.13],
                        [0.55,0.53,0.4,0.13],[0.05,0.28,0.4,0.18],[0.55,0.28,0.4,0.18],
                        [0.05,0.05,0.4,0.18],[0.55,0.05,0.4,0.18]]):
    axs[i].set_position(pos)

for i,(d,xlab) in enumerate([(frequency_dict,'Cell frequency'),(size_dict,'Cell size (µm²)'),
                             (circularity_dict,'Cell circularity'),(number_marker_dict,'Number of markers')]):
    xs=[d[ct] for ct in cell_order]; ys=[f1_scores[ct] for ct in cell_order]
    axs[i].scatter(xs,ys,color='black')
    axs[i].set_xlabel(xlab,fontsize=default_fontsize)
    if i in (0,2): axs[i].set_ylabel('F1 score',fontsize=default_fontsize)
    axs[i].spines[['top','right']].set_visible(False); axs[i].tick_params(labelsize=default_fontsize)
    if i==0: axs[i].set_xlim(0, 0.35)   # linear, limit covers Erythroblasts (0.328); the old 0-0.3 limit clipped it
    r,p=spearmanr(xs,ys); p_adj=min(1.0,p*4)   # Spearman (skewed frequency, bounded F1 with outliers), Bonferroni x4
    axs[i].text(0.20,1.05,f'$r_s$= {r:.2f}       $p$-value={p_adj:.2g}',fontsize=default_fontsize,transform=axs[i].transAxes); print(f'5a {xlab}: rho={r:.3f} p={p:.2e} p_bonferroni={p_adj:.2e}')

# NOTE: threshold uses the F1 Score column explicitly (see — the original
# relies on `Fold` being object dtype so that .values[0] lands on F1 Score)
FLOOR,FRAC=0.02,0.25  # annotation rule mean dF1 <= -0.02 and >= 25% of the panel's largest drop (floor 0.05 -> 0.02 after the B-cell clipping fix)
mean_std_by_ct = long_df.groupby(['Gene','Cell Type'])['F1 Score'].std().groupby('Cell Type').mean()
for i,ct in enumerate(['KLS','Erythrocytes','Arterial ECs','CD8 T Cells']):
    sub=long_df[long_df['Cell Type']==ct]
    sns.swarmplot(data=sub,x='Gene',y='F1 Score',ax=axs[4+i],color='black',size=4,order=gene_order)
    mv=sub.groupby('Gene')['F1 Score'].mean().reindex(gene_order)
    sns.pointplot(x=mv.index,y=mv.values,ax=axs[4+i],color='red',errorbar=None,linestyle='none',order=gene_order)
    axs[4+i].spines[['top','right']].set_visible(False)
    axs[4+i].set_title(ct,fontsize=default_fontsize)
    axs[4+i].axhline(0,color='black',linestyle='--',linewidth=1.5)
    axs[4+i].set_ylabel('Δ F1 Score',fontsize=default_fontsize)
    axs[4+i].set_xticklabels([' ']*len(gene_order),fontsize=default_fontsize)
    axs[4+i].tick_params(labelsize=default_fontsize)
    thr=max(FLOOR,FRAC*(-sub.groupby('Gene')['F1 Score'].mean().min())); offset=-1; ylim=0
    for n,gene in enumerate(gene_order):
        md=sub[sub.Gene==gene]['F1 Score'].mean(); mn=sub[sub.Gene==gene]['F1 Score'].min()
        if md<=-thr:
            if n<5: offset=1
            if n>len(gene_order)-4: offset=-1
            if gene=='Endomucin': offset=1
            axs[4+i].text(n+1.6*offset+0.2*len(gene)*offset,1.1*mn,gene,
                          fontsize=default_fontsize,ha='center',va='bottom')
            offset*=-1
            if mn<ylim: ylim=mn
    axs[4+i].set_ylim([1.2*min(ylim, sub['F1 Score'].min()),None])  # never clip
    print(f"5b {ct:<14} annotated: {[g for g in gene_order if sub[sub.Gene==g]['F1 Score'].mean()<=-thr]}")
axs[4].set_xlabel(''); axs[5].set_xlabel('')
axs[6].set_xlabel('Marker excluded',fontsize=default_fontsize)
axs[7].set_xlabel('Marker excluded',fontsize=default_fontsize)
axs[0].text(-0.15,1.1,'A',fontsize=big_fontsize,fontweight='bold',transform=axs[0].transAxes)
axs[4].text(-0.15,1.1,'B',fontsize=big_fontsize,fontweight='bold',transform=axs[4].transAxes)
fig.savefig(OUT,dpi=70,bbox_inches='tight')
print(f"written -> {OUT}")
