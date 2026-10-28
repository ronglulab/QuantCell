#!/usr/bin/env python
""" Supplementary Fig 5 — training time of the ten base (default-parameter) classifiers, mean per fold of the 10-fold CV
(fit + predict, as timed by quantcell.test_model_list in r2_base_models.py; 12 threads where the estimator supports it).
Read from retrain_lgbm/QuantCellPaper/base_models/*_time_elapsed.joblib (the Fig 2a inputs). Published layout: horizontal bars,
log x axis 1-1000 s, value labels, published model order. 24 pt."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import joblib, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
P=f'{QCP.OUT}/retrain_lgbm/QuantCellPaper/base_models'; OUT=f'{QCP.OUT}/retrain_lgbm/figures/suppl5_lgbm.png'
NAMES={'Random Forest':'Random forest','Multilayer Perceptron':'Multilayer perceptron','Extra Trees':'Extra trees','Logistic Regression':'Logistic regression',
       'Linear SVC':'Linear SVC','Nearest Neighbors':'Nearest neighbors','Decision Tree':'Decision tree','Ridge Classifier':'Ridge classifier',
       'LightGBM':'LightGBM','Gaussian Naive-Bayes':'Gaussian naive-Bayes'}
# order and colours as Fig 2a (tuned 10-fold macro AP, best on top; fig2_render_v3.py color_dict)
tsum=json.load(open(f'{QCP.OUT}/retrain/QuantCellPaper/base_models_tuned/base_models_tuned_summary.json'))
ORDER=[(k,NAMES[k]) for k in sorted(NAMES, key=lambda n: -tsum[n]['macro_AP_mean'])]
NFOLD=10; t={k:joblib.load(f'{P}/{k}_time_elapsed.joblib')/NFOLD for k,_ in ORDER}
color_dict={'Nearest Neighbors':'#1f77b4','Multilayer Perceptron':'mediumslateblue','Random Forest':'#2ca02c','Extra Trees':'#d62728','Decision Tree':'darkgoldenrod',
    'Gaussian Naive-Bayes':'#8c564b','Ridge Classifier':'darkcyan','Linear SVC':'#e377c2','LightGBM':'#7f7f7f','Logistic Regression':'#bcbd22'}
COL=[color_dict[k] for k,_ in ORDER]
FS=24; plt.rcParams['font.size']=FS
fig,ax=plt.subplots(figsize=(14,8))
y=np.arange(len(ORDER))[::-1]; v=[t[k] for k,_ in ORDER]
ax.barh(y,v,color='white',edgecolor='black',lw=0.8,height=0.75); ax.barh(y,v,color=COL,alpha=0.45,edgecolor='black',lw=0.8,height=0.75)   # Fig 2a 'Default' convention: colour at 45 % alpha over white
for yy,vv in zip(y,v): ax.text(vv*1.06,yy,f'{vv:.2f}',va='center',ha='left',fontsize=FS)
ax.set_xscale('log'); ax.set_xlim(1,1000); ax.set_xticks([1,10,100,1000],['1','10','100','1000']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_yticks(y,[n for _,n in ORDER]); ax.set_xlabel('Runtime per fold (seconds)')
ax.spines[['top','right']].set_visible(False); ax.tick_params(axis='y',length=0)
fig.tight_layout(); fig.savefig(OUT,dpi=150); print('written ->',OUT)
json.dump({n:round(t[k],2) for k,n in ORDER},open(OUT.replace('.png','.json'),'w'),indent=1); print({n:round(t[k],2) for k,n in ORDER})
