""" Fig 1 multiplexed image: locate the Voronoi window (X_LIM 9150-9257, Y_LIM 878-947 um, section Y3) in the
PhenoCycler slide scan 'Little t.qptiff' (45 ch, uint8, 0.3774 um/px), check centroid alignment on DAPI, crop all channels
with a 50 um margin, save npz + contact sheet."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import re, json, numpy as np, pandas as pd, tifffile, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
F=f'{QCP.QPTIFF}'; OUT=f'{QCP.OUT}/fig1_image'
CSV=f'{QCP.OUT}/retrain_lgbm/QuantCellPaper/codex_rule_based_QuantCellPaper.csv'
X_LIM=(9150,9257); Y_LIM=(878,947); M=50
t=tifffile.TiffFile(F); s=t.series[0]; p0=s.pages[0]
xr=p0.tags['XResolution'].value; UMPP=10000*xr[1]/xr[0]; print('um/px', UMPP, 'tiled', p0.is_tiled, p0.compression)
ch=[re.findall(r'<Name>(.*?)</Name>', pg.description or '')[0] for pg in s.pages]
d=pd.read_csv(CSV, index_col=0, usecols=lambda c: c in ('Unnamed: 0','x','y','Parent','cell_type'))
# 1) alignment check: DAPI downsampled 8x with all centroids
dapi=s.pages[0].asarray(); ds=8; small=dapi[::ds, ::ds]
fig,ax=plt.subplots(figsize=(24, 24*small.shape[0]/small.shape[1])); ax.imshow(small, cmap='gray', vmax=np.percentile(small,99.5))
for par,g in d.groupby('Parent'): ax.scatter(g.x/UMPP/ds, g.y/UMPP/ds, s=0.05, alpha=0.3, label=par)
ax.add_patch(plt.Rectangle((X_LIM[0]/UMPP/ds, Y_LIM[0]/UMPP/ds), (X_LIM[1]-X_LIM[0])/UMPP/ds, (Y_LIM[1]-Y_LIM[0])/UMPP/ds, fill=False, ec='red', lw=2))
ax.legend(markerscale=40, fontsize=16); ax.set_title('DAPI (8x down) with all QuPath centroids; red = Fig 1 window', fontsize=18)
fig.savefig(f'{OUT}/alignment_check_dapi_centroids.png', dpi=100, bbox_inches='tight'); plt.close(fig)
# 2) crop with margin, full-res DAPI with Y3 centroids overlaid
x0,x1=int((X_LIM[0]-M)/UMPP), int(np.ceil((X_LIM[1]+M)/UMPP)); y0,y1=int((Y_LIM[0]-M)/UMPP), int(np.ceil((Y_LIM[1]+M)/UMPP))
print('crop px', x0,x1,y0,y1, 'size', x1-x0, y1-y0)
stack=np.stack([pg.asarray()[y0:y1, x0:x1] for pg in s.pages]); print('stack', stack.shape, stack.dtype)
np.savez_compressed(f'{OUT}/fig1_window_crop_margin50um.npz', stack=stack, channels=np.array(ch), x0=x0,y0=y0,x1=x1,y1=y1, umpp=UMPP, X_LIM=X_LIM, Y_LIM=Y_LIM, margin_um=M)
sec=d[(d.x>X_LIM[0]-M)&(d.x<X_LIM[1]+M)&(d.y>Y_LIM[0]-M)&(d.y<Y_LIM[1]+M)]
fig,ax=plt.subplots(figsize=(12, 12*(y1-y0)/(x1-x0))); ax.imshow(stack[0], cmap='gray', vmax=np.percentile(stack[0],99.8))
ax.scatter(sec.x/UMPP-x0, sec.y/UMPP-y0, s=12, c='red', marker='+'); ax.add_patch(plt.Rectangle((X_LIM[0]/UMPP-x0, Y_LIM[0]/UMPP-y0),(X_LIM[1]-X_LIM[0])/UMPP,(Y_LIM[1]-Y_LIM[0])/UMPP, fill=False, ec='yellow', lw=2))
ax.set_title(f'DAPI full-res crop, {len(sec)} Y3 centroids; yellow = Voronoi window', fontsize=14); ax.axis('off')
fig.savefig(f'{OUT}/alignment_check_window_dapi.png', dpi=150, bbox_inches='tight'); plt.close(fig)
# 3) contact sheet of all channels in the window (each scaled to its own 1-99.9 percentile)
n=len(ch); nc=9; nr=int(np.ceil(n/nc)); fig,axs=plt.subplots(nr,nc,figsize=(nc*3.2, nr*3.2*(y1-y0)/(x1-x0)+1))
stats={}
for i,a in enumerate(axs.ravel()):
    a.axis('off')
    if i>=n: continue
    im=stack[i].astype(float); lo,hi=np.percentile(im,[1,99.9]); stats[ch[i]]=dict(p1=float(lo),p99_9=float(hi),mean=float(im.mean()))
    a.imshow(np.clip((im-lo)/max(hi-lo,1),0,1), cmap='gray', vmin=0, vmax=1); a.set_title(f'{i}: {ch[i]}', fontsize=11)
fig.savefig(f'{OUT}/contact_sheet_all_channels.png', dpi=110, bbox_inches='tight'); plt.close(fig)
json.dump(dict(umpp=UMPP, crop_px=dict(x0=x0,x1=x1,y0=y0,y1=y1), channels=ch, n_cells_in_margin_window=int(len(sec)), stats=stats), open(f'{OUT}/crop_meta.json','w'), indent=1)
print('done')
