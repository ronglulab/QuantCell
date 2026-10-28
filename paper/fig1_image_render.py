""" Render RGB composites of the Fig 1 window from fig1_window_crop_margin50um.npz. DAPI blue; other channels additive
in distinct colours, each scaled 1st-99.7th percentile within the margin crop. Usage: fig1_image_render.py NAME "ch:color,ch:color,..."
Outputs <NAME>.png (exact Voronoi window, 6x4 in like the Voronoi panels), <NAME>_margin.png (with 50 um margin,
window outlined), <NAME>_channels.png (one coloured panel per channel)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
OUT=f'{QCP.OUT}/fig1_image'; z=np.load(f'{OUT}/fig1_window_crop_margin50um.npz')
S=z['stack'].astype(float); CH=list(z['channels']); UMPP=float(z['umpp']); x0,y0=int(z['x0']),int(z['y0']); XL,YL=z['X_LIM'],z['Y_LIM']
name=sys.argv[1]; spec=[(c.split(':')[0], c.split(':')[1]) for c in sys.argv[2].split(',')]
def norm(im, lo=1, hi=99.7):
    a,b=np.percentile(im,[lo,hi]); return np.clip((im-a)/max(b-a,1),0,1)
def composite(spec):
    rgb=np.zeros(S.shape[1:]+(3,))
    for ch,col in spec: rgb+=norm(S[CH.index(ch)])[...,None]*np.array(to_rgb(col))[None,None,:]
    return np.clip(rgb,0,1)
rgb=composite(spec)
wx0,wx1=int(round(XL[0]/UMPP-x0)),int(round(XL[1]/UMPP-x0)); wy0,wy1=int(round(YL[0]/UMPP-y0)),int(round(YL[1]/UMPP-y0))
win=rgb[wy0:wy1, wx0:wx1]; bar=20/UMPP
def legend(ax, spec):
    groups={}
    for ch,col in spec: groups.setdefault(col,[]).append(ch)
    for i,(col,chs) in enumerate(groups.items()):
        ax.text(0.01, -0.03-0.075*i, ' / '.join(chs), color=col, transform=ax.transAxes, fontsize=24, va='top', ha='left', fontweight='bold', bbox=dict(facecolor='black', edgecolor='none', pad=2))
fig,ax=plt.subplots(figsize=(6,4)); ax.imshow(win, interpolation='nearest'); ax.axis('off')
fig.savefig(f'{OUT}/{name}.png', dpi=300, bbox_inches='tight', pad_inches=0); plt.close(fig)
fig,ax=plt.subplots(figsize=(9,9*rgb.shape[0]/rgb.shape[1])); ax.imshow(rgb, interpolation='nearest'); ax.axis('off')
ax.add_patch(plt.Rectangle((wx0,wy0),wx1-wx0,wy1-wy0, fill=False, ec='white', lw=1.5, ls='--')); legend(ax, spec)
fig.savefig(f'{OUT}/{name}_margin.png', dpi=300, bbox_inches='tight', pad_inches=0.1); plt.close(fig)
n=len(spec); nc=min(n,6); nr=int(np.ceil(n/nc)); fig,axs=plt.subplots(nr,nc,figsize=(4*nc,nr*(4*win.shape[0]/win.shape[1]+0.7)))
for i,a in enumerate(np.atleast_1d(axs).ravel()):
    a.axis('off')
    if i>=n: continue
    ch,col=spec[i]; a.imshow(composite([(ch,col)])[wy0:wy1,wx0:wx1], interpolation='nearest'); a.set_title(ch, color=col, fontsize=24, fontweight='bold', bbox=dict(facecolor='black', edgecolor='none', pad=2))
fig.savefig(f'{OUT}/{name}_channels.png', dpi=200, bbox_inches='tight'); plt.close(fig)
print('wrote', name, 'window px', win.shape, '')
