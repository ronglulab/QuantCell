""" Suppl Fig 3a, wide field panel a the same height as the other rows -> image area 27.8 x 6.2 in,
aspect 4.484). Field = the Fig 1 window's vertical extent (y 878-947 um), widened symmetrically about the window centre to the
target aspect; left half DAPI only, right half DAPI + StarDist (white) + Cellpose (red, traced & smoothed) outlines; white divider.
Inputs: Little t.qptiff (DAPI = page 0), work/cellpose/masks.npy, the full-slide QuPath geojson. No scale bar, no legend in the image
(the legend strip is drawn by the composite). Outputs fig1_image/suppl3a_wide_split.png + suppl3a_wide_meta.json."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import re, json, time, numpy as np, tifffile, contourpy, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
D=f'{QCP.OUT}/fig1_image'; F=f'{QCP.QPTIFF}'
GEO=f'{QCP.STARDIST_GEOJSON}'
MASKS=f'{QCP.OUT}/cellpose/masks.npy'
import sys; AW,AH=(float(sys.argv[1]),float(sys.argv[2])) if len(sys.argv)>2 else (27.8,6.2); X_LIM=(9150,9257); Y_LIM=(878,947); UMPP=0.37740; SMOOTH_SIGMA=1.0; LW=2.0  # lw scaled to the 27.8-in panel (same relative weight as 0.6 pt on the 6-in crop)
# field in px: window height, widened to the panel aspect about the window centre
y0=int(Y_LIM[0]/UMPP); y1=int(np.ceil(Y_LIM[1]/UMPP)); H=y1-y0; W=int(round(H*AW/AH)); xc=(X_LIM[0]+X_LIM[1])/2/UMPP
x0=int(round(xc-W/2)); x1=x0+W; print(f'field px x {x0}-{x1} ({W}) y {y0}-{y1} ({H}); aspect {W/H:.3f} target {AW/AH:.3f}; {W*UMPP:.0f} x {H*UMPP:.0f} um')
t=time.time(); dapi=tifffile.TiffFile(F).series[0].pages[0].asarray()[y0:y1, x0:x1].astype(float); print(f'DAPI read {time.time()-t:.0f}s')
lo,hi=np.percentile(dapi,[1,99.7]); b=np.clip((dapi-lo)/(hi-lo),0,1); rgb=np.zeros(dapi.shape+(3,)); rgb[...,2]=b
cp=np.load(MASKS, mmap_mode='r')[y0:y1, x0:x1]; cp=np.asarray(cp)
# StarDist polygons whose first vertex lies in the field (+60 px)
M=60; geo=re.compile(rb'"geometry":\{"type":"Polygon","coordinates":(\[\[\[[^\]]*\](?:,\[[^\]]*\])*\]\])\}')
t=time.time(); sd=[]; n=0; tail=b''
with open(GEO,'rb') as f:
    while True:
        chunk=f.read(64*1024*1024)
        if not chunk: break
        buf=tail+chunk; last=0
        for m in geo.finditer(buf):
            n+=1; last=m.end(); c=json.loads(m.group(1))[0]; x,y=c[0]
            if x0-M<=x<=x1+M and y0-M<=y<=y1+M: sd.append(np.array(c,float))
        tail=buf[max(last-100000,len(buf)-200000):] if last else buf[-200000:]
print(f'StarDist scanned {n}, in field {len(sd)}, {time.time()-t:.0f}s')
def mask_polygons(cp):
    polys=[]
    for lab in np.unique(cp):
        if lab==0: continue
        ys,xs=np.nonzero(cp==lab); ya,yb,xa,xb=ys.min(),ys.max(),xs.min(),xs.max()
        PAD=3; sub=gaussian_filter(np.pad((cp[ya:yb+1,xa:xb+1]==lab).astype(float),PAD), SMOOTH_SIGMA)
        for line in contourpy.contour_generator(z=sub).lines(0.5): polys.append(np.column_stack([line[:,0]+xa-PAD, line[:,1]+ya-PAD]))
    return polys
cpp=mask_polygons(cp); print('Cellpose labels', len(np.unique(cp))-1, 'polygons', len(cpp))
def clip_right(px, py, xm):
    runs=[]; cur=[]
    for i in range(len(px)-1):
        xa,ya,xb,yb=px[i],py[i],px[i+1],py[i+1]
        if xa>=xm and xb>=xm: cur += ([(xa,ya)] if not cur else []) + [(xb,yb)]
        elif xa>=xm and xb<xm: tt=(xm-xa)/(xb-xa); cur += ([(xa,ya)] if not cur else []) + [(xm,ya+tt*(yb-ya))]; runs.append(cur); cur=[]
        elif xa<xm and xb>=xm: tt=(xm-xa)/(xb-xa); cur=[(xm,ya+tt*(yb-ya)),(xb,yb)]
    if cur: runs.append(cur)
    return runs
fig=plt.figure(figsize=(AW,AH)); ax=fig.add_axes([0,0,1,1]); ax.imshow(rgb, interpolation='nearest'); xm=W/2
for q in [p-np.array([x0,y0]) for p in sd]+cpp:
    pass
for p in sd:
    q=p-np.array([x0,y0])
    if q[:,0].max()<xm: continue
    for run in clip_right(q[:,0],q[:,1],xm): r=np.array(run); ax.plot(r[:,0],r[:,1],color='white',lw=LW,solid_capstyle='butt')
for q in cpp:
    if q[:,0].max()<xm: continue
    for run in clip_right(q[:,0],q[:,1],xm): r=np.array(run); ax.plot(r[:,0],r[:,1],color='red',lw=LW,solid_capstyle='butt')
ax.axvline(xm,color='white',lw=1); ax.set_xlim(-0.5,W-0.5); ax.set_ylim(H-0.5,-0.5); ax.axis('off')
fig.savefig(f'{D}/suppl3a_wide_split.png', dpi=150); plt.close(fig)
json.dump(dict(field_px=dict(x0=x0,x1=x1,y0=y0,y1=y1), field_um=dict(w=W*UMPP,h=H*UMPP), umpp=UMPP, panel_in=dict(w=AW,h=AH), n_stardist=len(sd), n_cellpose=int(len(np.unique(cp))-1), smooth_sigma=SMOOTH_SIGMA, lw=LW), open(f'{D}/suppl3a_wide_meta.json','w'), indent=1)
print('written', f'{D}/suppl3a_wide_split.png')
