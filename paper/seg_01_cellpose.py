"""Membrane-based segmentation for Supplementary Fig. 3: Cellpose-SAM (cellpose 4.0.6) on the full PhenoCycler slide.
Two runs, as in the paper: (1) a two-channel image of a membrane composite (F4/80, CD45, Endomucin, Sca1, CD71, weighted) and
DAPI, giving whole-cell masks; (2) DAPI alone, giving nuclear masks. Saves masks.npy and dapi_masks.npy (uint32 label images,
full slide) to OUT/cellpose/. A GPU is strongly recommended; on CPU this takes many hours and a lot of memory.
Requires: pip install cellpose==4.0.6 tifffile"""
import os, re, sys, numpy as np, tifffile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import config as QCP
OUT=f'{QCP.OUT}/cellpose'; os.makedirs(OUT, exist_ok=True)
def rescale(a):
    a=a.astype(np.float64); a=a-a.min(); a=a/a.max() if a.max()>0 else a; return (a*255).astype(np.uint8)
t=tifffile.TiffFile(QCP.QPTIFF); s=t.series[0]
names=[re.findall(r'<Name>(.*?)</Name>', pg.description or '')[0] for pg in s.pages]
img=lambda n: s.pages[names.index(n)].asarray()
dapi=img('dapi1')
factors={'F4_80':1.0, 'CD45':0.6, 'Endomucin':0.4, 'Sca1':0.4, 'CD71':0.6}
composite=np.zeros(dapi.shape, dtype=np.float64)
for ch,f in factors.items(): composite+=rescale(img(ch))*f
two=np.zeros((*dapi.shape, 2), dtype=np.uint8); two[...,0]=rescale(composite); two[...,1]=rescale(dapi)
from cellpose import models
model=models.CellposeModel(gpu=os.environ.get('QC_GPU','1')=='1')
dapi_masks,_,_=model.eval(dapi); np.save(f'{OUT}/dapi_masks.npy', dapi_masks.astype(np.uint32)); print('nuclear masks', int(dapi_masks.max()))
masks,_,_=model.eval(two); np.save(f'{OUT}/masks.npy', masks.astype(np.uint32)); print('cell masks', int(masks.max()))
