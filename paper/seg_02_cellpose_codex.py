""" Rebuild the membrane-based (Cellpose) codex for Suppl Fig 3 from the full-slide Cellpose label images and the
PhenoCycler slide scan, with labels transferred from the REVISION nuclei-based codex (corrected marker calls, renamed types).
Compartments: Cell = Cellpose whole-cell mask; Nucleus = whole-cell mask where the DAPI-only Cellpose mask is >0; Cytoplasm =
whole-cell minus nucleus (manuscript Methods). Valid cell = has both nucleus and cytoplasm pixels (segmentation_performance.ipynb
cell 7). Kept cell = valid AND contains exactly one StarDist centroid from the revision codex (ipynb cells 33-36; .
Stats per marker x compartment: Mean, Median, Min, Max, Std.Dev. (population) from per-cell 8-bit histograms at native 0.3774 um/px.
x, y = Cellpose whole-cell centroid in um. Output: retrain_lgbm/QuantCellPaperCellpose/codex_rule_based_QuantCellPaperCellpose.csv"""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import re, json, time, numpy as np, pandas as pd, tifffile
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
W=f'{QCP.OUT}/cellpose'; OUTD=f'{QCP.OUT}/retrain_lgbm/QuantCellPaperCellpose'
QP=f'{QCP.QPTIFF}'
SRC=f'{QCP.OUT}/retrain/QuantCellPaper/codex_rule_based_QuantCellPaper.csv'
NAME_MAP={'B220':'CD45R_B220','CD11b':'Cd11b','CD8':'CD8a','Lepr':'LePr'}
masks=np.load(f'{W}/masks.npy'); dapi=np.load(f'{W}/dapi_masks.npy'); H,Wd=masks.shape; nmax=int(masks.max()); log(f'masks {masks.shape} labels {nmax}')
t=tifffile.TiffFile(QP); s=t.series[0]; xr=s.pages[0].tags['XResolution'].value; UMPP=10000*xr[1]/xr[0]
CH=[re.findall(r'<Name>(.*?)</Name>', pg.description or '')[0] for pg in s.pages]; log(f'um/px {UMPP:.5f}, {len(CH)} channels')
# --- compartments
nuc=(dapi>0); del dapi
area_cell=np.bincount(masks.ravel(), minlength=nmax+1); area_nuc=np.bincount(masks[nuc], minlength=nmax+1); area_cyt=area_cell-area_nuc
valid=(area_nuc>0)&(area_cyt>0); valid[0]=False; log(f'cells {nmax}: valid (nucleus and cytoplasm) {valid.sum()}')
# --- StarDist centroid transfer from the revision codex
src=pd.read_csv(SRC); log(f'revision codex {src.shape}')
px=(src.x.values/UMPP).astype(int); py=(src.y.values/UMPP).astype(int); inb=(px>=0)&(px<Wd)&(py>=0)&(py<H)
lab=np.zeros(len(src),dtype=np.int64); lab[inb]=masks[py[inb],px[inb]]
hits=np.bincount(lab, minlength=nmax+1); one=(hits==1); one[0]=False
keep=valid&one; log(f'StarDist centroids in a Cellpose cell {int((lab>0).sum())} / {len(src)}; cells hit exactly once {one.sum()}; kept (valid & one) {keep.sum()}; valid cells with >=2 centroids {int((valid&(hits>=2)).sum())}; valid with 0 {int((valid&(hits==0)).sum())}')
row_of=np.full(nmax+1,-1,dtype=np.int64); m=keep[lab]; row_of[lab[m]]=np.nonzero(m)[0]
K=int(keep.sum()); remap=np.full(nmax+1,-1,dtype=np.int64); kept_labels=np.nonzero(keep)[0]; remap[kept_labels]=np.arange(K)
# --- pixel index sets per compartment (kept cells only)
sel=keep[masks]; ys,xs=np.nonzero(sel); cid=remap[masks[ys,xs]]; isn=nuc[ys,xs]; del sel, nuc
comp={'Cell':(ys,xs,cid),'Nucleus':(ys[isn],xs[isn],cid[isn]),'Cytoplasm':(ys[~isn],xs[~isn],cid[~isn])}
log({k:len(v[0]) for k,v in comp.items()})
cx=np.bincount(cid, weights=xs, minlength=K)/np.bincount(cid, minlength=K); cy=np.bincount(cid, weights=ys, minlength=K)/np.bincount(cid, minlength=K)
vals=np.arange(256, dtype=np.float64)
def stats(idx, v):
    h=np.bincount(idx.astype(np.int64)*256+v.astype(np.int64), minlength=K*256).reshape(K,256).astype(np.float64)
    n=h.sum(1); mean=(h@vals)/n; var=(h@(vals**2))/n-mean**2; std=np.sqrt(np.clip(var,0,None))
    nz=h>0; mn=np.argmax(nz,axis=1).astype(float); mx=(255-np.argmax(nz[:,::-1],axis=1)).astype(float)
    c=np.cumsum(h,axis=1); med=np.argmax(c>=(n[:,None]/2.0),axis=1).astype(float)
    return mean,med,mn,mx,std
markers=[c for c in src.columns if c in NAME_MAP or (c in CH and ': ' not in c)]
markers=[c for c in ['B220','CD105','CD115','CD117','CD11b','CD11c','CD127','CD19','CD3','CD31','CD34','CD4','CD41','CD45','CD71','CD8','Endomucin','F4_80','Fcgr','Lepr','Ly6g','MHCII','Sca1','TCR','Ter119'] if c in src.columns]
assert len(markers)==25, markers
feat={}
for mk in markers:
    ch=NAME_MAP.get(mk,mk); page=s.pages[CH.index(ch)].asarray()
    for cname,(yy,xx,ii) in comp.items():
        mean,med,mn,mx,std=stats(ii, page[yy,xx])
        for sname,arr in zip(['Mean','Median','Min','Max','Std.Dev.'],[mean,med,mn,mx,std]): feat[f'{mk}: {cname}: {sname}']=arr
    log(f'  {mk} ({ch}) done'); del page
# --- assemble
rows=src.iloc[row_of[kept_labels]].reset_index(drop=True)
meta=[c for c in src.columns if not re.search(r"\w+: (Nucleus|Cytoplasm|Membrane|Cell): (Mean|Median|Min|Max|Std\.Dev\.)", c) and not c.startswith(('Nucleus:','Cell:','Nucleus/Cell'))]
out=rows[meta].copy(); out['x']=cx*UMPP; out['y']=cy*UMPP; out['Cellpose Label']=kept_labels
out['Nucleus: Area µm^2']=area_nuc[kept_labels]*UMPP**2; out['Cell: Area µm^2']=area_cell[kept_labels]*UMPP**2; out['Nucleus/Cell area ratio']=area_nuc[kept_labels]/area_cell[kept_labels]
out['StarDist x']=rows['x'].values; out['StarDist y']=rows['y'].values
out=pd.concat([out, pd.DataFrame(feat)], axis=1); assert out.isna().sum().sum()==0
out.to_csv(f'{OUTD}/codex_rule_based_QuantCellPaperCellpose.csv', index=False)
summ=dict(umpp=UMPP, cellpose_cells=nmax, valid=int(valid.sum()), stardist_cells=int(len(src)), stardist_in_cell=int((lab>0).sum()), cells_hit_once=int(one.sum()), kept=K,
          valid_multi_centroid=int((valid&(hits>=2)).sum()), valid_no_centroid=int((valid&(hits==0)).sum()), n_features=len(feat), markers=markers,
          cell_type_counts=out.cell_type.value_counts().to_dict(), parent_counts=out.Parent.value_counts().to_dict(),
          stardist_vs_cellpose_centroid_um=dict(mean_dx=float(np.mean(out['x']-out['StarDist x'])), mean_dy=float(np.mean(out['y']-out['StarDist y'])), median_abs=float(np.median(np.hypot(out['x']-out['StarDist x'], out['y']-out['StarDist y'])))))
json.dump(summ, open(f'{OUTD}/build_summary.json','w'), indent=1); log(json.dumps({k:v for k,v in summ.items() if k not in ('cell_type_counts','markers')}))
log(f'wrote {out.shape} -> {OUTD}/codex_rule_based_QuantCellPaperCellpose.csv')
