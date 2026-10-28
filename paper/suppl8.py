#!/usr/bin/env python
"""Supplementary Fig 8 on the LightGBM retrain artifacts: cell-type frequency, conventional vs QuantCell,
full range (left) and magnified dashed region (right), labelled points. Same inputs as Fig 3c (codex_conventional /
codex_quantcell CSVs). Label positions are placed by a small greedy repel (adjustText is not in repro311)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
project_path=f'{QCP.OUT}/retrain_lgbm/QuantCellPaper'; project_name='QuantCellPaper'
OUT=f'{QCP.OUT}/retrain_lgbm/figures/suppl8_lgbm.png'
orig=pd.read_csv(f'{project_path}/codex_rule_based_{project_name}.csv', usecols=['cell_type'])['cell_type']
rel =pd.read_csv(f'{project_path}/codex_quantcell_{project_name}.csv',   usecols=['cell_type'])['cell_type']
# match Fig 3b -- y = QuantCell-only frequency (rule-unannotated cells labelled by QuantCell), both / all cells
assert len(orig)==len(rel)
# composition within each method (matches Fig 3b): x = fraction of rule-based cells, y = fraction of QuantCell-only cells
of=orig[orig!='Other'].value_counts(normalize=True); rf=rel[(orig=='Other').values & (rel!='Other').values].value_counts(normalize=True)
idx=list(of.index)
x=of.reindex(idx).fillna(0); y=rf.reindex(idx).fillna(0)
r,p=spearmanr(x,y); print(f"rho={r:.4f} p={p:.2e} n={len(idx)}")   # Spearman, as Fig 3b
for c in idx: print(f"  {c:<26} conv {x[c]:.5f}  qc {y[c]:.5f}")
ZOOM=0.020; YZOOM=None   # composition scale: small types sit below 0.035
fs=24; plt.rcParams["font.size"]=fs  # same fonts as the main figures
fig,axs=plt.subplots(1,2,figsize=(21,9),dpi=150)  # same dpi as savefig so pixel offsets match; canvas 26x11 -> 18x7.6 so 24 pt reads larger
def repel(ax, names, xs, ys, extra=(), fit=None, zoom_flag=False):
    """Greedy label placement: try 8 directions at growing radius; accept the first spot whose text box does not
    overlap any point marker or already placed label (in display coords)."""
    fig.canvas.draw(); rend=fig.canvas.get_renderer()
    pts=np.array([ax.transData.transform((a,b)) for a,b in zip(xs,ys)])
    from matplotlib.transforms import Bbox
    allpts=np.array([ax.transData.transform((a,b)) for a,b in zip(x.values,y.values)])
    blockers=[Bbox([[q[0]-12,q[1]-12],[q[0]+12,q[1]+12]]) for q in allpts]
    for (x0,y0,x1,y1) in extra: blockers.append(Bbox([ax.transData.transform((x0,y0)),ax.transData.transform((x1,y1))]))
    if ax.get_legend() is not None: blockers.append(ax.get_legend().get_window_extent(rend))
    placed=[]; placed_raw=[]; arrows=[]
    fitline=[]
    for fx,fy in (fit or []):   # every reference line (fit and y = x) is a line that labels may not sit on
        fp=[ax.transData.transform((u,v)) for u,v in zip(fx,fy)]; fitline+=[(tuple(fp[k]),tuple(fp[k+1])) for k in range(len(fp)-1)]   # [2026-10-23] leaders crossed for close pairs (cDC / Other BM ECs; KLS / GMP) -> arrow-aware placement
    def seg_x(a,b,c,d):   # proper segment intersection a-b vs c-d
        def ccw(p,q,r): return (r[1]-p[1])*(q[0]-p[0])>(q[1]-p[1])*(r[0]-p[0])
        return ccw(a,c,d)!=ccw(b,c,d) and ccw(a,b,c)!=ccw(a,b,d)
    def seg_hits_box(a,b,box,n=25):
        return any(box.contains(a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t) for t in np.linspace(0.08,0.92,n))
    def hard_ok(i,px,py,bb):   # [2026-10-23] -- enforced in every pass
        a=(pts[i][0],pts[i][1]); b=(px,py)
        if any(seg_hits_box(a,b,pb) for pb in placed_raw): return False      # my leader through a placed label
        if any(seg_hits_box(c,d,bb,n=60) for c,d in arrows): return False     # my label on a drawn leader
        if any(seg_hits_box(c,d,bb,n=200) for c,d in fitline): return False   # my label on the dashed fit line
        if any(seg_x(a,b,c,d) for c,d in arrows): return False               # [ ] leaders may not cross each other
        return True
    def leader_ok(i,px,py,bb):
        a=(pts[i][0],pts[i][1]); b=(px,py)
        if any(seg_x(a,b,c,d) for c,d in fitline): return False              # strict pass only: leader may not cross the fit line
        near=lambda blk: np.hypot((blk.x0+blk.x1)/2-a[0],(blk.y0+blk.y1)/2-a[1])<30   # blockers overlapping the point itself cannot be avoided
        return not any(seg_hits_box(a,b,blk) for blk in blockers if not near(blk))
    PINNED=()   # placed first so their preferred spots are free; then outer points first
    order=[i for i in range(len(names)) if names[i] in PINNED]+[i for i in np.argsort(-np.hypot(xs/np.ptp(xs),ys/np.ptp(ys))) if names[i] not in PINNED]
    dirs=[(np.cos(t),np.sin(t)) for t in np.linspace(0,2*np.pi,48,endpoint=False)]
    # [2026-10-23] hand-placed labels (data coords) for the crowded lower-left of panel a; every rule is still checked below
    MANUAL_OLD={'cDC':(0.03,0.006),'Other BM ECs':(0.05,0.034),'Capillary ECs':(0.022,0.119),'B Cells':(0.075,0.06),
            'Neutrophils':(0.09,0.088),'Erythrocytes':(0.035,0.195),'Megakaryocytes':(0.07,0.27)} if not zoom_flag else {}
    # hand placement for the QuantCell-only frequencies (data coords); every rule is still checked and reported below
    # hand placement for the within-method composition scale (data coords); every rule is re-checked below
    MANUAL={'Capillary ECs':(0.045,0.105),'Other BM ECs':(0.085,0.028),'cDC':(0.06,0.006),'B Cells':(0.115,0.085),'Neutrophils':(0.175,0.040),
            'Erythrocytes':(0.065,0.215),'Megakaryocytes':(0.20,0.140)} if not zoom_flag else\
           {'GMP':(0.0065,0.0001),'KLS':(0.0012,0.0035),'CMP':(0.0045,0.0016),'CFU-E':(0.0040,0.0045),'Double Negative T Cells':(0.0105,0.0012)}
    for i in [names.index(nm) for nm in MANUAL if nm in names]:   # MANUAL dict order = placement order
        # MANUAL position = seed; search rings around it (display px) for the first spot passing every hard rule
        sx,sy=ax.transData.transform(MANUAL[names[i]]); axbb=ax.get_window_extent(rend)
        def check(px,py):
            t=ax.text(px,py,names[i],fontsize=fs,transform=None,ha='center',va='center'); raw=t.get_window_extent(rend); bb=raw.expanded(1.2,1.4); t.remove()
            v=[]
            if any(bb.overlaps(b) for b in placed): v.append('label-label')
            if any(bb.overlaps(b) for b in blockers): v.append('label-point/box')
            if not (bb.x0>axbb.x0 and bb.x1<axbb.x1 and bb.y0>axbb.y0 and bb.y1<axbb.y1): v.append('outside axes')
            if not hard_ok(i,px,py,raw): v.append('line-label or leader-leader')
            return v,raw,bb
        cands=[(sx,sy)]+[(sx+rr*np.cos(t),sy+rr*np.sin(t)) for rr in range(15,460,15) for t in np.linspace(0,2*np.pi,48,endpoint=False)]
        for px,py in cands:
            viol,raw,bb=check(px,py)
            if not viol: break
        note='' if leader_ok(i,px,py,bb) else ' (leader crosses the fit line; allowed)'
        moved=np.hypot(px-sx,py-sy)
        print(f"  manual {names[i]!r}: {'OK' if not viol else 'VIOLATION '+', '.join(viol)} (moved {moved:.0f} px from seed){note}")
        ax.annotate(names[i], xy=(xs[i],ys[i]), xycoords='data', xytext=((px-pts[i][0])*72/fig.dpi,(py-pts[i][1])*72/fig.dpi),
                    textcoords='offset points', fontsize=fs, ha='center', va='center',
                    arrowprops=dict(arrowstyle='->',lw=0.8,color='black',shrinkA=0,shrinkB=3), annotation_clip=False)
        placed.append(bb); placed_raw.append(raw); arrows.append(((pts[i][0],pts[i][1]),(px,py)))
    order=[i for i in order if names[i] not in MANUAL]
    for i in order:
        best=None; raw_best=None
        # [2026-10-23] near-coincident pairs (cDC / Other BM ECs; GMP / KLS): preferred leader directions tried first, upper point above, lower point below
        PREF={'cDC':(-15,-25,0,-35),'Other BM ECs':(10,15,5,20),'Capillary ECs':(0,-10,10,-20,20,30),'KLS':(100,80,120,135),'GMP':(-15,0,15,-30)}   # long labels near the left axis must go up-right; cDC right (no room below)
        pdirs=[(np.cos(np.radians(t)),np.sin(np.radians(t))) for t in PREF.get(names[i],())]
        RADII=(55,70,85,100,118,138,160,185,215,250,290,335,385,440,510,600,700)
        for strict,rad,dl in [(st,r,d) for d in ((pdirs,) if pdirs else ())+(dirs,) for st in (True,False) for r in RADII]:   # preferred dirs exhausted (strict, then relaxed) before the general search # pass 1 arrow-aware, pass 2 relaxed
            for dx,dy in dl:
                px,py=pts[i][0]+dx*rad, pts[i][1]+dy*rad
                t=ax.text(px,py,names[i],fontsize=fs,transform=None,ha='center',va='center')
                raw=t.get_window_extent(rend); bb=raw.expanded(1.2,1.4)
                ok=all(not bb.overlaps(b) for b in placed) and all(not bb.overlaps(b) for b in blockers)
                axbb=ax.get_window_extent(rend)
                ok=ok and bb.x0>axbb.x0 and bb.x1<axbb.x1 and bb.y0>axbb.y0 and bb.y1<axbb.y1
                ok=ok and hard_ok(i,px,py,raw) and (leader_ok(i,px,py,bb) if strict else True)
                t.remove()
                if ok: best=(px,py,bb); raw_best=raw; print(f"  placed {names[i]!r}: strict={strict} rad={rad} dir=({dx:+.2f},{dy:+.2f}) pref={dl is pdirs}") if names[i] in PINNED else None; break
            if best: break
        if best is None: print(f"  WARNING: no admissible spot for {names[i]!r}; fallback offset"); best=(pts[i][0]+150,pts[i][1]+150,None)
        px,py,bb=best
        ax.annotate(names[i], xy=(xs[i],ys[i]), xycoords='data', xytext=((px-pts[i][0])*72/fig.dpi,(py-pts[i][1])*72/fig.dpi),
                    textcoords='offset points', fontsize=fs, ha='center', va='center',
                    arrowprops=dict(arrowstyle='->',lw=0.8,color='black',shrinkA=0,shrinkB=3), annotation_clip=False)
        if bb is None:
            t=ax.text(px,py,names[i],transform=None); bb=t.get_window_extent(rend); t.remove()
        placed.append(bb); placed_raw.append(bb if raw_best is None else raw_best); arrows.append(((pts[i][0],pts[i][1]),(px,py)))
todo=[]
for ax,zoom in zip(axs,(False,True)):
    ax.scatter(x,y,s=18,color='black',zorder=3)
    a,b=np.polyfit(x,y,1); xx=np.linspace(0,x.max(),10)
    ax.plot([0,1],[0,1],color='black',ls='--',lw=1.5,zorder=1)   # fit removed; y = x in the reference-line style
    ax.set_xlabel('Fraction of rule-based cells'); ax.set_ylabel('Fraction of QuantCell cells')
    ax.set_title(f'$r_s$={r:.3f}   $p$-value={p:.2g}')   # same format as Fig 3b
    ax.spines[['top','right']].set_visible(False)
    if zoom:
        YZOOM=max(y[x<=ZOOM].max(),ZOOM)*1.1; ax.set_xlim(-0.0002,ZOOM); ax.set_ylim(-0.0002,YZOOM)
        m=(x<=ZOOM)
    else:
        ax.set_xlim(-0.002,x.max()*1.12); ax.set_ylim(-0.005,y.max()*1.12)
        ax.add_patch(plt.Rectangle((0,0),ZOOM,max(y[x<=ZOOM].max(),ZOOM)*1.1,fill=False,linestyle='--',edgecolor='red',lw=1.5))   # zoom-region border red
        m=(x>ZOOM)
    xl,yl=ax.get_xlim(),ax.get_ylim(); fx=np.linspace(xl[0],xl[1],200); fy=a*fx+b; keep=(fy>=yl[0])&(fy<=yl[1]); ki=(fx>=yl[0])&(fx<=yl[1])
    names=[c for c in idx if m[c]]; todo.append((ax,names,zoom,[(fx[ki],fx[ki])]))   # only y = x is a reference line now
fig.tight_layout()
for ax,names,zoom,fit in todo:
    repel(ax,names,x[names].values,y[names].values, extra=() if zoom else ((0,0,ZOOM,max(y[x<=ZOOM].max(),ZOOM)*1.1),), fit=fit, zoom_flag=zoom)
fig.savefig(OUT,dpi=150,bbox_inches='tight'); print('written ->',OUT)
