""" Generic table-as-PNG renderer shared by suppl_table2_png.py and suppl_table3_png.py (24 pt, figure typeface)."""
import os as _qos, sys as _qsys; _qsys.path.insert(0, _qos.path.dirname(_qos.path.abspath(__file__))); import config as QCP
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FS=24; plt.rcParams["font.size"]=FS; plt.rcParams["font.family"]="DejaVu Sans"
LINE_IN=FS/72*1.45   # inches per text line
def render(cols, widths, rows, out, group_breaks=(), pad=0.18):
    """cols: header strings; widths: inches; rows: list of lists of strings (may contain newlines); group_breaks: row indices before which a heavier rule is drawn."""
    txt=lambda c: c[0] if isinstance(c,tuple) else c; bold=lambda c: isinstance(c,tuple) and c[1]
    nl=[max(txt(c).count('\n')+1 for c in r) for r in rows]; hl=max(c.count('\n')+1 for c in cols)
    heights=[h*LINE_IN+2*pad for h in nl]; hh=hl*LINE_IN+2*pad; W=sum(widths)+2*pad; H=hh+sum(heights)+2*pad
    fig=plt.figure(figsize=(W,H)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off')
    x0=[pad+sum(widths[:i]) for i in range(len(widths))]; y=H-pad
    ax.plot([pad,W-pad],[y,y],color='black',lw=2.0)
    for i,c in enumerate(cols): ax.text(x0[i]+pad, y-hh/2, c, ha='left', va='center', fontweight='bold')
    y-=hh; ax.plot([pad,W-pad],[y,y],color='black',lw=2.0)
    for ri,(r,h) in enumerate(zip(rows,heights)):
        if ri in group_breaks: ax.plot([pad,W-pad],[y,y],color='black',lw=1.2)
        elif ri>0: ax.plot([pad,W-pad],[y,y],color='#bbbbbb',lw=0.6)
        for i,c in enumerate(r): ax.text(x0[i]+pad, y-h/2, txt(c), ha='left', va='center', fontweight='bold' if bold(c) else 'normal')
        y-=h
    ax.plot([pad,W-pad],[y,y],color='black',lw=2.0); fig.savefig(out, dpi=100); plt.close(fig); print('written ->', out, f'({W:.1f} x {H:.1f} in)')
