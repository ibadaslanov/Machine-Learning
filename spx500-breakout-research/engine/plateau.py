import json, numpy as np, re, sys
COST=0.5
res=np.load("grid2_res.npy", mmap_mode="r"); combos=np.load("grid2_combos.npy"); labels=json.load(open("grid2_labels.json"))
lab_idx={l:i for i,l in enumerate(labels)}
def netio(c,j,fade=False):
    r=np.asarray(res[c,j]); g=r[:,0:8]; n=r[:,8:16]; f=r[:,16:24]
    if fade:
        IS=(-g[:,:4]-COST*n[:,:4]).sum(1)-r[:,24]; OOS=(-g[:,4:]-COST*n[:,4:]).sum(1)-r[:,25]
    else:
        net=g-COST*n-f; IS=net[:,:4].sum(1); OOS=net[:,4:].sum(1)
    return float(IS.min()), float(OOS.min())
def cidx(S,F,b,a):
    w=np.where((combos[:,0]==S)&(combos[:,1]==F)&(np.abs(combos[:,2]-b)<1e-9)&(combos[:,3]==a))[0]; return int(w[0]) if len(w) else None
def label_neighbours(l):
    out=[]
    m=re.match(r"DAY R=(\d\d) WF \| (\w+) \| entries (.*)",l)
    if m:
        R=int(m.group(1))
        for dR in (-2,-1,1,2):
            R2=(R+dR)%24; w=m.group(3)
            w2=w if w=='any hour' else re.sub(r"^\d\d",f"{R2:02d}",w)
            out.append(f"DAY R={R2:02d} WF | {m.group(2)} | entries {w2}")
    m=re.match(r"TIME (\d+)h WF \| (\w+) \| entries (\d\d):00 only",l)
    if m:
        E=int(m.group(3))
        for dE in (-2,-1,1,2): out.append(f"TIME {m.group(1)}h WF | {m.group(2)} | entries {(E+dE)%24:02d}:00 only")
        for N in (12,24,48,120):
            if str(N)!=m.group(1): out.append(f"TIME {N}h WF | {m.group(2)} | entries {E:02d}:00 only")
    m=re.match(r"WEEK (\w\w\w) (\d\d):00 \| (\w+)",l)
    if m:
        H=int(m.group(2))
        for dH in (-2,-1,1,2):
            H2=H+dH
            if 0<=H2<=23: out.append(f"WEEK {m.group(1)} {H2:02d}:00 | {m.group(3)}")
    return [x for x in out if x in lab_idx]
def plateau(S,F,b,a,l,fade=False):
    j=lab_idx[l]; c=cidx(S,F,b,a); base=netio(c,j,fade)
    pts=[]
    for dS in (-1,0,1):
        for dF in (-1,0,1):
            if dS==dF==0: continue
            c2=cidx((S+dS)%24,(F+dF)%24,b,a)
            if c2 is not None: pts.append(('SF',netio(c2,j,fade)))
    for (b2,a2) in [(0.0,0),(0.11,0),(0.3,0),(0.3,1)]:
        if (abs(b2-b)>1e-9 or a2!=a):
            c2=cidx(S,F,b2,a2); pts.append(('buf',netio(c2,j,fade)))
    for l2 in label_neighbours(l): pts.append(('time',netio(c,lab_idx[l2],fade)))
    IS=np.array([p[1][0] for p in pts]); OOS=np.array([p[1][1] for p in pts])
    return dict(strategy=f"{'FADE ' if fade else ''}S={S} F={F} b={b} {'always' if a else 'overnight'} | {l}", IS=round(base[0]), OOS=round(base[1]),
        neighbours=len(pts), nb_IS_median=round(float(np.median(IS))), nb_IS_p25=round(float(np.percentile(IS,25))),
        nb_OOS_median=round(float(np.median(OOS))), nb_OOS_p25=round(float(np.percentile(OOS,25))), nb_OOS_share_pos=round(float((OOS>0).mean()),2))
if __name__=='__main__':
    for c in json.loads(sys.argv[1]): print(json.dumps(plateau(*c[:5], fade=bool(c[5]) if len(c)>5 else False)))
