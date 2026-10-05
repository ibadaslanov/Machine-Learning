import numpy as np, json
from scipy.stats import norm, skew, kurtosis
import engine2 as E2
X=E2.Ctx(); d=X.d
def daily_pct(S,F,b,al,label,fade=False,dirmode=0):
    worst=None
    for fb in (0,1,2):
        st,tr=X.run(S,F,b,al,label,fb=fb,reopen_mode=2,dirmode=dirmode)
        g=(tr[:,4]-tr[:,3])*tr[:,2]*(-1 if fade else 1)
        net=g-0.5-(tr[:,5] if not fade else 0)  # fade financing approx ignored here (small)
        if worst is None or net.sum()<worst[0].sum(): worst=(net,tr)
    net,tr=worst
    # mark-to-market daily % returns on a 1-unit position, financing/cost booked at exit
    C=d['C']; day=d['eh']//24
    pnl=np.zeros(len(C))
    for (e,x,dr,ep,xp,f),nt in zip(tr,net):
        e,x=int(e),int(x); dr=dr*(-1 if fade else 1)
        path=np.r_[ep,C[e:x],xp]; pnl[e:x+1]+=np.diff(path)*dr
        pnl[x]+=nt-(xp-ep)*dr
    u,inv=np.unique(day,return_inverse=True)
    dp=np.bincount(inv,pnl); px=np.bincount(inv,C)/np.bincount(inv)
    return dp/px
def dsr(r,N):
    T=len(r); sr=r.mean()/r.std(); g3=skew(r); g4=kurtosis(r,fisher=False)
    V=1.0/T; gam=0.5772
    srstar=np.sqrt(V)*((1-gam)*norm.ppf(1-1/N)+gam*norm.ppf(1-1/(N*np.e)))
    z=(sr-srstar)*np.sqrt(T-1)/np.sqrt(1-g3*sr+(g4-1)/4*sr**2)
    return float(norm.cdf(z)), float(sr*np.sqrt(252))
C=[("FADE first break, R=06 hold 24h",(8,18,0.3,False,"DAY R=06 WF | HOLD | entries any hour",True)),
   ("08:00-range break after 09:00, hold to 09:00",(15,8,0,False,"DAY R=09 WF | HOLD | entries any hour",False)),
   ("same, LONG-ONLY",(15,8,0,False,"DAY R=09 WF | HOLD | entries any hour",False,1)),
   ("15:00 entry only, hold 24h",(0,12,0.3,False,"TIME 24h WF | HOLD | entries 15:00 only",False)),
   ("weekly: first break from Mon 10:00, hold to Fri",(0,8,0.3,True,"WEEK Mon 10:00 | HOLD",False)),
   ("10:00 entry, hold 120h",(0,8,0.3,False,"TIME 120h WF | HOLD | entries 10:00 only",False)),
   ("ORIGINAL Pine (13-20, freeze 20:00), always in",(13,20,0.11,False,"CONT | SAR",False))]
out=[]
for name,a in C:
    r=daily_pct(*a); r=r[np.abs(r)>0] if False else r
    row=dict(name=name, sharpe_ann=round(dsr(r,2)[1],2))
    for N in (10,100,1000,10000): row[f"DSR_N{N}"]=round(dsr(r,N)[0],3)
    out.append(row); print(row)
json.dump(out,open('dsr.json','w'),indent=1)
