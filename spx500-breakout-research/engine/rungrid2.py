import engine2 as E2, numpy as np, time, json, sys
x=E2.Ctx(); d=x.d; cfg=x.cfg
c_cyc=np.array([c[0] for c in cfg]); c_N=np.array([c[1] for c in cfg]); c_em=np.array([c[2] for c in cfg]); c_E=np.array([c[3] for c in cfg]); c_L=np.array([c[4] for c in cfg])
combos=E2.combos2()
if len(sys.argv)>1: combos=combos[:int(sys.argv[1])]
args=(d['O'],d['H'],d['L'],d['C'],d['hr'],d['eh'],d['yidx'],d['reopen'],d['nights'],d['rate'])
t=time.time()
res=E2.grid(*args,combos,x.starts,x.endcs,x.actives,c_cyc,c_N,c_em,c_E,c_L,3,2,0.5,E2.MK/100)
print('done',res.shape,round(time.time()-t),'s',flush=True)
if len(sys.argv)==1:
    np.save('grid2_res.npy',res); np.save('grid2_combos.npy',combos); json.dump([c[5] for c in cfg],open('grid2_labels.json','w'))
