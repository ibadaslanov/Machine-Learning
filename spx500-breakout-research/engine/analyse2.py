import json, numpy as np, re, sys
COST = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
res = np.load("grid2_res.npy", mmap_mode="r"); combos = np.load("grid2_combos.npy"); labels = json.load(open("grid2_labels.json"))
NC, NJ, NF, _ = res.shape
gross = np.asarray(res[..., 0:8], np.float32); n = np.asarray(res[..., 8:16], np.float32); fin = np.asarray(res[..., 16:24], np.float32)
ffinIS = np.asarray(res[..., 24]); ffinOOS = np.asarray(res[..., 25])
net_y = gross - COST * n - fin                       # (NC,NJ,NF,8)
fade_y = -gross - COST * n                           # financing for fades only known for IS/OOS totals
IS = slice(0, 4); OOS = slice(4, 8)
netIS = net_y[..., IS].sum(-1); netOOS = net_y[..., OOS].sum(-1)
fIS = fade_y[..., IS].sum(-1) - ffinIS; fOOS = fade_y[..., OOS].sum(-1) - ffinOOS
nIS = n[..., IS].sum(-1).min(-1); nOOS = n[..., OOS].sum(-1).min(-1)
rIS, rOOS = netIS.min(-1), netOOS.min(-1)         # robust: worst double-break rule
frIS, frOOS = fIS.min(-1), fOOS.min(-1)
Fh = combos[:, 1]
valid = (nIS >= 150) & (nOOS >= 75) & ~np.isin(Fh, [21, 22])[:, None]


def fam(lbl):
    if lbl.startswith("CONT"): return lbl.split(" |")[0]
    if lbl.startswith("DAY"):
        p = lbl.split(" | "); return f"DAY | {p[1]} | {'window 1h' if '+1h' in p[2] else 'window 4h' if '+4h' in p[2] else 'any hour'}"
    if lbl.startswith("WEEK"): return f"WEEK | {lbl.split(' | ')[1]}"
    p = lbl.split(" | "); N = int(re.search(r"TIME (\d+)h", p[0]).group(1))
    return f"TIME {N}h | {p[1]} | {'1-hour window' if 'only' in p[2] else 'any hour'}"


fams = [fam(l) for l in labels]; famset = sorted(set(fams))


def desc(c, j, fade=False):
    S, F, b, a = combos[c]
    buf = "no buffer" if b == 0 else f"buf {int(round(b*100))}% {'always' if a else 'overnight'}"
    return f"{'FADE ' if fade else ''}S={int(S):02d} F={int(F):02d} {buf} | {labels[j]}"


def rank(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


def row(c, j, fade=False):
    yrs = (-(gross[c, j]) - COST * n[c, j]) if fade else net_y[c, j]  # per fb x year
    worst = yrs.min(0)
    return dict(strategy=desc(c, j, fade), IS=round(float((frIS if fade else rIS)[c, j])), OOS=round(float((frOOS if fade else rOOS)[c, j])),
                trades_IS=int(nIS[c, j]), trades_OOS=int(nOOS[c, j]),
                fin_total=round(float((fin[c, j].sum(-1)).max())), by_year_worst_fb={str(2019 + k): round(float(v)) for k, v in enumerate(worst)},
                ddIS=round(float(res[c, j, :, 26].max())), ddOOS=round(float(res[c, j, :, 27].max())))


out = dict(cost=COST)
U = valid.sum(); out["eligible"] = int(U)
for nm, (a, b) in dict(normal=(rIS, rOOS), fade=(frIS, frOOS)).items():
    A, B = a[valid], b[valid]
    out[f"{nm}_share_pos_IS"] = float((A > 0).mean()); out[f"{nm}_share_pos_OOS"] = float((B > 0).mean())
    out[f"{nm}_share_pos_both"] = float(((A > 0) & (B > 0)).mean())
    out[f"{nm}_spearman"] = float(np.corrcoef(rank(A), rank(B))[0, 1])

# per-family stats (normal + fade)
fam_idx = {f: np.array([j for j in range(NJ) if fams[j] == f]) for f in famset}
fs = {}
for f, js in fam_idx.items():
    v = valid[:, js]
    if v.sum() < 10: continue
    A = rIS[:, js][v]; B = rOOS[:, js][v]; FA = frIS[:, js][v]; FB = frOOS[:, js][v]
    sub = np.where(v, rIS[:, js], -np.inf); c, jj = np.unravel_index(np.argmax(sub), sub.shape)
    fsub = np.where(v, frIS[:, js], -np.inf); fc, fjj = np.unravel_index(np.argmax(fsub), fsub.shape)
    fs[f] = dict(n=int(v.sum()), med_IS=round(float(np.median(A))), med_OOS=round(float(np.median(B))), share_OOS_pos=round(float((B > 0).mean()), 3),
                 spearman=round(float(np.corrcoef(rank(A), rank(B))[0, 1]), 3) if len(A) > 2 else None,
                 best=row(c, js[jj]), fade_med_OOS=round(float(np.median(FB))), fade_best=row(fc, js[fjj], True))
out["families"] = fs

# top-k by IS (normal and fade pooled, deduplicated), OOS reported for all of them
cand_score = np.concatenate([np.where(valid, rIS, -np.inf).ravel(), np.where(valid, frIS, -np.inf).ravel()])
order = np.argsort(cand_score)[::-1]
seen = set(); top = []
for k in order:
    if len(top) >= 50: break
    fade = k >= NC * NJ; kk = k - NC * NJ if fade else k; c, j = divmod(kk, NJ)
    key = (round(float((frIS if fade else rIS)[c, j]), 1), round(float((frOOS if fade else rOOS)[c, j]), 1))
    if key in seen: continue
    seen.add(key); top.append(row(c, j, fade))
out["top50_by_IS"] = top
o = np.array([t["OOS"] for t in top])
for N in (10, 20, 50):
    out[f"top{N}_OOS_mean"] = float(o[:N].mean()); out[f"top{N}_OOS_median"] = float(np.median(o[:N])); out[f"top{N}_OOS_share_pos"] = float((o[:N] > 0).mean())

# walk-forward: choose on the 2 previous years (worst fb), trade the next year; candidates = all eligible normal + fade
wf = []
for ty in range(2, 8):
    tr = slice(ty - 2, ty)
    sN = np.where(valid, net_y[..., tr].sum(-1).min(-1), -np.inf)
    sF = np.where(valid, (fade_y[..., tr].sum(-1)).min(-1), -np.inf)  # fade financing not per-year: ignore in selection (small)
    for nm, s, test in (("normal", sN, net_y[..., ty].min(-1)), ("fade", sF, fade_y[..., ty].min(-1))):
        flat = np.argsort(s, axis=None)[::-1][:20]
        cj = [np.unravel_index(k, s.shape) for k in flat]
        tv = np.array([test[c, j] for c, j in cj])
        wf.append(dict(test_year=2019 + ty, pool=nm, best=desc(*cj[0], nm == "fade"), best_train=round(float(s[cj[0]])),
                       best_test=round(float(tv[0])), top20_test_mean=round(float(tv.mean())), top20_test_share_pos=float((tv > 0).mean())))
out["walk_forward"] = wf
json.dump(out, open(f"analysis2_c{COST}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (list, dict))}, indent=1))
