import csv, sys, math
from collections import defaultdict

import os
F = os.environ.get("SPX_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "FOREXCOM_SPX500_60.csv"))
rows = list(csv.DictReader(open(F)))
Y = [int(r["Year"]) for r in rows]; H = [int(r["hour"]) for r in rows]
O = [float(r["open"]) for r in rows]; Hi = [float(r["high"]) for r in rows]
Lo = [float(r["low"]) for r in rows]; C = [float(r["close"]) for r in rows]
UPOS = [r["Position"] for r in rows]


def run(sessStart=13, freezeHr=20, bufPct=0.11, applyTo="Overnight only",
        fallback="worst", cost=0.0, log=False):
    frzH = frzL = None
    pos = 0; entry = None
    trades = []  # (bar, year, pnl)
    dbl = dbl_gap = dbl_fb = 0
    states = []
    for i in range(len(rows)):
        h = H[i]
        insess = sessStart <= h <= freezeHr
        if insess and i > 0:
            r = Hi[i-1] - Lo[i-1]
            b = bufPct * r if applyTo == "Always" else 0.0
            lvlH, lvlL = Hi[i-1] + b, Lo[i-1] - b
        elif insess:
            lvlH = lvlL = None
        else:
            lvlH, lvlL = frzH, frzL
        prevGreen = i > 0 and C[i-1] >= O[i-1]
        up = lvlH is not None and Hi[i] > lvlH
        dn = lvlL is not None and Lo[i] < lvlL
        steps = []
        if up and dn:
            dbl += 1
            if O[i] > lvlH: first = 1; dbl_gap += 1
            elif O[i] < lvlL: first = -1; dbl_gap += 1
            else:
                dbl_fb += 1  # no 1-minute data in the file
                if fallback == "worst":
                    first = -1 if pos == 1 else 1 if pos == -1 else (1 if prevGreen else -1)
                else:
                    first = 1 if prevGreen else -1
            pxUp, pxDn = max(lvlH, O[i]), min(lvlL, O[i])
            steps = [(1, pxUp), (-1, lvlL)] if first == 1 else [(-1, pxDn), (1, lvlH)]
        elif up:
            steps = [(1, max(lvlH, O[i]))]
        elif dn:
            steps = [(-1, min(lvlL, O[i]))]
        for d, px in steps:
            if d != pos:
                if pos != 0:
                    trades.append((i, Y[i], (px - entry) * pos - cost))
                pos, entry = d, px
        states.append(pos)
        if h == freezeHr:
            rng = Hi[i] - Lo[i]
            frzH, frzL = Hi[i] + bufPct * rng, Lo[i] - bufPct * rng
    return trades, dbl, dbl_gap, dbl_fb, pos, states


def summary(trades):
    n = len(trades); pl = [t[2] for t in trades]
    gw = sum(p for p in pl if p > 0); gl = -sum(p for p in pl if p <= 0)
    eq = mdd = peak = 0.0
    for p in pl:
        eq += p; peak = max(peak, eq); mdd = max(mdd, peak - eq)
    return dict(trades=n, net=sum(pl), win=100*sum(p > 0 for p in pl)/n,
                pf=gw/gl, avg=sum(pl)/n, maxdd=mdd)


if __name__ == "__main__":
    for fb in ["worst", "colour"]:
        tr, dbl, gap, fbn, pos, st = run(fallback=fb)
        s = summary(tr)
        print(fb, {k: round(v, 2) for k, v in s.items()}, "dbl", dbl, "gap-resolved", gap, "fallback", fbn, "open", pos)
        by = defaultdict(list)
        for _, y, p in tr: by[y].append(p)
        print("  per year:", {y: round(sum(v), 1) for y, v in sorted(by.items())})
        # breakeven cost per trade
        print("  breakeven cost/trade:", round(s["net"]/s["trades"], 2))
        for c in [0.5, 1.0]:
            print(f"  cost {c}:", round(summary(run(fallback=fb, cost=c)[0])["net"], 1))
    # compare with the file's own Position column (final state per bar)
    _, *_, st = run(fallback="worst")
    m = {"long": 1, "short": -1, "short and then long": 1, "long and then short": -1}
    agree = sum(1 for a, b in zip(st, UPOS) if m[b] == a)
    print("agreement with file Position column (final state):", agree, "/", len(st))
    dbl_file = [i for i, b in enumerate(UPOS) if "then" in b]
    print("file double-break rows:", len(dbl_file))
    # buy & hold reference
    print("buy&hold points:", round(C[-1] - O[0], 1))
