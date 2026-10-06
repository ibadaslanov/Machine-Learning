"""Trade-management variants for the session breakout (same base rules as all_sessions.pine).

Each session time is simulated on its own, day by day:
- the session candle breaks the previous candle's high/low (touch fills, nearest extreme first)
- after the session candle closes: reverse at its low/high, up to max_rev times,
  then the next cross closes the trade (close_after) or it is held
- exit at the next day's session open
Variants add: take profit (tp, in R = reference candle range), adding units every
add_step R in profit (pyramiding), shorter holding time (hold_min), break-even stop
(be: after be R in profit the stop goes to entry), trailing stop (trail R behind the best price).
"""
import numpy as np
import pandas as pd

BASE = dict(max_rev=4, close_after=True, tp=None, adds=0, add_step=2.0, hold_min=None, be=None, trail=None)


def session_days(bars, hhmm):
    """(session bar, next day's session bar) pairs. Works for any bar size: the reference
    candle is the bar right before the session bar."""
    t = bars.index
    step = t.to_series().diff().mode().iloc[0]
    mins = t.hour * 60 + t.minute
    sm = int(hhmm[:2]) * 60 + int(hhmm[3:])
    sig = np.flatnonzero(mins == sm)
    prev_ok = np.zeros(len(sig), bool)
    for n, i in enumerate(sig):
        prev_ok[n] = i > 0 and (t[i] - t[i - 1]) == step
    days = []
    for n in range(len(sig)):  # the last day ends with the data; its open trade is not counted
        if prev_ok[n]:
            days.append((sig[n], sig[n + 1] if n + 1 < len(sig) else len(bars)))
    return days


def sim_day(o, h, l, c, tns, i0, i1, v):
    H, L = h[i0 - 1], l[i0 - 1]
    R = max(H - L, 1e-9)
    t_exit = tns[i0] + v["hold_min"] * 60_000_000_000 if v["hold_min"] else None
    st = dict(pos=0, units=[], revs=0, win=True, sigH=None, sigL=None, best=None, adds=0,
              buy=H, sell=L, tp=None, add=None, done=False)
    trades = []  # (pnl points summed over units, units, exit bar)

    def close_all(px, i):
        p = st["pos"]
        trades.append((sum(p * (px - e) for e in st["units"]), len(st["units"]), i, p))
        st["pos"], st["units"] = 0, []

    def finish():
        st.update(done=True, buy=None, sell=None, tp=None, add=None)

    def open_pos(side, px):
        st.update(pos=side, units=[px], best=px, adds=0, buy=None, sell=None)
        st["tp"] = px + side * v["tp"] * R if v["tp"] else None
        st["add"] = px + side * v["add_step"] * R if v["adds"] else None

    def exit_level():
        """Effective protective level and whether it reverses."""
        p, e = st["pos"], st["units"][0]
        cands = []
        if not st["win"] and (st["revs"] < v["max_rev"] or v["close_after"]):
            cands.append((st["sigL"] if p > 0 else st["sigH"], st["revs"] < v["max_rev"]))
        if v["be"] and p * (st["best"] - e) >= v["be"] * R:
            cands.append((e, False))
        if v["trail"] and p * (st["best"] - e) >= v["trail"] * R:
            cands.append((st["best"] - p * v["trail"] * R, False))
        if not cands:
            return None, False
        # the level closest to price (highest for a long, lowest for a short) is hit first
        lv, rev = max(cands, key=lambda x: p * x[0])
        return lv, rev

    def orders():
        """(level, kind, direction it triggers on: +1 = price rising to it, -1 = falling)"""
        out = []
        if st["pos"] == 0:
            if st["buy"] is not None: out.append((st["buy"], "entry_long", 1))
            if st["sell"] is not None: out.append((st["sell"], "entry_short", -1))
            return out
        p = st["pos"]
        if st["tp"] is not None: out.append((st["tp"], "tp", p))
        if st["add"] is not None: out.append((st["add"], "add", p))
        lv, rev = exit_level()
        if lv is not None: out.append((lv, "rev" if rev else "stop", -p))
        return out

    def fire(kind, px, i):
        if kind == "entry_long": open_pos(1, px)
        elif kind == "entry_short": open_pos(-1, px)
        elif kind == "tp": close_all(px, i); finish()
        elif kind == "add":
            st["units"].append(px); st["adds"] += 1
            st["add"] = px + st["pos"] * v["add_step"] * R if st["adds"] < v["adds"] else None
        elif kind == "rev":
            side = -st["pos"]; close_all(px, i); st["revs"] += 1; open_pos(side, px)
        elif kind == "stop":
            close_all(px, i); finish()

    for i in range(i0, i1):
        if st["done"]:
            break
        if t_exit is not None and tns[i] >= t_exit:
            if st["pos"]: close_all(o[i], i)
            finish(); break
        pA = h[i] if h[i] - o[i] <= o[i] - l[i] else l[i]
        pB = l[i] if pA == h[i] else h[i]
        # gaps: anything the open is already beyond fills at the open
        for _ in range(6):
            hit = [x for x in orders() if (x[2] > 0 and o[i] >= x[0]) or (x[2] < 0 and o[i] <= x[0])]
            if not hit or st["done"]: break
            fire(hit[0][1], o[i], i)
        px = o[i]
        for target in (pA, pB, c[i]):
            for _ in range(12):
                if st["done"]: break
                d = 1 if target > px else -1 if target < px else 0
                if d == 0: break
                hit = [x for x in orders() if x[2] == d and min(px, target) <= x[0] <= max(px, target)]
                if not hit: break
                lv, kind, _ = min(hit, key=lambda x: abs(x[0] - px))
                px = lv
                fire(kind, lv, i)
            if st["pos"]:
                st["best"] = max(st["best"], target) if st["pos"] > 0 else min(st["best"], target)
            px = target
        if i == i0:  # session candle closed
            st.update(win=False, sigH=h[i0], sigL=l[i0])
            if st["pos"] == 0:
                finish(); break
    else:
        if st["pos"] and not st["done"] and i1 < len(o):
            close_all(o[i1], i1)
    return trades


def run(bars, times, v, cost):
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    tns = bars.index.asi8 if bars.index.dtype == "datetime64[ns, UTC]" else bars.index.as_unit("ns").asi8
    rows = []
    for s in times:
        for i0, i1 in session_days(bars, s):
            for pnl, units, ie, side in sim_day(o, h, l, c, tns, i0, i1, v):
                rows.append((s, bars.index[i0], bars.index[min(ie, len(o) - 1)], "long" if side > 0 else "short",
                             pnl - cost * units, units))
    return pd.DataFrame(rows, columns=["time", "session", "exit", "side", "net", "units"])


def stats(tr):
    if tr.empty:
        return dict(trades=0, net=0.0, pf=np.nan, dd=0.0)
    eq = tr.sort_values("exit")["net"].cumsum()
    w, ls = tr.net[tr.net > 0].sum(), -tr.net[tr.net <= 0].sum()
    return dict(trades=len(tr), net=tr.net.sum(), pf=w / ls if ls else np.nan, dd=(eq - eq.cummax()).min())


VARIANTS = (
    [("base", {})]
    + [(f"max reversals {n}", dict(max_rev=n)) for n in (0, 1, 2, 3, 6, 8)]
    + [(f"max reversals {n}, then hold", dict(max_rev=n, close_after=False)) for n in (2, 4, 8)]
    + [(f"take profit {k}R", dict(tp=k)) for k in (3, 5, 8, 12, 20, 30)]
    + [(f"hold max {m // 60}h", dict(hold_min=m)) for m in (60, 120, 240, 480, 720)]
    + [(f"add {n} every {s:g}R", dict(adds=n, add_step=s)) for n, s in ((1, 2), (2, 2), (3, 2), (1, 4), (2, 4), (3, 4))]
    + [(f"break-even after {k}R", dict(be=k)) for k in (3, 5, 8)]
    + [(f"trailing stop {k}R", dict(trail=k)) for k in (5, 8, 12, 20)]
)


def main():
    import argparse

    from .data import load_ohlc

    ap = argparse.ArgumentParser(description="Compare trade-management variants: tune on the part before --split, "
                                             "judge on the part after it.")
    ap.add_argument("csv", help="OHLC bars (5-minute, hourly, ...)")
    ap.add_argument("--sessions", required=True, help="comma-separated session times (UTC unless --data-tz)")
    ap.add_argument("--split", required=True, help="date that separates the tuning and test periods, YYYY-MM-DD")
    ap.add_argument("--cost", type=float, default=0.5, help="cost per unit in points (default 0.5)")
    ap.add_argument("--data-tz", default="UTC")
    args = ap.parse_args()

    bars = load_ohlc(args.csv, args.data_tz)
    times = [s.strip() for s in args.sessions.split(",")]
    first, second = bars[bars.index < args.split], bars[bars.index >= args.split]
    rows = []
    for name, change in VARIANTS:
        v = dict(BASE, **change)
        a, b = (stats(run(x, times, v, args.cost)) for x in (first, second))
        rows.append((name, a["net"], a["net"] / -a["dd"] if a["dd"] else np.nan, a["pf"],
                     b["net"], b["net"] / -b["dd"] if b["dd"] else np.nan, b["pf"]))
    cols = ["variant", "tune net", "tune net/DD", "tune PF", "test net", "test net/DD", "test PF"]
    print(f"sessions {times} | cost {args.cost} per unit | tune < {args.split} <= test")
    print(pd.DataFrame(rows, columns=cols).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
