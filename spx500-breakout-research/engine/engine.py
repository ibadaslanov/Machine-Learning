"""Generalised version of the 'Session + Frozen 20:00 Breakout v2' Pine logic.

Level logic (identical to the Pine script):
  - in session (hours S..F, wrapping past midnight allowed): levels = previous candle
    high/low (+ buffer if applyTo == Always)
  - outside session: levels frozen from the last candle whose hour == F, +/- buffer * its range
  - double-break candles: gap through a level decides; otherwise a fallback rule

Trading-period modes (new):
  CONT          always in the market (original script)
  CONT_WF       always in the market, but flat over the weekend (close Friday's last candle)
  DAY(R, wf)    24-hour cycles starting at hour R: at the first candle of a new cycle the
                position is closed at that candle's open; wf=1 also closes at Friday's last close
  WEEK(D, R)    starts on weekday D at hour R, runs until the week is over (close of the
                last candle before the weekend); flat until D/R next week

Entry modes (new):
  SAR           stop-and-reverse on every break (original)
  FIRST_HOLD    first break in the cycle sets the direction, held until the cycle ends
  FIRST_STOP    first break enters; a break of the opposite level exits flat (at that level);
                no more trades until the next cycle
"""
import csv, os, datetime as dt
import numpy as np
from numba import njit, prange

F_CSV = os.environ.get("SPX_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "FOREXCOM_SPX500_60.csv"))

SAR, FIRST_HOLD, FIRST_STOP = 0, 1, 2
ENTRY_NAMES = ["SAR", "FIRST_HOLD", "FIRST_STOP"]
FB_AGAINST, FB_COLOUR, FB_NEAREST, FB_FAVOUR = 0, 1, 2, 3
FB_NAMES = ["against", "colour", "nearest", "favour"]
DAYNAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]
SPLIT_YEAR = 2023  # trades closed before 2023 = in-sample, from 2023 = out-of-sample


def load():
    rows = list(csv.DictReader(open(F_CSV)))
    O = np.array([float(r["open"]) for r in rows]); H = np.array([float(r["high"]) for r in rows])
    L = np.array([float(r["low"]) for r in rows]); C = np.array([float(r["close"]) for r in rows])
    ts = [dt.datetime(int(r["Year"]), int(r["month"]), int(r["day"]), int(r["hour"]), tzinfo=dt.timezone.utc) for r in rows]
    eh = np.array([int(t.timestamp()) // 3600 for t in ts], dtype=np.int64)  # epoch hours
    hr = np.array([t.hour for t in ts], dtype=np.int64)
    year = np.array([t.year for t in ts], dtype=np.int64)
    return dict(O=O, H=H, L=L, C=C, eh=eh, hr=hr, year=year, ts=ts)


def period_modes(d):
    """Return list of (name, cyc_start, cyc_end_close, active) boolean arrays."""
    eh = d["eh"]; n = len(eh)
    sat0 = 2 * 24 + 4 * 24  # 1970-01-01 is Thursday; Saturday 00:00 UTC = 48h later ... computed below
    # epoch hour of a Saturday 00:00 UTC: 1970-01-03 00:00 -> 48
    sat0 = 48
    week = (eh - sat0) // 168
    wpos = (eh - sat0) % 168  # hours since Saturday 00:00 UTC (Sunday 22:00 = 46, Monday 00:00 = 48)
    last_of_week = np.zeros(n, bool); last_of_week[:-1] = week[1:] != week[:-1]
    first_of_week = np.zeros(n, bool); first_of_week[1:] = week[1:] != week[:-1]
    F = np.zeros(n, bool); T = np.ones(n, bool)
    modes = [("CONT", F, F, T), ("CONT_WF", F, last_of_week, T)]
    for wf in (0, 1):
        for R in range(24):
            cyc = (eh - R) // 24
            start = np.zeros(n, bool); start[1:] = cyc[1:] != cyc[:-1]
            modes.append((f"DAY R={R:02d}{' WF' if wf else ''}", start, last_of_week if wf else F, T))
    for D in range(6):  # Sun..Fri
        for R in range(24):
            thr = 24 + 24 * D + R  # Sunday 00:00 = 24h after Saturday 00:00
            active = wpos >= thr
            modes.append((f"WEEK {DAYNAMES[D]} {R:02d}:00", first_of_week, last_of_week, active))
    return modes


@njit(cache=True)
def levels(H, L, hr, S, Fh, b, always):
    n = len(H)
    lh = np.full(n, np.nan); ll = np.full(n, np.nan)
    fh = np.nan; fl = np.nan
    for i in range(n):
        h = hr[i]
        if S <= Fh:
            ins = S <= h and h <= Fh
        else:
            ins = h >= S or h <= Fh
        if ins:
            if i > 0:
                r = H[i - 1] - L[i - 1]
                bb = b * r if always else 0.0
                lh[i] = H[i - 1] + bb; ll[i] = L[i - 1] - bb
        else:
            lh[i] = fh; ll[i] = fl
        if h == Fh:
            r = H[i] - L[i]
            fh = H[i] + b * r; fl = L[i] - b * r
    return lh, ll


@njit(cache=True)
def simulate(O, H, L, C, lh, ll, start, endc, active, emode, fb, isIS, cost_dd, out_trades):
    """Returns (netIS, netOOS, nIS, nOOS, maxDD at cost_dd). If out_trades is non-empty
    (shape (k,4)), fills it with (exit bar, dir, entry px, exit px) and returns count in [5]."""
    n = len(O)
    pos = 0; entry = 0.0; done = False
    netIS = 0.0; netOOS = 0.0; nIS = 0; nOOS = 0
    eq = 0.0; peak = 0.0; dd = 0.0; k = 0
    rec = out_trades.shape[0] > 0
    sd = np.zeros(2, np.int64); sp = np.zeros(2)
    for i in range(n):
        if start[i]:
            if pos != 0:
                p = (O[i] - entry) * pos
                if isIS[i]:
                    netIS += p; nIS += 1
                else:
                    netOOS += p; nOOS += 1
                eq += p - cost_dd; peak = max(peak, eq); dd = max(dd, peak - eq)
                if rec:
                    out_trades[k, 0] = i; out_trades[k, 1] = pos; out_trades[k, 2] = entry; out_trades[k, 3] = O[i]; k += 1
                pos = 0
            done = False
        if active[i] and not (done and emode != SAR):
            a, z = lh[i], ll[i]
            up = (a == a) and H[i] > a
            dn = (z == z) and L[i] < z
            ns = 0
            if up and dn:
                first = 0
                if O[i] > a:
                    first = 1
                elif O[i] < z:
                    first = -1
                else:
                    pg = 1 if (i > 0 and C[i - 1] >= O[i - 1]) else -1
                    if fb == 0:
                        first = -pos if pos != 0 else pg
                    elif fb == 1:
                        first = pg
                    elif fb == 2:
                        first = 1 if (a - O[i]) < (O[i] - z) else -1
                    else:
                        first = pos if pos != 0 else pg
                if first == 1:
                    sd[0] = 1; sp[0] = max(a, O[i]); sd[1] = -1; sp[1] = z
                else:
                    sd[0] = -1; sp[0] = min(z, O[i]); sd[1] = 1; sp[1] = a
                ns = 2
            elif up:
                sd[0] = 1; sp[0] = max(a, O[i]); ns = 1
            elif dn:
                sd[0] = -1; sp[0] = min(z, O[i]); ns = 1
            for s in range(ns):
                d = sd[s]; px = sp[s]
                close_it = False; open_dir = 0
                if emode == SAR:
                    if d != pos:
                        close_it = pos != 0; open_dir = d
                elif emode == FIRST_HOLD:
                    if pos == 0 and not done:
                        open_dir = d
                else:  # FIRST_STOP
                    if pos == 0 and not done:
                        open_dir = d
                    elif pos != 0 and d != pos:
                        close_it = True
                if close_it:
                    p = (px - entry) * pos
                    if isIS[i]:
                        netIS += p; nIS += 1
                    else:
                        netOOS += p; nOOS += 1
                    eq += p - cost_dd; peak = max(peak, eq); dd = max(dd, peak - eq)
                    if rec:
                        out_trades[k, 0] = i; out_trades[k, 1] = pos; out_trades[k, 2] = entry; out_trades[k, 3] = px; k += 1
                    pos = 0
                if open_dir != 0:
                    pos = open_dir; entry = px
                    if emode != SAR:
                        done = True
        if endc[i] and pos != 0:
            p = (C[i] - entry) * pos
            if isIS[i]:
                netIS += p; nIS += 1
            else:
                netOOS += p; nOOS += 1
            eq += p - cost_dd; peak = max(peak, eq); dd = max(dd, peak - eq)
            if rec:
                out_trades[k, 0] = i; out_trades[k, 1] = pos; out_trades[k, 2] = entry; out_trades[k, 3] = C[i]; k += 1
            pos = 0
            done = True
    return netIS, netOOS, nIS, nOOS, dd, k


@njit(parallel=True, cache=True)
def grid(O, H, L, C, hr, isIS, combos, starts, endcs, actives, cfg_mode, cfg_entry, cost_dd):
    nc = combos.shape[0]; ncfg = cfg_mode.shape[0]
    res = np.zeros((nc, ncfg, 4, 5), np.float32)
    empty = np.zeros((0, 4))
    for c in prange(nc):
        S = int(combos[c, 0]); Fh = int(combos[c, 1]); b = combos[c, 2]; al = combos[c, 3] > 0.5
        lh, ll = levels(H, L, hr, S, Fh, b, al)
        for j in range(ncfg):
            m = cfg_mode[j]
            for fb in range(4):
                r = simulate(O, H, L, C, lh, ll, starts[m], endcs[m], actives[m], cfg_entry[j], fb, isIS, cost_dd, empty)
                res[c, j, fb, 0] = r[0]; res[c, j, fb, 1] = r[1]; res[c, j, fb, 2] = r[2]; res[c, j, fb, 3] = r[3]; res[c, j, fb, 4] = r[4]
    return res


def build(d):
    modes = period_modes(d)
    names = [m[0] for m in modes]
    starts = np.array([m[1] for m in modes]); endcs = np.array([m[2] for m in modes]); actives = np.array([m[3] for m in modes])
    cfg = []
    for mi, nm in enumerate(names):
        for e in ((SAR,) if nm.startswith("CONT") else (SAR, FIRST_HOLD, FIRST_STOP)):
            cfg.append((mi, e))
    cfg_mode = np.array([c[0] for c in cfg], np.int64); cfg_entry = np.array([c[1] for c in cfg], np.int64)
    return names, starts, endcs, actives, cfg_mode, cfg_entry


BUFFERS = [(0.0, 0)] + [(b, a) for b in (0.05, 0.11, 0.2, 0.3, 0.5) for a in (0, 1)]


def all_combos():
    return np.array([(S, Fh, b, a) for S in range(24) for Fh in range(24) for (b, a) in BUFFERS], float)


def run_one(d, S, Fh, b, always, mode_name, emode, fb, cost_dd=0.5, ctx=None):
    names, starts, endcs, actives, _, _ = ctx or build(d)
    m = names.index(mode_name)
    lh, ll = levels(d["H"], d["L"], d["hr"], S, Fh, b, always)
    isIS = d["year"] < SPLIT_YEAR
    out = np.zeros((40000, 4))
    r = simulate(d["O"], d["H"], d["L"], d["C"], lh, ll, starts[m], endcs[m], actives[m], emode, fb, isIS, cost_dd, out)
    return r, out[: r[5]]
