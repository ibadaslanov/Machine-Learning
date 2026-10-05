"""Engine v2 for the 'Session + Frozen 20:00 Breakout v2' strategy family.

Levels: identical to the Pine script (session S..F in UTC, previous-candle levels in session,
frozen F-candle levels +/- buffer outside the session, Pine's prevGreen uses close[1] >= open[1]).

Changes vs v1 (after independent verification):
  * FIRST_STOP now stops out on any later candle (v1 bug).
  * Reopen candles (>= 2h after the previous candle) whose recorded open equals the previous
    close have an unknown real open (the feed hides the gap). reopen_mode:
        0 = trust the recorded open (optimistic)
        1 = fills on those candles at the candle's worst extreme (H for buys, L for sells)
        2 = no orders on those candles (cancel stops into the break, re-place after the first candle)
  * Overnight CFD financing: every 21:00 UTC rollover crossed with an open position costs
    price * (rate + MK)/365 for a long and price * (MK - rate)/365 for a short (MK = 2.5%,
    rate = approx. Fed funds by year). Stored separately so results can be shown with/without.
  * New exits: TIME_N (close N hours after entry, at the first candle at/after entry+N h).
  * New entry window: entries only on candles whose hour is in [E, E+Lw) (mod 24).
  * Per-year gross P&L, trade count and financing are stored; costs applied in analysis.
"""
import numpy as np
from numba import njit, prange
import engine as E1

SAR, HOLD, STOP = 0, 1, 2
ENTRY_NAMES = ["SAR", "HOLD", "STOP"]
FB_NAMES = ["against", "colour", "nearest", "favour"]
YEARS = np.arange(2019, 2027)
RATE = {2019: 2.16, 2020: 0.38, 2021: 0.08, 2022: 1.68, 2023: 5.02, 2024: 5.14, 2025: 4.2, 2026: 3.6}
MK = 2.5
DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]


def load():
    d = E1.load()
    eh = d["eh"]; n = len(eh)
    gap = np.zeros(n, np.int64); gap[1:] = eh[1:] - eh[:-1]
    reopen = np.zeros(n, np.bool_); reopen[1:] = (gap[1:] >= 2) & (np.abs(d["O"][1:] - d["C"][:-1]) < 1e-9)
    f = (eh - 21) // 24  # number of 21:00 UTC rollovers up to time eh
    nights = np.zeros(n, np.int64); nights[1:] = f[1:] - f[:-1]
    rate = np.array([RATE[y] for y in d["year"]]) / 100.0
    d.update(reopen=reopen, nights=nights, rate=rate, yidx=(d["year"] - 2019).astype(np.int64))
    return d


def cycles(d):
    eh = d["eh"]; n = len(eh)
    week = (eh - 48) // 168; wpos = (eh - 48) % 168
    low = np.zeros(n, np.bool_); low[:-1] = week[1:] != week[:-1]
    fow = np.zeros(n, np.bool_); fow[1:] = week[1:] != week[:-1]
    Fz = np.zeros(n, np.bool_); Tr = np.ones(n, np.bool_)
    cyc = [("CONT", Fz, Fz, Tr), ("CONT_WF", Fz, low, Tr), ("TIME_WF", Fz, low, Tr)]
    for R in range(24):
        c = (eh - R) // 24
        st = np.zeros(n, np.bool_); st[1:] = c[1:] != c[:-1]
        cyc.append((f"DAY R={R:02d} WF", st, low, Tr))
    cyc.append(("WEEK Sun-open", fow, low, Tr))
    for D in range(1, 6):
        for R in range(24):
            if D == 5 and R >= 21:
                continue  # never trades
            cyc.append((f"WEEK {DAYS[D]} {R:02d}:00", fow, low, wpos >= 24 + 24 * D + R))
    return cyc


TIME_NS = [1, 2, 4, 8, 12, 24, 48, 120]


def configs(cyc):
    """(cycle index, timeN, entry mode, window start E, window length Lw, label)."""
    names = [c[0] for c in cyc]; out = []
    out.append((names.index("CONT"), 0, SAR, 0, 24, "CONT | SAR"))
    out.append((names.index("CONT_WF"), 0, SAR, 0, 24, "CONT_WF | SAR"))
    for R in range(24):
        ci = names.index(f"DAY R={R:02d} WF")
        for em in (SAR, HOLD, STOP):
            for Lw in (1, 4, 24):
                out.append((ci, 0, em, R, Lw, f"DAY R={R:02d} WF | {ENTRY_NAMES[em]} | entries {'any hour' if Lw == 24 else f'{R:02d}:00+{Lw}h'}"))
    for ci, nm in enumerate(names):
        if nm.startswith("WEEK"):
            for em in (SAR, HOLD, STOP):
                out.append((ci, 0, em, 0, 24, f"{nm} | {ENTRY_NAMES[em]}"))
    ti = names.index("TIME_WF")
    for N in TIME_NS:
        for em in (SAR, HOLD, STOP):
            out.append((ti, N, em, 0, 24, f"TIME {N}h WF | {ENTRY_NAMES[em]} | entries any hour"))
            for E in range(24):
                out.append((ti, N, em, E, 1, f"TIME {N}h WF | {ENTRY_NAMES[em]} | entries {E:02d}:00 only"))
    return out


levels = E1.levels

NST = 3 * 8 + 4  # per-year gross, per-year n, per-year fin, + fadeFinIS, fadeFinOOS, ddIS, ddOOS


@njit(cache=True)
def simulate(O, H, L, C, hr, eh, yidx, reopen, nights, rate, lh, ll, start, endc, active,
             timeN, emode, wE, wL, fb, reopen_mode, dirmode, cost, mk, st, trades):
    """st: float64[NST] output. trades: (k,6) recorder [entry bar, exit bar, dir, entry px, exit px, financing] or empty.
    dirmode: 0 both, 1 long only, -1 short only. Returns number of trades recorded."""
    n = len(O)
    for q in range(st.shape[0]):
        st[q] = 0.0
    pos = 0; entry = 0.0; ebar = 0; T = 0; done = False; fin_tr = 0.0
    eqI = 0.0; pkI = 0.0; ddI = 0.0; eqO = 0.0; pkO = 0.0; ddO = 0.0
    rec = trades.shape[0] > 0; k = 0
    sd = np.zeros(2, np.int64); sp = np.zeros(2)
    for i in range(n):
        y = yidx[i]; isIS = y < 4
        # 1. financing for positions carried into this candle
        if pos != 0 and nights[i] > 0:
            base = C[i - 1] * nights[i] / 365.0
            f = base * (rate[i] + mk) if pos == 1 else base * (mk - rate[i])
            g = base * (mk - rate[i]) if pos == 1 else base * (rate[i] + mk)
            st[16 + y] += f; fin_tr += f
            if isIS:
                st[24] += g; eqI -= f
            else:
                st[25] += g; eqO -= f
        # 2. forced exits at the open: new cycle, or time exit
        if (start[i] or (timeN > 0 and pos != 0 and eh[i] >= T)) and pos != 0:
            px = O[i]
            if reopen[i] and reopen_mode == 1:
                px = L[i] if pos == 1 else H[i]
            p = (px - entry) * pos
            st[y] += p; st[8 + y] += 1
            if isIS:
                eqI += p - cost; pkI = max(pkI, eqI); ddI = max(ddI, pkI - eqI)
            else:
                eqO += p - cost; pkO = max(pkO, eqO); ddO = max(ddO, pkO - eqO)
            if rec:
                trades[k, 0] = ebar; trades[k, 1] = i; trades[k, 2] = pos; trades[k, 3] = entry; trades[k, 4] = px; trades[k, 5] = fin_tr; k += 1
            pos = 0
        if start[i]:
            done = False
        # 3. breaks
        blocked = reopen[i] and reopen_mode == 2
        can_open = active[i] and ((hr[i] - wE) % 24) < wL and not blocked
        evaluate = active[i] and not blocked and (emode == SAR or timeN > 0 or not done or pos != 0)
        if evaluate:
            a = lh[i]; z = ll[i]
            up = (a == a) and H[i] > a
            dn = (z == z) and L[i] < z
            ns = 0
            pess = reopen[i] and reopen_mode == 1
            if up and dn:
                first = 0
                if O[i] > a and not pess:
                    first = 1
                elif O[i] < z and not pess:
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
                    sd[0] = 1; sp[0] = H[i] if pess else max(a, O[i]); sd[1] = -1; sp[1] = L[i] if pess else z
                else:
                    sd[0] = -1; sp[0] = L[i] if pess else min(z, O[i]); sd[1] = 1; sp[1] = H[i] if pess else a
                ns = 2
            elif up:
                sd[0] = 1; sp[0] = H[i] if pess else max(a, O[i]); ns = 1
            elif dn:
                sd[0] = -1; sp[0] = L[i] if pess else min(z, O[i]); ns = 1
            for s in range(ns):
                dd_ = sd[s]; px = sp[s]
                close_it = False; open_dir = 0
                allowed = can_open and (dirmode == 0 or dirmode == dd_)
                if emode == SAR:
                    if dd_ != pos:
                        close_it = pos != 0
                        if allowed:
                            open_dir = dd_
                elif emode == HOLD:
                    if pos == 0 and not done and allowed:
                        open_dir = dd_
                else:  # STOP
                    if pos == 0 and not done and allowed:
                        open_dir = dd_
                    elif pos != 0 and dd_ != pos:
                        close_it = True
                if close_it:
                    p = (px - entry) * pos
                    st[y] += p; st[8 + y] += 1
                    if isIS:
                        eqI += p - cost; pkI = max(pkI, eqI); ddI = max(ddI, pkI - eqI)
                    else:
                        eqO += p - cost; pkO = max(pkO, eqO); ddO = max(ddO, pkO - eqO)
                    if rec:
                        trades[k, 0] = ebar; trades[k, 1] = i; trades[k, 2] = pos; trades[k, 3] = entry; trades[k, 4] = px; trades[k, 5] = fin_tr; k += 1
                    pos = 0
                if open_dir != 0:
                    pos = open_dir; entry = px; ebar = i; fin_tr = 0.0
                    T = eh[i] + timeN
                    if emode != SAR and timeN == 0:
                        done = True
        # 4. forced exit at the close (end of week)
        if endc[i]:
            if pos != 0:
                p = (C[i] - entry) * pos
                st[y] += p; st[8 + y] += 1
                if isIS:
                    eqI += p - cost; pkI = max(pkI, eqI); ddI = max(ddI, pkI - eqI)
                else:
                    eqO += p - cost; pkO = max(pkO, eqO); ddO = max(ddO, pkO - eqO)
                if rec:
                    trades[k, 0] = ebar; trades[k, 1] = i; trades[k, 2] = pos; trades[k, 3] = entry; trades[k, 4] = C[i]; trades[k, 5] = fin_tr; k += 1
                pos = 0
            if timeN == 0:
                done = True
    st[26] = ddI; st[27] = ddO
    return k


@njit(parallel=True, cache=True)
def grid(O, H, L, C, hr, eh, yidx, reopen, nights, rate, combos, starts, endcs, actives,
         c_cyc, c_N, c_em, c_E, c_L, nfb, reopen_mode, cost, mk):
    nc = combos.shape[0]; ncfg = c_cyc.shape[0]
    res = np.zeros((nc, ncfg, nfb, NST), np.float32)
    for c in prange(nc):
        S = int(combos[c, 0]); Fh = int(combos[c, 1]); b = combos[c, 2]; al = combos[c, 3] > 0.5
        lh, ll = levels(H, L, hr, S, Fh, b, al)
        st = np.zeros(NST); empty = np.zeros((0, 6))
        for j in range(ncfg):
            m = c_cyc[j]
            for fb in range(nfb):
                simulate(O, H, L, C, hr, eh, yidx, reopen, nights, rate, lh, ll, starts[m], endcs[m], actives[m],
                         c_N[j], c_em[j], c_E[j], c_L[j], fb, reopen_mode, 0, cost, mk, st, empty)
                for q in range(NST):
                    res[c, j, fb, q] = st[q]
    return res


BUFFERS2 = [(0.0, 0), (0.11, 0), (0.3, 0), (0.3, 1)]


def combos2():
    return np.array([(S, Fh, b, a) for S in range(24) for Fh in range(24) for (b, a) in BUFFERS2], float)


class Ctx:
    def __init__(self):
        self.d = load(); self.cyc = cycles(self.d); self.cfg = configs(self.cyc)
        self.names = [c[0] for c in self.cyc]
        self.starts = np.array([c[1] for c in self.cyc]); self.endcs = np.array([c[2] for c in self.cyc]); self.actives = np.array([c[3] for c in self.cyc])

    def run(self, S, Fh, b, always, label, fb=2, reopen_mode=2, dirmode=0, cost=0.5, mk=MK / 100, lh=None, ll=None):
        d = self.d
        j = [c[5] for c in self.cfg].index(label); ci, N, em, wE, wL, _ = self.cfg[j]
        if lh is None:
            lh, ll = levels(d["H"], d["L"], d["hr"], S, Fh, b, always)
        st = np.zeros(NST); tr = np.zeros((60000, 6))
        k = simulate(d["O"], d["H"], d["L"], d["C"], d["hr"], d["eh"], d["yidx"], d["reopen"], d["nights"], d["rate"], lh, ll,
                     self.starts[ci], self.endcs[ci], self.actives[ci], N, em, wE, wL, fb, reopen_mode, dirmode, cost, mk, st, tr)
        return st, tr[:k]
