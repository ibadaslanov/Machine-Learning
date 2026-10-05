"""Strategy A (daily cycle, first break, hold) with an optional take-profit and month filters.

Same rules and costs as engine2 (DAY R WF | HOLD, reopen_mode 2, financing), plus:
  tp_kind 0 = no take-profit, 1 = fixed points, 2 = multiple of the reference candle's range, 3 = percent of entry price
  (the candle that set the broken level: the frozen F candle outside the session, else the
  previous candle). The take-profit is a resting limit order: it is not checked on the entry
  candle itself (the order inside that hour is unknown, so this is conservative); on later
  candles it fills at the open if the candle opens beyond it, else at the TP price if touched.
  After a take-profit the cycle is done (no re-entry until the next cycle).
  tp_months[m]   : take-profit active for entries in month m (0..11)
  trade_months[m]: entries allowed in month m
"""
import numpy as np
from numba import njit
import engine2 as E2

TP_NONE, TP_POINTS, TP_RANGE, TP_PCT = 0, 1, 2, 3


@njit(cache=True)
def simulate_tp(O, H, L, C, hr, eh, mon, reopen, nights, rate, S, Fh, b, R, fb,
                tp_kind, tp_val, tp_months, trade_months, mk, out):
    """out: (k, 8) = entry bar, exit bar, dir, entry px, exit px, financing, reason (0 cycle, 1 week end, 2 TP), month."""
    n = len(O)
    frzH = np.nan; frzL = np.nan; frzR = np.nan
    pos = 0; entry = 0.0; ebar = 0; fin = 0.0; done = False; tp = np.nan; k = 0
    for i in range(n):
        h = hr[i]
        ins = (S <= h and h <= Fh) if S <= Fh else (h >= S or h <= Fh)
        if ins and i > 0:
            a = H[i - 1]; z = L[i - 1]; rr = H[i - 1] - L[i - 1]
        elif ins:
            a = np.nan; z = np.nan; rr = np.nan
        else:
            a = frzH; z = frzL; rr = frzR
        # financing
        if pos != 0 and nights[i] > 0:
            base = C[i - 1] * nights[i] / 365.0
            fin += base * (rate[i] + mk) if pos == 1 else base * (mk - rate[i])
        # new cycle: close at the open
        newc = i > 0 and (eh[i] - R) // 24 != (eh[i - 1] - R) // 24
        if newc and pos != 0:
            out[k, 0] = ebar; out[k, 1] = i; out[k, 2] = pos; out[k, 3] = entry; out[k, 4] = O[i]; out[k, 5] = fin; out[k, 6] = 0; out[k, 7] = mon[ebar]; k += 1
            pos = 0
        if newc:
            done = False
        # take-profit (resting limit), not on the entry candle
        if pos != 0 and tp == tp and i > ebar:
            px = np.nan
            if pos == 1:
                if O[i] >= tp:
                    px = O[i]
                elif H[i] >= tp:
                    px = tp
            else:
                if O[i] <= tp:
                    px = O[i]
                elif L[i] <= tp:
                    px = tp
            if px == px:
                out[k, 0] = ebar; out[k, 1] = i; out[k, 2] = pos; out[k, 3] = entry; out[k, 4] = px; out[k, 5] = fin; out[k, 6] = 2; out[k, 7] = mon[ebar]; k += 1
                pos = 0
        # first break of the cycle
        if pos == 0 and not done and not reopen[i] and trade_months[mon[i]]:
            up = (a == a) and H[i] > a
            dn = (z == z) and L[i] < z
            d = 0
            if up and dn:
                if O[i] > a:
                    d = 1
                elif O[i] < z:
                    d = -1
                else:
                    pg = 1 if C[i - 1] >= O[i - 1] else -1
                    if fb == 2:
                        d = 1 if (a - O[i]) < (O[i] - z) else -1
                    else:
                        d = pg
            elif up:
                d = 1
            elif dn:
                d = -1
            if d != 0:
                pos = d; ebar = i; fin = 0.0; done = True
                entry = max(a, O[i]) if d == 1 else min(z, O[i])
                tp = np.nan
                if tp_months[mon[i]]:
                    if tp_kind == 1:
                        tp = entry + d * tp_val
                    elif tp_kind == 2 and rr == rr:
                        tp = entry + d * tp_val * max(rr, 1.0)
                    elif tp_kind == 3:
                        tp = entry * (1.0 + d * tp_val / 100.0)
        # end of week: close at the close
        if i + 1 < n and (eh[i + 1] - 48) // 168 != (eh[i] - 48) // 168:
            if pos != 0:
                out[k, 0] = ebar; out[k, 1] = i; out[k, 2] = pos; out[k, 3] = entry; out[k, 4] = C[i]; out[k, 5] = fin; out[k, 6] = 1; out[k, 7] = mon[ebar]; k += 1
                pos = 0
            done = True
        # freeze after the F candle
        if h == Fh:
            r0 = H[i] - L[i]
            frzH = H[i] + b * r0; frzL = L[i] - b * r0; frzR = r0
    return k


class TP:
    def __init__(self):
        self.d = E2.load()
        self.mon = np.array([t.month - 1 for t in self.d["ts"]], np.int64)

    def run(self, S=15, F=8, b=0.0, R=9, fb=1, tp_kind=0, tp_val=0.0, tp_months=None, trade_months=None):
        d = self.d
        tm = np.ones(12, np.bool_) if tp_months is None else np.asarray(tp_months, np.bool_)
        trm = np.ones(12, np.bool_) if trade_months is None else np.asarray(trade_months, np.bool_)
        out = np.zeros((5000, 8))
        k = simulate_tp(d["O"], d["H"], d["L"], d["C"], d["hr"], d["eh"], self.mon, d["reopen"], d["nights"], d["rate"],
                        S, F, b, R, fb, tp_kind, tp_val, tm, trm, E2.MK / 100, out)
        return out[:k]

    def stats(self, tr, cost=0.5):
        g = (tr[:, 4] - tr[:, 3]) * tr[:, 2]
        net = g - cost - tr[:, 5]
        yr = self.d["year"][tr[:, 1].astype(int)]
        IS = yr < 2023
        eq = np.cumsum(net); dd = float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max())
        return dict(trades=len(tr), net=float(net.sum()), IS=float(net[IS].sum()), OOS=float(net[~IS].sum()),
                    win=float((net > 0).mean() * 100), tp_hits=int((tr[:, 6] == 2).sum()), maxDD=dd,
                    by_year={int(y): float(net[yr == y].sum()) for y in sorted(set(yr))}, net_arr=net, month=tr[:, 7].astype(int), IS_mask=IS)
