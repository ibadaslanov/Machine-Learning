"""Deep-dive of one strategy: per-year, long/short, reopen handling, costs, financing,
random-direction test (is the DIRECTION choice better than coin flips over the same holding windows?),
same-window always-long benchmark, DST split, mark-to-market drawdown."""
import numpy as np, json, datetime as dt
from zoneinfo import ZoneInfo
import engine2 as E2

X = E2.Ctx(); d = X.d
NY = ZoneInfo("America/New_York")


def dst_flags(bars):
    return np.array([bool(d["ts"][int(b)].astimezone(NY).dst()) for b in bars])


def mtm_dd(tr, cost):
    """mark-to-market drawdown: walk each trade's bars."""
    eq = 0.0; peak = 0.0; dd = 0.0
    for e, x, dr, ep, xp, f in tr:
        e, x = int(e), int(x)
        for i in range(e, x):
            m = eq + (d["C"][i] - ep) * dr
            peak = max(peak, m); dd = max(dd, peak - m)
        eq += (xp - ep) * dr - cost - f
        peak = max(peak, eq); dd = max(dd, peak - eq)
    return dd


def analyse(S, F, b, al, label, fade=False, fbs=(0, 1, 2), seed=0, dirmode=0):
    out = dict(strategy=f"{'FADE ' if fade else ''}{'LONG-ONLY ' if dirmode==1 else 'SHORT-ONLY ' if dirmode==-1 else ''}S={S} F={F} b={b} {'always' if al else 'overnight'} | {label}")
    worst = None
    for fb in fbs:
        st, tr = X.run(S, F, b, al, label, fb=fb, reopen_mode=2, dirmode=dirmode)
        g = (tr[:, 4] - tr[:, 3]) * tr[:, 2]
        if fade:
            g = -g
        tot = g.sum() - 0.5 * len(g) - tr[:, 5].sum()
        if worst is None or tot < worst[0]:
            worst = (tot, fb, tr, g)
    tot, fb, tr, g = worst
    out["worst_fb"] = E2.FB_NAMES[fb]
    fin = tr[:, 5] if not fade else None
    if fade:  # recompute financing for the faded position (opposite direction)
        st, tr2 = X.run(S, F, b, al, label, fb=fb, reopen_mode=2)
        fin = np.zeros(len(tr))
        for k, (e, x, dr, ep, xp, f) in enumerate(tr):
            e, x = int(e), int(x); s = 0.0
            for i in range(e + 1, x + 1):
                if d["nights"][i] > 0:
                    base = d["C"][i - 1] * d["nights"][i] / 365
                    s += base * (d["rate"][i] + 0.025) if dr == -1 else base * (0.025 - d["rate"][i])
            fin[k] = s
        tr = tr.copy(); tr[:, 2] *= -1; tr[:, 5] = fin
    net = g - 0.5 - fin
    yr = d["year"][tr[:, 1].astype(int)]
    out["trades"] = len(tr); out["net"] = round(float(net.sum())); out["gross"] = round(float(g.sum())); out["financing"] = round(float(fin.sum()))
    out["per_trade_net"] = round(float(net.mean()), 2); out["win_rate"] = round(100 * float((net > 0).mean()), 1)
    out["avg_hold_h"] = round(float(np.mean(d["eh"][tr[:, 1].astype(int)] - d["eh"][tr[:, 0].astype(int)])), 1)
    out["by_year"] = {int(y): round(float(net[yr == y].sum())) for y in sorted(set(yr))}
    out["years_positive"] = f"{sum(v > 0 for v in out['by_year'].values())}/{len(out['by_year'])}"
    L = tr[:, 2] == 1
    out["long"] = dict(n=int(L.sum()), net=round(float(net[L].sum()))); out["short"] = dict(n=int((~L).sum()), net=round(float(net[~L].sum())))
    out["cost_sensitivity"] = {f"{c} pt": round(float((g - c - fin).sum())) for c in (0.0, 0.5, 1.0, 1.5)}
    out["without_financing"] = round(float((g - 0.5).sum()))
    # reopen handling sensitivity (same fb)
    rm = {}
    for mode in (0, 1, 2):
        st, t2 = X.run(S, F, b, al, label, fb=fb, reopen_mode=mode, dirmode=dirmode)
        gg = (t2[:, 4] - t2[:, 3]) * t2[:, 2] * (-1 if fade else 1)
        rm[["trust recorded open", "worst-extreme fills", "no orders on reopen"][mode]] = round(float((gg - 0.5).sum()))
    out["reopen_sensitivity_ex_fin"] = rm
    # direction test: keep every trade's holding window, randomise the direction
    move = (tr[:, 4] - tr[:, 3]) * tr[:, 2] * np.sign(tr[:, 2])  # = exit - entry (long move)
    move = tr[:, 4] - tr[:, 3]
    rng = np.random.default_rng(seed)
    actual = float((move * tr[:, 2]).sum())
    sims = np.array([(move * rng.choice([-1, 1], len(move))).sum() for _ in range(5000)])
    out["direction_test"] = dict(actual_gross=round(actual), random_mean=round(float(sims.mean())), random_p95=round(float(np.percentile(sims, 95))),
                                 p_value=round(float((sims >= actual).mean()), 4), always_long_same_windows=round(float(move.sum())))
    dst = dst_flags(tr[:, 0])
    out["dst_split"] = dict(summer=round(float(net[dst].sum())), winter=round(float(net[~dst].sum())), n_summer=int(dst.sum()), n_winter=int((~dst).sum()))
    out["mtm_maxDD"] = round(mtm_dd(tr, 0.5))
    eq = np.cumsum(net); out["closed_maxDD"] = round(float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()))
    # annualised Sharpe on daily P&L (points), by exit day
    days = d["eh"][tr[:, 1].astype(int)] // 24
    u, inv = np.unique(days, return_inverse=True); dp = np.bincount(inv, net)
    alld = np.arange(days.min(), days.max() + 1); full = np.zeros(len(alld)); full[u - days.min()] = dp
    tradingdays = full[np.isin(alld % 7, [4, 5, 6, 0, 1])]  # Mon..Fri (epoch day 0 = Thursday -> Mon=4)
    out["sharpe_daily_ann"] = round(float(tradingdays.mean() / tradingdays.std() * np.sqrt(252)), 2) if tradingdays.std() > 0 else None
    return out


if __name__ == "__main__":
    import sys
    cfgs = json.loads(sys.argv[1])
    print(json.dumps([analyse(*c[:5], fade=bool(c[5]) if len(c) > 5 else False, dirmode=c[6] if len(c) > 6 else 0) for c in cfgs]))
