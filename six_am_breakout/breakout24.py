"""Python port of the two '24h Breakout - Long-only, 1 trade/day' Pine indicators.

No 1-minute data here, so double-break candles use the scripts' own fallback
(gap first, else 'against position first' when long, else previous candle colour) -
the same thing TradingView does for bars older than its 1-minute history.
"""
import math
import numpy as np
import pandas as pd


def run(bars, start_hr=10, sizing="v1", risk_pct=0.07, start_bal=1000.0, margin=500.0, usd_pt=1.0,
        lot_step=0.01, whole_qty=True, cost=0.0, fin_rate=0.0):
    t = bars.index
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    hours = (t.as_unit("s").asi8 / 3600.0)
    blk = np.floor((hours - start_hr) / 24.0).astype(np.int64)

    cur_blk = None; run_h = run_l = lvl_h = lvl_l = np.nan
    pos = 0; entry = np.nan; qty = 0.0; entries_today = 0; done_today = False; entry_t = None
    balance = peak = start_bal; max_dd = 0.0
    trades = []
    dbl = 0
    for i in range(len(o)):
        new_blk = cur_blk is None or blk[i] != cur_blk
        if new_blk:
            lvl_h, lvl_l = run_h, run_l
            run_h, run_l = h[i], l[i]
            cur_blk = blk[i]
            entries_today, done_today = 0, False
        else:
            run_h, run_l = max(run_h, h[i]), min(run_l, l[i])
        prev_green = i > 0 and c[i - 1] >= o[i - 1]
        up = not np.isnan(lvl_h) and h[i] > lvl_h
        dn = not np.isnan(lvl_l) and l[i] < lvl_l
        px_up, px_dn = max(lvl_h, o[i]) if up else np.nan, min(lvl_l, o[i]) if dn else np.nan
        steps = []
        if up and dn:
            dbl += 1
            if o[i] > lvl_h: first = 1
            elif o[i] < lvl_l: first = -1
            elif pos == 1: first = -1                     # fallback: against position first
            else: first = 1 if prev_green else -1
            steps = [(1, px_up), (-1, lvl_l)] if first == 1 else [(-1, px_dn), (1, lvl_h)]
        elif up:
            steps = [(1, px_up)]
        elif dn:
            steps = [(-1, px_dn)]
        for d, px in steps:
            if d == 1:
                if pos == 0 and not done_today:
                    max_qty = balance / margin
                    if sizing == "risk":
                        dist = max(px - lvl_l, 0.01)
                        raw = min(balance * risk_pct / (dist * usd_pt), max_qty)
                        qty = max(math.floor(raw / lot_step) * lot_step, 0.0)
                    else:
                        qty = math.floor(max_qty) if whole_qty else max_qty
                    pos, entry, entry_t = 1, px, t[i]
                    entries_today += 1
            else:
                if pos == 1:
                    pn = px - entry - cost
                    days = (t[i] - entry_t).total_seconds() / 86400
                    fin = qty * entry * usd_pt * fin_rate * days / 365
                    pnl = qty * pn * usd_pt - fin
                    balance += pnl
                    trades.append(dict(entry_time=entry_t, exit_time=t[i], entry=entry, exit=px, qty=qty,
                                       points=pn, dollars=pnl, days=days))
                    pos, qty = 0, 0.0
                    if entries_today >= 1:
                        done_today = True
        equity = balance + (qty * (c[i] - entry) * usd_pt if pos == 1 else 0.0)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
    tr = pd.DataFrame(trades)
    open_eq = balance + (qty * (c[-1] - entry) * usd_pt if pos == 1 else 0.0)
    return tr, dict(balance=balance, equity=open_eq, max_dd=max_dd, double_breaks=dbl, open=pos == 1)


def summary(tr, info=None):
    if tr.empty:
        return {}
    w, ls = tr.points[tr.points > 0].sum(), -tr.points[tr.points <= 0].sum()
    out = dict(trades=len(tr), win=100 * (tr.points > 0).mean(), pf=w / ls if ls else np.nan,
               net_pts=tr.points.sum(), avg_days=tr.days.mean())
    if info:
        out.update(end_equity=info["equity"], max_dd=info["max_dd"])
    return out


def main():
    import argparse

    from .data import load_ohlc

    ap = argparse.ArgumentParser(description="Backtest the 24h breakout (long only, 1 trade/day) on hourly bars.")
    ap.add_argument("csv", help="hourly OHLC bars (UTC unless --data-tz)")
    ap.add_argument("--data-tz", default="UTC")
    ap.add_argument("--start-hr", type=int, default=10, help="UTC hour the 24h day starts (default 10)")
    ap.add_argument("--risk", type=float, help="risk this fraction of the balance to the exit level (7_str); "
                                               "default: full size balance/margin (v1)")
    ap.add_argument("--balance", type=float, default=1000.0)
    ap.add_argument("--margin", type=float, default=500.0)
    ap.add_argument("--cost", type=float, default=0.5, help="cost per trade in points (default 0.5)")
    ap.add_argument("--financing", type=float, default=0.05, help="overnight financing per year (default 0.05)")
    args = ap.parse_args()

    bars = load_ohlc(args.csv, args.data_tz)
    kw = dict(sizing="risk", risk_pct=args.risk) if args.risk else dict(sizing="v1")
    tr, info = run(bars, start_hr=args.start_hr, start_bal=args.balance, margin=args.margin,
                   cost=args.cost, fin_rate=args.financing, **kw)
    s = summary(tr, info)
    print(f"trades {s['trades']} | win {s['win']:.1f}% | PF {s['pf']:.2f} | net {s['net_pts']:+,.0f} points | "
          f"avg hold {s['avg_days']:.1f} days | ${args.balance:,.0f} -> ${info['equity']:,.0f} | "
          f"max drop {info['max_dd']:.0%}")


if __name__ == "__main__":
    main()
