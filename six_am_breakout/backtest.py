"""Backtest the 06:00 breakout strategy on a CSV of OHLC bars.

    python -m six_am_breakout.backtest data/BTCUSDT-1m.csv --compare
    python -m six_am_breakout.backtest xauusd.csv --data-tz Europe/Athens --tz Asia/Baku

Prints the stats, then writes trades_<mode>.csv, sessions_<mode>.csv and
equity.png to --out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import bar_minutes, load_ohlc
from .strategy import MODES, Params, run_backtest

# Categorical slots 1-5 of the reference palette, in fixed order per mode.
MODE_COLORS = dict(zip(MODES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))


def format_stats(results: dict) -> str:
    table = pd.DataFrame({mode: r.stats for mode, r in results.items()})

    def fmt(v):
        if isinstance(v, str):
            return v
        if v != v:
            return "-"
        if float(v).is_integer():
            return f"{int(v):,}"
        return f"{v:,.2f}"

    return table.map(fmt).to_string()


def plot_equity(results: dict, path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for mode, r in results.items():
        equity = r.sessions["net_ret"].cumsum() * 100
        ax.plot(equity.index, equity.to_numpy(), color=MODE_COLORS[mode], linewidth=2, label=mode)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#a3a29d")
    ax.tick_params(colors="#52514e", labelsize=9, length=0)
    ax.set_ylabel("cumulative return, % (1x, not compounded)", color="#52514e", fontsize=9)
    ax.set_title(title, color="#0b0b0b", fontsize=12, loc="left", pad=30 if len(results) > 1 else 12)
    if len(results) > 1:
        ax.legend(
            frameon=False, fontsize=9, labelcolor="#0b0b0b", ncol=len(results),
            loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.5,
        )
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="OHLC bars, 1-minute or 5-minute, timestamps = bar open time")
    ap.add_argument("--tz", default="UTC", help="timezone the 06:00 is in (default UTC)")
    ap.add_argument("--data-tz", default="UTC", help="timezone of timestamps that carry no offset (default UTC)")
    ap.add_argument("--session", default="06:00", help="session time (default 06:00)")
    ap.add_argument("--mode", choices=MODES, default="flip", help="what the reverse order does (default flip)")
    ap.add_argument("--compare", action="store_true", help="run every mode side by side")
    ap.add_argument("--fee", type=float, default=0.0005, help="fee per side as a fraction (default 0.0005 = 0.05%%)")
    ap.add_argument("--slippage", type=float, default=0.0, help="slippage per fill as a fraction (default 0)")
    ap.add_argument("--start", help="first date to test, YYYY-MM-DD")
    ap.add_argument("--end", help="last date to test, YYYY-MM-DD")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"), help="output folder")
    args = ap.parse_args()

    bars = load_ohlc(args.csv, args.data_tz)
    local = bars.index.tz_convert(args.tz)
    keep = pd.Series(True, index=bars.index)
    if args.start:
        keep &= local >= pd.Timestamp(args.start, tz=args.tz)
    if args.end:
        keep &= local < pd.Timestamp(args.end, tz=args.tz) + pd.Timedelta(days=1)
    bars = bars[keep.to_numpy()]
    if bars.empty:
        raise SystemExit("no bars in the selected period")

    step = bar_minutes(bars)
    if step > 5 or 5 % step:
        raise SystemExit(f"bars are {step:g} minutes apart; the strategy needs 1-minute or 5-minute bars")
    if step == 5:
        print("note: 5-minute bars hide the order of moves inside a candle; 1-minute data is more accurate\n")

    modes = list(MODES) if args.compare else [args.mode]
    results = {
        m: run_backtest(bars, Params(args.session, args.tz, mode=m, fee=args.fee, slippage=args.slippage))
        for m in modes
    }

    print(f"{Path(args.csv).name}  |  session {args.session} {args.tz}  |  fee {args.fee:.4%}/side  |  slippage {args.slippage:.4%}")
    print(format_stats(results))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for m, r in results.items():
        r.trades.to_csv(out / f"trades_{m}.csv", index=False)
        r.sessions.to_csv(out / f"sessions_{m}.csv")
    try:
        title = f"{Path(args.csv).stem}: {args.session} breakout, equity by reverse-order mode"
        if not args.compare:
            title = f"{Path(args.csv).stem}: {args.session} breakout ({args.mode}), equity"
        plot_equity(results, out / "equity.png", title)
    except ImportError:
        print("\n(install matplotlib for the equity chart)")
    print(f"\nwrote trades, sessions and equity chart to {out}")


if __name__ == "__main__":
    main()
