"""Download Binance klines from data.binance.vision into one CSV.

    python -m six_am_breakout.fetch_binance BTCUSDT --start 2024-01 --end 2026-09

Monthly archives are used where they exist; for the current month (not
archived yet) it falls back to the daily files. Times are written as UTC
milliseconds of the bar's open.
"""

from __future__ import annotations

import argparse
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .data import BINANCE_COLUMNS, parse_times

URL = "https://data.binance.vision/data/{market}/{period}/klines/{symbol}/{interval}/{symbol}-{interval}-{date}.zip"
MARKETS = {"spot": "spot", "futures": "futures/um"}


def read_kline_zip(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, usecols=range(5), names=BINANCE_COLUMNS, dtype=str)
    if not df.iloc[0, 0].isdigit():  # futures archives carry a header row
        df = df.iloc[1:]
    bars = df[["open", "high", "low", "close"]].astype(float)
    bars.index = parse_times(df["time"].astype("int64"))
    return bars


def _get(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch(symbol: str, start: str, end: str, interval: str = "1m", market: str = "spot") -> pd.DataFrame:
    parts = []
    for month in pd.period_range(start, end, freq="M"):
        kw = dict(market=MARKETS[market], symbol=symbol, interval=interval)
        raw = _get(URL.format(period="monthly", date=f"{month}", **kw))
        if raw is not None:
            parts.append(read_kline_zip(raw))
            print(f"{month}: monthly archive")
            continue
        days = 0
        for day in pd.date_range(month.start_time, month.end_time, freq="D"):
            raw = _get(URL.format(period="daily", date=f"{day:%Y-%m-%d}", **kw))
            if raw is not None:
                parts.append(read_kline_zip(raw))
                days += 1
        print(f"{month}: {days} daily files")
    if not parts:
        raise SystemExit("nothing downloaded, check the symbol, market and dates")
    bars = pd.concat(parts).sort_index()
    return bars[~bars.index.duplicated(keep="last")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol", help="e.g. BTCUSDT")
    ap.add_argument("--start", required=True, help="first month, YYYY-MM")
    ap.add_argument("--end", required=True, help="last month, YYYY-MM")
    ap.add_argument("--interval", default="1m", help="kline interval, 1m or 5m (default 1m)")
    ap.add_argument("--market", choices=MARKETS, default="spot")
    ap.add_argument("--out", help="output CSV (default data/<SYMBOL>-<interval>.csv)")
    args = ap.parse_args()

    bars = fetch(args.symbol.upper(), args.start, args.end, args.interval, args.market)
    out = Path(args.out or Path(__file__).parent / "data" / f"{args.symbol.upper()}-{args.interval}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = bars.copy()
    csv.index = (csv.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    csv.index.name = "time"
    csv.to_csv(out)
    print(f"saved {len(bars):,} bars to {out}")


if __name__ == "__main__":
    main()
