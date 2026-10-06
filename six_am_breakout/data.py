"""Load OHLC bars from a CSV export into a UTC-indexed DataFrame.

Handles the usual layouts:
- Binance klines (with or without a header, ms or µs timestamps)
- TradingView "Export chart data" (unix seconds or ISO time with offset)
- MetaTrader 5 (tab separated, <DATE> and <TIME> columns in server time)
- anything with a time column plus open/high/low/close columns

Timestamps must be the bar's open time. Times without a UTC offset are read
in `data_tz` (for MT5 that is the broker's server timezone).
"""

from __future__ import annotations

import re

import pandas as pd

TIME_COLUMNS = ("time", "timestamp", "datetime", "date_time", "open_time", "open time", "gmt time", "local time", "date")
BINANCE_COLUMNS = ["time", "open", "high", "low", "close"]
_OFFSET = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")
_MT5_DATE = re.compile(r"^\d{4}\.\d{2}\.\d{2}")


def load_ohlc(path, data_tz: str = "UTC") -> pd.DataFrame:
    with open(path) as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ";" if first.count(";") > first.count(",") else ","

    if first.split(sep)[0].strip().replace(".", "", 1).isdigit():
        df = pd.read_csv(path, sep=sep, header=None, usecols=range(5), names=BINANCE_COLUMNS)
        times = df["time"]
    else:
        df = pd.read_csv(path, sep=sep)
        df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]
        if "date" in df.columns and "time" in df.columns:
            times = df["date"].astype(str) + " " + df["time"].astype(str)
        else:
            col = next((c for c in TIME_COLUMNS if c in df.columns), None)
            if col is None:
                raise ValueError(f"no time column found in {list(df.columns)}")
            times = df[col]

    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing} in {list(df.columns)}")

    bars = df[["open", "high", "low", "close"]].astype(float)
    bars.index = parse_times(times, data_tz)
    bars = bars[bars.index.notna()].dropna()
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars.index.name = "time"
    return bars


def parse_times(times: pd.Series, data_tz: str = "UTC") -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(times):
        biggest = times.abs().max()
        unit = "ns" if biggest > 1e17 else "us" if biggest > 1e14 else "ms" if biggest > 1e11 else "s"
        return pd.DatetimeIndex(pd.to_datetime(times, unit=unit, utc=True))

    times = times.astype(str).str.strip()
    sample = times.iloc[0]
    if _MT5_DATE.match(sample):
        times = times.str.replace(".", "-", n=2, regex=False)
    if _OFFSET.search(sample):
        return pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    naive = pd.DatetimeIndex(pd.to_datetime(times))
    return naive.tz_localize(data_tz, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")


def bar_minutes(bars: pd.DataFrame) -> float:
    """Most common spacing between bars, in minutes."""
    return bars.index.to_series().diff().mode().iloc[0] / pd.Timedelta(minutes=1)
