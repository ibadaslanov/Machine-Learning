"""06:00 breakout strategy with a stop-and-reverse order.

Rules (times are in the strategy timezone, candles are 5 minutes):

1. Reference candle: the 05:55 candle. Its high is H and its low is L.
2. At 06:00 a buy stop goes at H and a sell stop at L. The first one to fill
   cancels the other. If neither fills during the 06:00 candle there is no
   trade that day.
3. Reverse order: once long, a sell stop sits at the low of the 06:00 candle
   (a buy stop at its high once short). That low is only known when the
   candle closes at 06:05, so until then the order sits at L (H for a short).
   What happens when it fills depends on the mode (see MODES).
4. No take profit. Whatever is open at the next session's 06:00 is closed at
   that bar's open and the new setup starts. Days without data (weekends,
   holidays) are skipped, so a trade is held until the next 06:00 that exists.

Bars can be any resolution that divides the 5-minute candle (1-minute is
best). Inside a bar the price path is assumed to be open -> low -> high ->
close for a green bar and open -> high -> low -> close for a red bar. A stop
fills when price trades through its level, not when it only touches it; a bar
that opens beyond the level fills at the open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# mode -> (max reversals per day (None = unlimited), keep a plain stop once
# no reversals are left)
MODES = {
    "hold": (0, False),  # no reverse order, hold to the next 06:00
    "stop": (0, True),  # the reverse order only closes the trade
    "flip": (1, False),  # reverse once, the new trade holds to the next 06:00
    "flip_stop": (1, True),  # reverse once, the new trade is stopped at the other 06:00 extreme
    "flip_always": (None, False),  # keep reversing between the 06:00 high and low
}

NAN = float("nan")


@dataclass(frozen=True)
class Params:
    session_time: str = "06:00"
    tz: str = "UTC"
    candle_minutes: int = 5
    mode: str = "flip"
    fee: float = 0.0  # per side, as a fraction of notional
    slippage: float = 0.0  # per fill, as a fraction of price, always adverse

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, got {self.mode!r}")


@dataclass
class BacktestResult:
    params: Params
    trades: pd.DataFrame  # one row per position (a reversal makes two)
    sessions: pd.DataFrame  # one row per 06:00 session
    stats: dict = field(default_factory=dict)


def find_sessions(bars: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Locate every 06:00 session that has both a 05:55 and a 06:00 candle."""
    idx = bars.index
    if idx.tz is None:
        raise ValueError("bars must have a timezone-aware DatetimeIndex")
    hour, minute = (int(x) for x in p.session_time.split(":"))
    days = idx.tz_convert(p.tz).tz_localize(None).normalize().unique()
    t0 = (days + pd.Timedelta(hours=hour, minutes=minute)).tz_localize(
        p.tz, ambiguous="NaT", nonexistent="NaT"
    )
    t0 = t0[~t0.isna()]
    candle = pd.Timedelta(minutes=p.candle_minutes)
    ref_start = idx.searchsorted(t0 - candle)
    start = idx.searchsorted(t0)
    sig_end = idx.searchsorted(t0 + candle)
    ok = (ref_start < start) & (start < sig_end)

    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    rows = []
    for t, a, b, c in zip(t0[ok], ref_start[ok], start[ok], sig_end[ok]):
        rows.append(
            {
                "t0": t,
                "start": b,
                "sig_end": c,
                "ref_high": high[a:b].max(),
                "ref_low": low[a:b].min(),
                "sig_high": high[b:c].max(),
                "sig_low": low[b:c].min(),
            }
        )
    cols = ["t0", "start", "sig_end", "ref_high", "ref_low", "sig_high", "sig_low"]
    return pd.DataFrame(rows, columns=cols)


class _Simulator:
    def __init__(self, bars: pd.DataFrame, sessions: pd.DataFrame, p: Params):
        self.p = p
        self.max_flips, self.final_stop = MODES[p.mode]
        self.times = bars.index
        self.o = bars["open"].to_numpy(float).tolist()
        self.h = bars["high"].to_numpy(float).tolist()
        self.l = bars["low"].to_numpy(float).tolist()
        self.c = bars["close"].to_numpy(float).tolist()
        self.s = sessions

        self.cur = None  # index of the current session
        self.in_window = False  # inside the 06:00 candle
        self.pos = 0
        self.entry_px = NAN
        self.entry_i = None
        self.entry_reason = None
        self.flips = 0
        self.buy = NAN  # pending buy stop level
        self.sell = NAN  # pending sell stop level
        self.legs = []
        self.multi_fill_bars = 0

    def run(self):
        starts = self.s["start"].tolist()
        sig_end = self.s["sig_end"].tolist()
        n_sessions = len(starts)
        k_next = 0
        for i in range(len(self.o)):
            if self.in_window and i >= sig_end[self.cur]:
                self._close_window()
            if k_next < n_sessions and i == starts[k_next]:
                self._start_session(k_next, i)
                k_next += 1
            if self.buy == self.buy or self.sell == self.sell:
                self._simulate_bar(i)
        if self.pos:
            last = len(self.c) - 1
            self._close(last, self._px(self.c[last], -self.pos), "end_of_data")
        return self.legs

    # -- session clock -----------------------------------------------------

    def _start_session(self, k, i):
        if self.pos:
            self._close(i, self._px(self.o[i], -self.pos), "session_close")
        self.cur = k
        self.in_window = True
        self.flips = 0
        self.buy = self.s.at[k, "ref_high"]
        self.sell = self.s.at[k, "ref_low"]

    def _close_window(self):
        """The 06:00 candle has closed: drop unfilled entries, move the reverse order."""
        self.in_window = False
        if self.pos == 0:
            self.buy = self.sell = NAN
        elif self.pos == 1 and self.sell == self.sell:
            self.sell = self.s.at[self.cur, "sig_low"]
        elif self.pos == -1 and self.buy == self.buy:
            self.buy = self.s.at[self.cur, "sig_high"]

    # -- orders and fills --------------------------------------------------

    def _px(self, price, side):
        return price * (1 + side * self.p.slippage)

    def _can_flip(self):
        return self.max_flips is None or self.flips < self.max_flips

    def _place_exit_order(self):
        self.buy = self.sell = NAN
        if not (self._can_flip() or self.final_stop):
            return
        k = self.cur
        if self.pos == 1:
            self.sell = self.s.at[k, "ref_low" if self.in_window else "sig_low"]
        else:
            self.buy = self.s.at[k, "ref_high" if self.in_window else "sig_high"]

    def _fill(self, side, level, i):
        px = self._px(level, side)
        if self.pos == 0:
            self._open(side, px, i, "breakout")
        elif self._can_flip():
            self._close(i, px, "reverse")
            self.flips += 1
            self._open(side, px, i, "reverse")
        else:
            self._close(i, px, "stop")
            self.buy = self.sell = NAN

    def _open(self, side, px, i, reason):
        self.pos = side
        self.entry_px = px
        self.entry_i = i
        self.entry_reason = reason
        self._place_exit_order()

    def _close(self, i, px, reason):
        s = self.s.loc[self.cur]
        side = self.pos
        gross = side * (px / self.entry_px - 1)
        risk = s["ref_high"] - s["ref_low"]
        self.legs.append(
            {
                "session": s["t0"],
                "side": "long" if side == 1 else "short",
                "entry_time": self.times[self.entry_i],
                "entry_price": self.entry_px,
                "entry_reason": self.entry_reason,
                "exit_time": self.times[i],
                "exit_price": px,
                "exit_reason": reason,
                "gross_ret": gross,
                "net_ret": gross - 2 * self.p.fee,
                "r": side * (px - self.entry_px) / risk if risk > 0 else NAN,
            }
        )
        self.pos = 0

    def _simulate_bar(self, i):
        o, h, l, c = self.o[i], self.h[i], self.l[i], self.c[i]
        if c > o:
            path = (l, h, c)
        elif c < o:
            path = (h, l, c)
        elif h - o <= o - l:
            path = (h, l, c)
        else:
            path = (l, h, c)

        fills = 0
        # An order the bar opens beyond fills at the open.
        for _ in range(4):
            if self.buy == self.buy and o > self.buy:
                self._fill(1, o, i)
            elif self.sell == self.sell and o < self.sell:
                self._fill(-1, o, i)
            else:
                break
            fills += 1

        px = o
        for target in path:
            for _ in range(4):
                if target > px and self.buy == self.buy and px <= self.buy < target:
                    px = self.buy
                    self._fill(1, px, i)
                elif target < px and self.sell == self.sell and target < self.sell <= px:
                    px = self.sell
                    self._fill(-1, px, i)
                else:
                    break
                fills += 1
            px = target
        if fills > 1:
            self.multi_fill_bars += 1


def run_backtest(bars: pd.DataFrame, p: Params = Params()) -> BacktestResult:
    """Run the strategy on OHLC bars indexed by bar open time (timezone-aware)."""
    bars = bars.sort_index()
    sessions = find_sessions(bars, p)
    sim = _Simulator(bars, sessions, p)
    trades = pd.DataFrame(
        sim.run(),
        columns=[
            "session", "side", "entry_time", "entry_price", "entry_reason",
            "exit_time", "exit_price", "exit_reason", "gross_ret", "net_ret", "r",
        ],
    )

    per_session = trades.groupby("session").agg(
        direction=("side", "first"),
        legs=("side", "size"),
        reversals=("entry_reason", lambda s: int((s == "reverse").sum())),
        gross_ret=("gross_ret", "sum"),
        net_ret=("net_ret", "sum"),
        r=("r", "sum"),
    )
    out = sessions.set_index("t0")[["ref_high", "ref_low", "sig_high", "sig_low"]]
    out = out.join(per_session)
    out["direction"] = out["direction"].fillna("none")
    out[["legs", "reversals"]] = out[["legs", "reversals"]].fillna(0).astype(int)
    out[["gross_ret", "net_ret"]] = out[["gross_ret", "net_ret"]].fillna(0.0)
    out.index.name = "session"

    result = BacktestResult(p, trades, out)
    result.stats = summarize(result, bars, sim.multi_fill_bars)
    return result


def summarize(result: BacktestResult, bars: pd.DataFrame, multi_fill_bars: int = 0) -> dict:
    s = result.sessions
    traded = s[s["legs"] > 0]
    rets = traded["net_ret"]
    wins, losses = rets[rets > 0], rets[rets < 0]
    equity = s["net_ret"].cumsum()
    years = (bars.index[-1] - bars.index[0]) / pd.Timedelta(days=365.25)
    per_year = len(s) / years if years > 0 else NAN
    std = s["net_ret"].std()

    def pct(x):
        return 100 * x if x == x else NAN

    return {
        "period": f"{bars.index[0]:%Y-%m-%d} to {bars.index[-1]:%Y-%m-%d}",
        "sessions": len(s),
        "days traded": len(traded),
        "reversals": int(s["reversals"].sum()),
        "win rate %": pct(len(wins) / len(traded)) if len(traded) else NAN,
        "avg win %": pct(wins.mean()),
        "avg loss %": pct(losses.mean()),
        "profit factor": wins.sum() / -losses.sum() if len(losses) else NAN,
        "avg day %": pct(rets.mean()),
        "total % (1x, no compounding)": pct(rets.sum()),
        "total % (compounded)": pct(np.prod(1 + rets.to_numpy()) - 1),
        "max drawdown %": pct((equity - equity.cummax()).min()) if len(s) else NAN,
        "sharpe (annualized)": s["net_ret"].mean() / std * math.sqrt(per_year) if std > 0 else NAN,
        "total R": traded["r"].sum(),
        "avg R per day": traded["r"].mean(),
        "buy & hold %": pct(bars["close"].iloc[-1] / bars["open"].iloc[0] - 1),
        "bars with 2+ fills": multi_fill_bars,
    }
