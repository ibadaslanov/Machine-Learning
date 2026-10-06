import io
import zipfile

import pandas as pd
import pytest

from six_am_breakout.data import load_ohlc
from six_am_breakout.fetch_binance import read_kline_zip
from six_am_breakout.strategy import MODES, Params, run_backtest


def bars_from(rows, tz="UTC"):
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows]).tz_localize(tz)
    return pd.DataFrame([r[1:] for r in rows], columns=["open", "high", "low", "close"], index=idx)


def legs(result):
    return [
        (t.side, round(t.entry_price, 6), round(t.exit_price, 6), t.exit_reason)
        for t in result.trades.itertuples()
    ]


# The screenshot, 5-minute candles, price = 1000 - pixel row.
CHART = [
    ("2026-10-05 05:35", 573, 593, 563, 573),
    ("2026-10-05 05:40", 574, 578, 514, 516),
    ("2026-10-05 05:45", 516, 517, 480, 486),
    ("2026-10-05 05:50", 486, 518, 470, 508),  # sweep of the low
    ("2026-10-05 05:55", 509, 510, 501, 502),  # reference candle
    ("2026-10-05 06:00", 502, 526, 501, 524),  # breaks the 05:55 high, low only touches 501
    ("2026-10-05 06:05", 524, 535, 522, 532),
    ("2026-10-05 06:10", 532, 552, 530, 548),
    ("2026-10-05 06:15", 548, 578, 546, 574),
    ("2026-10-05 06:20", 574, 585, 560, 563),
    ("2026-10-05 06:25", 563, 622, 559, 614),
    ("2026-10-05 06:30", 614, 750, 612, 729),
    ("2026-10-05 06:35", 729, 750, 690, 707),
    ("2026-10-05 06:40", 707, 749, 688, 721),
    ("2026-10-05 06:45", 721, 757, 684, 721),
    ("2026-10-05 06:50", 721, 748, 690, 694),
    ("2026-10-06 05:55", 700, 702, 698, 700),
    ("2026-10-06 06:00", 700, 701, 699, 700),
]


def test_chart_long_held_to_next_0600():
    r = run_backtest(bars_from(CHART))
    assert legs(r) == [("long", 510, 700, "session_close")]
    t = r.trades.iloc[0]
    assert t.entry_time == pd.Timestamp("2026-10-05 06:00", tz="UTC")
    assert t.exit_time == pd.Timestamp("2026-10-06 06:00", tz="UTC")
    assert t.r == pytest.approx(190 / 9)
    assert t.gross_ret == pytest.approx(700 / 510 - 1)
    # Day 2 never breaks its 05:55 range.
    assert list(r.sessions["direction"]) == ["long", "none"]


# Long at the 05:55 high, then a drop through the 06:00 low, then a rally
# through the 06:00 high.
REVERSAL = [
    ("2026-10-05 05:55", 100, 101, 99, 100),
    ("2026-10-05 06:00", 100, 103, 99.5, 102),
    ("2026-10-05 06:05", 102, 102, 97, 98),
    ("2026-10-05 06:10", 98, 104, 97, 104),
    ("2026-10-06 05:55", 105, 106, 104, 105),
    ("2026-10-06 06:00", 105, 105.5, 104.5, 105),
]


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("hold", [("long", 101, 105, "session_close")]),
        ("stop", [("long", 101, 99.5, "stop")]),
        ("flip", [("long", 101, 99.5, "reverse"), ("short", 99.5, 105, "session_close")]),
        ("flip_stop", [("long", 101, 99.5, "reverse"), ("short", 99.5, 103, "stop")]),
        (
            "flip_always",
            [("long", 101, 99.5, "reverse"), ("short", 99.5, 103, "reverse"), ("long", 103, 105, "session_close")],
        ),
        (
            "flip2_stop",
            [("long", 101, 99.5, "reverse"), ("short", 99.5, 103, "reverse"), ("long", 103, 105, "session_close")],
        ),
    ],
)
def test_reverse_order_modes(mode, expected):
    assert legs(run_backtest(bars_from(REVERSAL), Params(mode=mode))) == expected


def test_all_modes_covered():
    assert set(MODES) == {"hold", "stop", "flip", "flip_stop", "flip_always", "flip2_stop"}


def test_no_reverse_inside_0600_candle():
    rows = [
        ("2026-10-05 05:55", 100, 101, 99, 100),
        ("2026-10-05 06:00", 100, 102, 98, 98.5),  # up through 101, then down through 99: stays long
        ("2026-10-05 06:05", 98.5, 99, 98, 98.5),  # only touches the 06:00 low (98)
        ("2026-10-05 06:10", 98.5, 98.6, 97, 97.5),  # through the 06:00 low
    ]
    r = run_backtest(bars_from(rows))
    assert legs(r) == [("long", 101, 98, "reverse"), ("short", 98, 97.5, "end_of_data")]
    assert r.stats["bars with 2+ fills"] == 0


def test_no_breakout_in_0600_candle_means_no_trade():
    rows = [
        ("2026-10-05 05:55", 100, 101, 99, 100),
        ("2026-10-05 06:00", 100, 100.5, 99.5, 100),
        ("2026-10-05 06:05", 100, 110, 100, 110),
    ]
    r = run_backtest(bars_from(rows))
    assert r.trades.empty
    assert list(r.sessions["direction"]) == ["none"]


def test_flip2_stop_closes_on_the_third_cross():
    rows = REVERSAL[:4] + [
        ("2026-10-05 06:15", 104, 104, 98, 98.5),  # through the 06:00 low again: stop, no third reversal
        ("2026-10-05 06:20", 98.5, 110, 98.5, 110),
    ]
    r = run_backtest(bars_from(rows), Params(mode="flip2_stop"))
    assert legs(r) == [("long", 101, 99.5, "reverse"), ("short", 99.5, 103, "reverse"), ("long", 103, 99.5, "stop")]


def test_tradingview_fills_on_touch_and_goes_to_nearest_extreme_first():
    rows = [
        ("2026-10-05 05:55", 100, 101, 99, 100),
        ("2026-10-05 06:00", 100, 101, 99, 100.5),  # touches both levels, high and low equally far from the open
        ("2026-10-05 06:05", 100.5, 100.8, 100.2, 100.6),
    ]
    assert run_backtest(bars_from(rows)).trades.empty  # default: a touch does not fill
    r = run_backtest(bars_from(rows), Params(tradingview=True))
    assert legs(r) == [("long", 101, 100.6, "end_of_data")]  # high first, the short side is cancelled


def test_gap_beyond_level_fills_at_open():
    rows = [
        ("2026-10-05 05:55", 100, 101, 99, 100),
        ("2026-10-05 06:00", 103, 104, 102.5, 103.5),
    ]
    assert legs(run_backtest(bars_from(rows))) == [("long", 103, 103.5, "end_of_data")]


def test_one_minute_bars_build_the_5_minute_candles():
    rows = [(f"2026-10-05 05:5{m}", 100, 100.5, 99.5, 100) for m in range(5, 10)]
    rows[2] = ("2026-10-05 05:57", 100, 101, 100, 100)  # 05:55 candle high = 101
    rows[3] = ("2026-10-05 05:58", 100, 100, 99, 100)  # 05:55 candle low = 99
    rows += [
        ("2026-10-05 06:00", 100, 100.8, 99.8, 100.5),
        ("2026-10-05 06:01", 100.5, 101.5, 100.4, 101.2),  # long at 101
        ("2026-10-05 06:02", 101.2, 101.3, 99.6, 99.7),  # no reverse order until the 06:00 candle closes
        ("2026-10-05 06:03", 99.7, 100, 99.7, 100),
        ("2026-10-05 06:04", 100, 100.2, 100, 100.1),
        ("2026-10-05 06:05", 100.1, 100.1, 99.5, 99.5),  # through the 06:00 low (99.6)
    ]
    r = run_backtest(bars_from(rows))
    assert legs(r) == [("long", 101, 99.6, "reverse"), ("short", 99.6, 99.5, "end_of_data")]
    assert r.trades.entry_time.iloc[0] == pd.Timestamp("2026-10-05 06:01", tz="UTC")


def test_session_time_is_in_strategy_timezone():
    rows = [
        ("2026-10-05 01:55", 100, 101, 99, 100),  # 05:55 in Baku
        ("2026-10-05 02:00", 100, 102, 100, 102),
        ("2026-10-05 05:55", 200, 201, 199, 200),  # 05:55 in UTC
        ("2026-10-05 06:00", 200, 210, 190, 205),
    ]
    r = run_backtest(bars_from(rows), Params(tz="Asia/Baku"))
    assert legs(r) == [("long", 101, 205, "end_of_data")]
    assert r.trades.entry_time.iloc[0] == pd.Timestamp("2026-10-05 02:00", tz="UTC")


def test_trade_held_over_days_without_data():
    rows = [
        ("2026-10-02 05:55", 100, 101, 99, 100),  # Friday
        ("2026-10-02 06:00", 100, 102, 100, 102),
        ("2026-10-02 12:00", 102, 103, 101, 102),
        ("2026-10-05 05:55", 110, 111, 109, 110),  # Monday
        ("2026-10-05 06:00", 110, 110.5, 109.5, 110),
    ]
    r = run_backtest(bars_from(rows))
    assert legs(r) == [("long", 101, 110, "session_close")]
    assert r.trades.exit_time.iloc[0] == pd.Timestamp("2026-10-05 06:00", tz="UTC")


def test_fees_and_slippage():
    r = run_backtest(bars_from(CHART), Params(fee=0.001, slippage=0.0005))
    t = r.trades.iloc[0]
    assert t.entry_price == pytest.approx(510 * 1.0005)
    assert t.exit_price == pytest.approx(700 * 0.9995)
    assert t.net_ret == pytest.approx(t.gross_ret - 0.002)


def test_no_edge_on_a_random_walk():
    """No look-ahead: on a driftless random walk the strategy should make ~0."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 1440 * 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    open_ = np.concatenate([[100], close[:-1]])
    wiggle = np.abs(rng.normal(0, 0.0004, (2, n)))
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1 + wiggle[0]),
            "low": np.minimum(open_, close) * (1 - wiggle[1]),
            "close": close,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC"),
    )
    for mode in MODES:
        s = run_backtest(bars, Params(mode=mode)).sessions["net_ret"]
        t_stat = s.mean() / (s.std() / len(s) ** 0.5)
        assert abs(t_stat) < 3, mode


# -- data loading ------------------------------------------------------------


def test_load_tradingview_unix_seconds(tmp_path):
    f = tmp_path / "tv.csv"
    f.write_text("time,open,high,low,close,Volume\n1759644000,1,2,0.5,1.5,10\n")
    bars = load_ohlc(f)
    assert bars.index[0] == pd.Timestamp(1759644000, unit="s", tz="UTC")
    assert bars.iloc[0].tolist() == [1, 2, 0.5, 1.5]


def test_load_iso_with_offset(tmp_path):
    f = tmp_path / "iso.csv"
    f.write_text("time,open,high,low,close\n2026-10-05T06:00:00+04:00,1,2,0.5,1.5\n")
    assert load_ohlc(f).index[0] == pd.Timestamp("2026-10-05 02:00", tz="UTC")


def test_load_mt5_export_in_server_time(tmp_path):
    f = tmp_path / "mt5.csv"
    f.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2026.10.05\t06:00:00\t1\t2\t0.5\t1.5\t10\t0\t5\n"
    )
    bars = load_ohlc(f, data_tz="Europe/Athens")  # UTC+3 in October
    assert bars.index[0] == pd.Timestamp("2026-10-05 03:00", tz="UTC")


def test_load_headerless_binance_microseconds(tmp_path):
    f = tmp_path / "binance.csv"
    f.write_text("1759644000000000,1,2,0.5,1.5,10,1759644059999999,15,3,5,7,0\n")
    assert load_ohlc(f).index[0] == pd.Timestamp(1759644000, unit="s", tz="UTC")


def test_read_futures_kline_zip_with_header():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "BTCUSDT-1m-2026-09.csv",
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
            "taker_buy_volume,taker_buy_quote_volume,ignore\n"
            "1759644000000,1,2,0.5,1.5,10,1759644059999,15,3,5,7,0\n",
        )
    bars = read_kline_zip(buf.getvalue())
    assert bars.index[0] == pd.Timestamp(1759644000, unit="s", tz="UTC")
    assert bars.iloc[0].tolist() == [1, 2, 0.5, 1.5]
