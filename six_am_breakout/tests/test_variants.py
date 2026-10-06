import pandas as pd
import pytest

from six_am_breakout import variants as vs
from six_am_breakout.strategy import Params, run_backtest

from .test_strategy import CHART, REVERSAL, bars_from


@pytest.mark.parametrize("rows", [CHART, REVERSAL])
def test_base_variant_matches_the_backtester_in_tradingview_mode(rows):
    bars = bars_from(rows)
    engine = run_backtest(bars, Params("06:00", mode="flip", tradingview=True, max_reversals=4))
    expected = engine.trades.loc[engine.trades.exit_reason != "end_of_data", "points"].round(6).tolist()
    got = vs.run(bars, ["06:00"], vs.BASE, 0.0)["net"].round(6).tolist()
    assert got == expected


ADDS = [
    ("2026-10-05 05:55", 100, 101, 99, 100),  # R = 2
    ("2026-10-05 06:00", 100, 103, 100, 103),  # long at 101
    ("2026-10-05 06:05", 103, 110, 103, 110),  # adds at 105 and 109 (every 2R = 4 points)
    ("2026-10-06 05:55", 112, 113, 111, 112),
    ("2026-10-06 06:00", 112, 112.5, 111.5, 112),  # closed at the open
]


def test_adding_units():
    tr = vs.run(bars_from(ADDS), ["06:00"], dict(vs.BASE, adds=2, add_step=2), 0.5)
    assert tr["units"].tolist() == [3]
    assert tr["net"].iloc[0] == pytest.approx((112 - 101) + (112 - 105) + (112 - 109) - 3 * 0.5)


def test_take_profit():
    tr = vs.run(bars_from(ADDS), ["06:00"], dict(vs.BASE, tp=3), 0.0)
    assert tr["net"].tolist() == [pytest.approx(6.0)]  # 101 + 3R
