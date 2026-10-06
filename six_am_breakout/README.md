# 06:00 breakout with a reverse order

A daily breakout strategy built around one candle: the 05:55 candle on the
5-minute chart.

## Rules

1. **Reference candle:** mark the high (H) and low (L) of the 05:55 candle.
2. **Entry:** at 06:00 place a buy stop at H and a sell stop at L. The first
   one to fill cancels the other. If price crosses neither during the 06:00
   candle (06:00 to 06:05), there is no trade that day.
3. **Reverse order:** once long, a sell stop sits at the **low of the 06:00
   candle**. Once short, a buy stop sits at its high. That low is only known
   when the candle closes at 06:05, so until then the order sits at the 05:55
   low (the 05:55 high for a short). When it fills, it closes the trade and
   opens the opposite one.
4. **Exit:** no take profit. Whatever is open at the next day's 06:00 is closed
   and the new setup starts. Weekends and holidays without data are skipped, so
   a Friday trade runs to Monday's 06:00.

Example from the chart this was designed on: the 05:55 candle is small and red.
The 06:00 candle breaks its high, so the strategy goes long. The reverse short
at the 06:00 low (about the 05:55 low) is never reached, and price runs up to
06:30. This exact case is a test: `test_chart_long_held_to_next_0600`.

## Reverse-order modes

What happens after the reverse order fills is still open, so the backtest
runs it several ways (`--mode`, or `--compare` for all at once):

| mode | reverse order | after reversing |
|------|---------------|-----------------|
| `flip` (default) | closes the trade and opens the opposite one | the new trade holds to the next 06:00, no stop |
| `flip_stop` | closes and reverses | the new trade is stopped at the other 06:00 extreme |
| `flip_always` | closes and reverses | keeps reversing between the 06:00 high and low |
| `stop` | only closes the trade (plain stop loss) | flat until the next 06:00 |
| `hold` | none | the first trade holds to the next 06:00 |

## TradingView

Paste `six_am_breakout.pine` into the Pine Editor, add it to a **5-minute**
chart, and set the timezone input to your chart's timezone. The Pine version
runs the same setup at every time in its session list (default: every hour
00:00 to 13:00, plus 05:30 and 06:30). Each session's reference candle is the
5-minute candle before it, and each trade runs until the next session in the
list. It reverses between the session candle's high and low up to 2 times; the
next cross after that closes the trade. All of these are inputs.
A table in the top-right corner shows trades, win %, net profit and profit
factor for each session time (closed trades, grouped by the session that
opened them), plus a total row.

## Usage (Python backtest)

```bash
pip install -r six_am_breakout/requirements.txt

# 1. Get 1-minute data, for example from Binance (any symbol, spot or futures)
python -m six_am_breakout.fetch_binance BTCUSDT --start 2024-01 --end 2026-09

# 2. Backtest every mode side by side
python -m six_am_breakout.backtest six_am_breakout/data/BTCUSDT-1m.csv --compare
```

Any CSV with a time column and open/high/low/close columns works. That covers
TradingView "Export chart data", MetaTrader 5 exports and Binance files.
Timestamps must be each bar's **open** time. 1-minute bars are best. 5-minute
bars work, but they hide the order of moves inside a candle.

Options you will probably need:

- `--tz Asia/Baku`: the timezone the 06:00 is in. Use your chart's timezone
  (TradingView shows it in the bottom-right corner). The default is UTC.
- `--data-tz Europe/Athens`: the timezone of timestamps that have no UTC
  offset. MT5 exports use the broker's server time.
- `--fee 0.0005`: cost per side as a fraction of the position. The default
  0.05% is the Binance futures taker fee. Use your spread and commission for
  gold or forex.
- `--slippage 0.0002`: extra adverse slippage per fill.
- `--start 2025-01-01 --end 2025-12-31`: test a sub-period.
- `--session 06:00`: try the same idea at another time.

The backtest prints a stats table and writes these files to
`six_am_breakout/results/`:

- `trades_<mode>.csv`: every position with entry, exit, reason, return and R
- `sessions_<mode>.csv`: one row per day with the 05:55 and 06:00 levels and
  the day's result
- `equity.png`: cumulative return per mode

## How the backtest fills orders

- A stop fills when price trades **through** its level. A touch does not fill
  it, so on the chart the 06:00 low that only touches the 05:55 low does not
  trigger the short. A bar that opens beyond the level fills at the open.
- Inside one bar, price is assumed to move open → low → high → close for a
  green bar and open → high → low → close for a red bar. `bars with 2+ fills`
  in the stats counts the bars where that assumption decided the order of
  fills. With 1-minute data this number should be tiny.
- Returns are per trade on a 1x position, after `fee` on both sides. `R` is the
  move in units of the 05:55 candle's range (the initial risk), before fees.

## Tests

```bash
python -m pytest six_am_breakout -q
```

The tests cover the chart example, every mode, reversals inside and after the
06:00 candle, gaps, timezones, weekends, fees, and the CSV formats. One test
also runs the strategy on a random walk and checks that it makes nothing
before costs, which guards against look-ahead bugs.
