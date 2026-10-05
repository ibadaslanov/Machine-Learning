# SPX500 "Session + Frozen 20:00 Breakout": every holding-period variant, backtested

Data: FOREX.com SPX500 CFD, 1-hour candles, 2019-01-01 to 2026-03-05 (42,941 candles, UTC).
The CSV is not committed; point `SPX_CSV` at it (or put it in `data/FOREXCOM_SPX500_60.csv`).

## What was tested

The levels are exactly those of the Pine indicator:

- **In session** (start..end hour, UTC), the levels are the previous candle's high and low, plus the buffer if "Always".
- **Outside the session**, the levels are frozen at the end-hour candle's high/low ± buffer × its range.

On top of that, every reading of "the trade runs from one hour to the same hour next day" or "it starts on one day and runs until the week is over" was generated:

| Dimension | Values |
|---|---|
| Session start / freeze hour | all 24 × 24 combinations |
| Buffer | 0, 11 % overnight, 30 % overnight, 30 % always |
| Holding period | Always in (original), always in but flat on weekends, **daily cycle from hour R to R next day** (R = 0..23), **weekly from day D hour H to the end of the week** (Mon–Fri × 0..23 + Sunday open), **fixed N hours after entry** (N = 1, 2, 4, 8, 12, 24, 48, 120) |
| Entry rule | Stop-and-reverse on every break (original) · first break of the cycle and hold · first break, closed when the opposite level breaks |
| Entry window | any hour · only during the first 1 or 4 hours of the daily cycle · (fixed-hours mode) only on a single chosen hour |
| Direction | as signalled, **faded** (opposite side), plus long-only / short-only for the finalists |
| Double-break candles | 3 rules (against position first, previous candle colour, level nearest the open); every strategy is scored on its **worst** rule, because the hourly file can't tell which level broke first |

That is 8.1 million backtests, plus the mirrored fades. The engine (`engine/engine2.py`, numba) was re-implemented independently from a written spec and matched on 280 of 280 random configurations, every statistic and every trade. A first version had a bug in "first break, stop"; it was found by that check and fixed before these results.

**Costs used everywhere:**

- **Spread/commission:** 0.5 index points per round trip.
- **Overnight CFD financing:** price × (Fed funds + 2.5 %)/365 per night for longs, and (2.5 % − Fed funds) for shorts. This is an approximation; check your broker's rates.
- **Reopen candles:** no orders on the first candle after the daily or weekend break. From 2020 on, the feed records that candle's open as the previous close, so a fill price there would be fiction.

**Overfitting controls:**

- Strategies are chosen on 2019–2022 only (in-sample) and judged on 2023–2026 (out-of-sample).
- Each strategy's neighbouring settings are checked, so a single lucky setting doesn't count.
- A random-direction test asks whether the long/short choice adds anything beyond holding over the same hours.
- The deflated Sharpe ratio corrects for how many strategies were tried.

## Results

### 1. The original "always in, stop-and-reverse" idea loses in every variant

The original settings (13–20 UTC, freeze at 20:00, 11 %) over 2019–2026, after 0.5 pt and financing:

| Double-break rule | Trades | Net points |
|---|---|---|
| against position first (script default) | 10,560 | **−26,448** |
| previous candle colour | 9,120 | **−3,838** |
| level nearest the open | 9,108 | **−3,406** |
| most favourable possible order (not achievable) | 8,454 | **−181** |

Across all 576 session/freeze combinations × buffers, every stop-and-reverse family that may enter at any hour (always in, flat on weekends, daily cycles, weekly cycles, fixed hours) is positive out-of-sample in **0–2 %** of settings. Restricting entries to a 1- or 4-hour window helps only a little: 6–36 % of settings are positive, and every family's median is still negative. Too many trades and the whipsaw on double-break candles are the cause, and no choice of hours fixes it.

### 2. "First break, then hold" is where anything positive appears, but it is weak

Families, by median out-of-sample net (2023–2026, worst rule, after costs):

| Family | Settings | Median OOS | Share OOS > 0 |
|---|---|---|---|
| Daily cycle, first break, stop at opposite level | 50,688 | +25 | 52 % |
| Weekly cycle, first break, stop | 247,986 | −22 | 45 % |
| Weekly cycle, first break, hold to Friday | 247,986 | −263 | 37 % |
| Fixed 120 h hold, one entry hour | 46,693 | −457 | 39 % |
| Fixed 24 h hold, one entry hour ("same hour next day") | 50,259 | −762 | 25 % |
| Daily cycle, first break, hold to same hour next day | 50,688 | −861 | 28 % |
| Stop-and-reverse, entries at any hour | — | −4,000 to −10,000 | 0–2 % |
| Stop-and-reverse, entries only in a 1–4 h window | — | −150 to −1,250 | 6–36 % |

A typical setting in every family loses. Only specific settings are positive, and the question is whether those are skill or luck.

### 3. Finalists

These are the best in-sample settings that also survived out-of-sample and have positive neighbouring settings. Figures are after 0.5 pt per trade, financing and the worst double-break rule. "Always long, same hours" is the gross P&L of simply being long during exactly the same holding windows.

| | Strategy (all hours UTC) | Net 2019–26 | In-sample / OOS | Positive years | Trades | Max DD | Long / short net | Gross vs always long, same hours | Sharpe | DSR (100 / 1000 tries) |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | Freeze the **08:00 candle**; first break after **09:00**, hold until 09:00 next day (S=15, F=8, no buffer) | +4,822 | 3,207 / 1,615 | 6 / 8 | 2,231 | 1,084 | +4,456 / +367 | 6,469 vs 4,796 | 0.75 | 0.38 / 0.16 |
| **B** | **Fade** the first break after **06:00** (frozen 18:00 candle ±30 %, then hourly), hold until 06:00 next day | +6,655 | 4,552 / 2,102 | 5 / 8 (2023, 2024 negative) | 2,182 | 1,142 | +5,408 / +1,247 | 8,167 vs 4,551 | 1.10 | 0.78 / 0.50 |
| **C** | Only the **15:00 candle** may trigger (frozen 12:00 candle ±30 %); hold **24 h** | +3,677 | 2,515 / 1,161 | 6 / 8 | 1,682 | 1,050 | +2,902 / +775 | 4,998 vs 2,732 | 0.67 | 0.29 / 0.10 |
| **D** | **Weekly**: first break from **Monday 10:00** (frozen 08:00 candle ±30 %), hold to Friday close | +4,806 | 3,428 / 1,375 | 8 / 8, but 2024 = +3 and 2025 = +10 | 374 | 1,723 | +4,218 / +588 | 5,541 vs 4,311 | 1.01 | 0.68 / 0.39 |
| **E** | Only the **10:00 candle** may trigger (frozen 08:00 candle ±30 %); hold **120 h** | +4,762 | 3,709 / 1,049 | 7 / 8, 2024–26 ≈ 0 | 363 | 1,233 | +3,578 / +1,184 | 5,356 vs 2,901 | 1.05 | 0.73 / 0.45 |
| — | Buy and hold (no financing) | +4,364 | — | — | 1 | — | — | — | 0.73 | — |

Read with care:

- **None passes the multiple-testing bar.** A deflated Sharpe ratio above 0.95 is the usual standard. With millions of settings tried, even assuming only 100 truly independent ones, the best scores 0.78.
- **Most of the profit is on the long side** in a market that rose about 175 %. The direction choice does beat random direction (p < 0.003 for all five) and beats being long over the same hours. But the short side adds little, except in fade B.
- **The out-of-sample profit is concentrated in one year:**
  - B: 2025 contributes +2,491 of its +2,102 out-of-sample total (2023 and 2024 lost).
  - D and E: 2023 contributes 1,210 of D's 1,375 and 991 of E's 1,049.
- **D and E work only in US summer time.** D makes +5,813 in summer and −1,006 in winter; E makes +4,918 and −156. Hours fixed in UTC land on different points of the trading day once the clocks change.
- **A depends heavily on how reopen candles are filled.** Its result before financing ranges from +7,673 (trusting the recorded open) to +2,354 (worst fill on reopen candles); the figure above skips them.
- **Re-choosing the best setting each year doesn't work.** Picking the best of the previous two years and trading it the following year, 2021–2026, totals −485 points for normal strategies and +715 for fades (−986 / +956 averaged over the top 20). Last period's winner does not reliably win next.

### 4. Bottom line

- **Always in the market, stop-and-reverse:** loses money in every variant once realistic costs are included. Don't trade it.
- **First break, then hold, closing at the same hour next day or at week end:** some settings made money from 2019 to 2026. None of them is statistically convincing once you account for how many variants were searched, and most of the profit comes from the long side in a strong bull market.
- **Worth forward-testing (paper trade first):** A (simple, many trades, 6 of 8 years positive) and B (fade, the best risk-adjusted result, but with two losing years in a row). Run them in TradingView with the Bar Magnifier on so double-break candles are resolved with real intrabar data. Then compare live fills with the backtest before risking money.

## Files

- `pine/session_frozen_breakout_cycles.pine`: TradingView `strategy()` with every mode above as inputs, plus presets for A–E and the original. It places real stop orders (limit orders when fading) at the next candle's levels. Turn on the Bar Magnifier (Premium) to resolve double-break candles. Add your spread under *Properties → Slippage/Commission*. Financing is not modelled by TradingView.
- `engine/engine2.py`: backtest engine; `engine/rungrid2.py` runs the full grid (about 20 min on 4 cores, writing a 900 MB result array); `engine/analyse2.py`, `plateau.py`, `deep2.py` and `dsr.py` produce the tables above.
- `engine/engine.py`, `engine/bt.py`: the first engine and the line-by-line Pine port it was validated against.
- `results/`: JSON outputs (family statistics, top lists, walk-forward, finalist deep-dives, deflated Sharpe).

Known differences between the Pine strategy and the Python engine:

- **Double-break candles:** TradingView resolves them with real intrabar data, while the engine assumes the worst order.
- **Missed cycle boundary:** if a daily-cycle boundary falls inside the market break, the Pine script closes one candle later.
- **Re-entry on the exit candle:** in fixed-hours mode, the Pine script doesn't re-enter on the same candle a time-exit fills.
- **Reopen candles:** the engine detects them from the data; the Pine script uses `session.islastbar` to skip the first candle after each break.
