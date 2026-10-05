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
| Direction | as signalled, **faded** (opposite side), plus long-only runs for finalists A, D and E |
| Double-break candles | 3 rules (against position first, previous candle colour, level nearest the open); every strategy is scored on its **worst** rule, because the hourly file can't tell which level broke first |

That is 8.1 million backtests, plus the mirrored fades. The engine (`engine/engine2.py`, numba) was re-implemented independently from a written spec and matched on 280 of 280 configurations (250 random, 30 hand-picked), every statistic and every trade. A first version had a bug in "first break, stop"; it was found by that check and fixed before these results.

**Costs used everywhere:**

- **Spread/commission:** 0.5 index points per round trip.
- **Overnight CFD financing:** price × (Fed funds + 2.5 %)/365 per night for longs, and (2.5 % − Fed funds) for shorts. This is an approximation; check your broker's rates.
- **Reopen candles:** no orders on the first candle after the daily or weekend break. The feed records that candle's open as the previous close (in 62 % of 2019 reopen candles and almost all of them from 2020 on), so a fill price there would be fiction.

**Overfitting controls:**

- Settings are ranked on 2019–2022 (in-sample). The 2023–2026 results were then also used to pick the finalists, so they are not a clean holdout (see section 3).
- Each finalist's neighbouring settings are checked, so a single lucky setting doesn't count.
- Each strategy is compared with simply being long over the same hours, and with random directions.
- The deflated Sharpe ratio corrects for how many strategies were tried.

## Results

### 1. The original "always in, stop-and-reverse" idea loses in almost every variant

The original settings (13–20 UTC, freeze at 20:00, 11 %) over 2019–2026, after 0.5 pt and financing:

| Double-break rule | Trades | Net points |
|---|---|---|
| against position first (script default) | 10,560 | **−26,448** |
| previous candle colour | 9,120 | **−3,838** |
| level nearest the open | 9,108 | **−3,406** |
| most favourable possible order (not achievable) | 8,454 | **−181** |

These are the settings checked: 528 session/freeze combinations (freeze hours 21 and 22 excluded, because the daily market break leaves those levels stale for weeks) × 4 buffers. Results are for 2019–26, after costs, under the worst double-break rule:

- **Always in, or always in but flat on weekends:** 0 of 4,224 settings made money.
- **Daily-cycle stop-and-reverse, entries at any hour:** 0 of 50,688 made money.
- **Weekly stop-and-reverse:** 2,182 of 248,892 (0.9 %) made money, the best +1,356.
- **Faded always-in:** 49 of 4,224 made money, the best +2,138.
- **Daily stop-and-reverse with only a 1-hour entry window:** 21 % made money, the best +2,950. Its median setting still lost out-of-sample.

Too many trades and the whipsaw on double-break candles are the cause; no choice of hours fixes it.

### 2. First-break entries do best, but the edge is weak

Families, by median out-of-sample net (2023–2026, worst rule, after costs):

| Family | Settings | Median OOS | Share OOS > 0 |
|---|---|---|---|
| Daily cycle, first break, stop at opposite level | 50,688 | +25 | 52 % |
| Weekly cycle, first break, stop | 247,986 | −22 | 45 % |
| Weekly cycle, first break, hold to Friday | 247,986 | −263 | 37 % |
| Fixed 120 h hold, one entry hour | 46,693 | −457 | 39 % |
| Fixed 24 h hold, one entry hour ("same hour next day") | 50,259 | −762 | 25 % |
| Daily cycle, first break, hold to same hour next day | 50,688 | −861 | 28 % |
| Stop-and-reverse, entries only in a 1–4 h window | — | −150 to −1,250 | 6–36 % |
| Stop-and-reverse, entries at any hour | — | −4,200 to −10,200 | 0–2 % |

A typical setting loses in every family except "daily cycle, first break, stop", whose median is break-even (+25). Only specific settings are positive, and the question is whether those are skill or luck.

### 3. Finalists

**How they were picked:**

- **B, C, D:** the best 2019–22 setting of their family that was also positive in 2023–26.
- **A:** picked by hand from the round-1 results. It ranks about 800th in-sample within its family, and almost all the settings above it were also positive in 2023–26.
- **E:** the best 2019–22 setting of its family. It turned out to be largely the same trade as D.

Only 16 of 62 family winners stayed positive in 2023–26. Because that filter used the 2023–26 data, the OOS column below is not a true holdout.

All figures are after 0.5 pt per trade, financing and the worst double-break rule.

| | Strategy (all hours UTC) | Net 2019–26 | 2019–22 / 2023–26 | Positive years | Trades | Max DD | Sharpe | DSR, 100 / 1000 tries |
|---|---|---|---|---|---|---|---|---|
| **A** | First break of the **08:00 candle's high/low** from 09:00 to 14:00; from 15:00 on, the previous hour's high/low. Hold until **09:00 next day** (S=15, F=8, no buffer). Includes a Sunday-evening trade held to Monday 09:00. | +4,822 | 3,207 / 1,615 | 6 / 8 | 2,231 | 1,084 | 0.75 | 0.38 / 0.16 |
| **B** | **Fade** the first break after 06:00 of the previous day's 18:00 candle ±30 %. From 08:00 to 18:00 the previous hour's high/low is used. Hold until **06:00 next day**. Most entries fill at the 06:00 open. | +6,655 | 4,552 / 2,102 | 5 / 8 (2019, 2023, 2024 negative) | 2,182 | 1,142 | 1.03 | 0.71 / 0.43 |
| **C** | Only the **15:00 candle** may trigger (frozen 12:00 candle ±30 %); hold **24 h** | +3,677 | 2,515 / 1,162 | 6 / 8 | 1,682 | 1,050 | 0.67 | 0.29 / 0.10 |
| **D** | **Weekly**: first break from **Monday 10:00** (frozen 08:00 candle ±30 %), hold to Friday close | +4,806 | 3,428 / 1,379 | 8 / 8, but 2024 = +3 and 2025 = +10 | 374 | 1,723 | 1.01 | 0.68 / 0.39 |
| **E** | The first **10:00 candle** of the week that breaks the frozen 08:00 candle ±30 % triggers. Hold to the Friday close; the 120 h limit is never reached. **Largely the same trade as D.** | +4,762 | 3,709 / 1,052 | 7 / 8, 2024–26 ≈ 0 | 363 | 1,233 | 1.05 | 0.73 / 0.45 |
| — | Buy and hold (CFD long) | +4,364 gross, **+2,627 after financing** | 1,338 / 3,026 gross; **1,799 after financing** | — | 1 | — | 0.73 | — |

Out of sample (2023–26), compared with simply being long in the same holding windows:

| | A | B | C | D | E |
|---|---|---|---|---|---|
| Strategy, gross | 2,416 | 2,785 | 1,812 | 1,782 | 1,366 |
| Always long in the same windows, gross | **2,829** | 2,755 | 1,284 | **2,648** | **1,816** |
| Short trades, net | −329 | −255 | +189 | −368 | −172 |

What this means:

1. **The long/short choice added nothing out of sample, except in C.** Being long in the same hours made more than A, D and E, and about the same as B. The short trades lost money in 2023–26 in every finalist but C. Over 2019–26 the direction choice does beat random directions (p ≤ 0.002 for all five), but that period includes the years the settings were chosen on. On 2023–26 alone, p = 0.03–0.14.
2. **A plain CFD long beat A, C, D and E from 2023 to 2026.** It made +1,799 after financing; only B did better.
3. **None passes the multiple-testing bar,** which is a deflated Sharpe ratio (DSR) above 0.95. Even assuming only 100 truly independent settings were tried, the best scores 0.73.
   - That DSR is also optimistic. It uses the variance of a single Sharpe estimate (1/T), but the spread of Sharpe ratios measured across this grid is 2–3 times larger.
   - On that basis, DSR at 100 tries is 0.05–0.3 for every finalist.
4. **The out-of-sample profit is concentrated:**
   - **B:** 2025 contributes +2,491 of its +2,102 OOS total.
   - **D:** 2023 contributes 1,210 of its 1,379.
   - **E:** 2023 contributes 991 of its 1,052.
5. **D and E work only in US summer time.** D makes +5,813 in summer and −1,006 in winter; E makes +4,918 and −156. Hours fixed in UTC land on a different part of the trading day once the clocks change.
6. **A depends heavily on how reopen candles are filled.** Its result before financing ranges from +7,673 (trusting the recorded open) to +2,354 (worst fill on reopen candles); the figure above skips them.
7. **Re-choosing the best setting each year doesn't work.** Picking the best of the previous two years and trading it the following year, 2021–2026, totals −485 points for normal strategies (−986 averaged over the top 20). For fades it's +715 (+956 for the top 20), but that is before the fades' own financing, so the true figure is lower. Last period's winner does not reliably win next.
8. **Long-only versions** of A, D and E made +3,256, +2,675 and +2,242 over 2019–26 after costs and financing, against +2,627 for buy and hold. Their Sharpe ratios (about 0.5–0.6) are below buy and hold's 0.73.

How the Sharpe ratio is computed: daily P&L divided by price, annualised with √252. Sunday-evening sessions count as days, so the true figures are about 10 % higher for every strategy alike; the ranking is unaffected.

### 4. Bottom line

- **Always in the market, stop-and-reverse (your original idea):** loses after realistic costs in every always-in and daily variant. Under 1 % of the weekly or faded versions made money, and those are isolated settings. Don't trade it.
- **First break, then hold to the same hour next day, or until the week ends:** some settings made money from 2019 to 2026. None of them is statistically convincing once you count how many variants were searched. Out of sample, most did no better than just being long over the same hours, and a plain buy-and-hold CFD beat four of the five.
- **If you want to keep going, forward-test (paper trade) only B** (fade the first break after 06:00, hold to 06:00 next day). It is the only finalist that beat both buy and hold and same-window long out of sample, but with three losing years. **C** is the only one whose long/short choice beat being long out of sample, though its total was small (+1,162).
  - Run them in TradingView with the Bar Magnifier on, so double-break candles use real intrabar data.
  - Compare live fills with the backtest before risking money.

## Files

- `pine/session_frozen_breakout_cycles.pine`: TradingView `strategy()` with every mode above as inputs, plus presets for A–E and the original. It places real stop orders (limit orders when fading) at the next candle's levels. Turn on the Bar Magnifier (Premium) to resolve double-break candles. Add your spread under *Properties → Slippage/Commission*. Financing is not modelled by TradingView.
- `engine/engine2.py`: backtest engine; `engine/rungrid2.py` runs the full grid (about 20 min on 4 cores, writing a 900 MB result array); `engine/analyse2.py`, `plateau.py`, `deep2.py` and `dsr.py` produce the tables above.
- `engine/engine.py`, `engine/bt.py`: the first engine and the line-by-line Pine port it was validated against.
- `results/`: JSON outputs (family statistics, top lists, walk-forward, finalist deep-dives, deflated Sharpe).

Known differences between the Pine strategy and the Python engine:

- **Double-break candles:** TradingView resolves them with real intrabar data, while the engine assumes the worst order.
- **Missed cycle boundary:** if a daily-cycle boundary or a week end falls inside a market break (or on a holiday or early close), the Pine script closes at the close of the first candle after it, one candle later than the engine.
- **Touching a level:** TradingView fills a stop or limit when price touches the level; the engine needs it to trade beyond the level.
- **Reopen candles:** the engine detects them from the data; the Pine script uses `session.islastbar` to skip the first candle after each break.
