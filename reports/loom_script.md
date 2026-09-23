# Loom talking points (2–3 minutes)

Five beats, ~30 seconds each. Numbers are all reproducible from `python -m src.backtest`
and `python -m src.ablation`.

---

## 0:00 — Framing (15s)

> "48,000 labelled loads, January to October. The 12,000 I have to price are November
> and December. So this isn't a random holdout — it's a two-month-ahead forecast, and
> that one fact drove every decision I made."

## 0:15 — Key findings from the data (35s)

Screen share: `reports/figures/distance.png`.

> "Rate is almost linear in distance — correlation 0.91. But the price *of a mile*
> isn't constant: it falls from about $2.80 on short hauls to $1.90 over 2,500 miles,
> and Reefer runs about 13% above Dry Van at every length. So I model dollars per mile
> in log space and multiply distance back in, rather than making the model re-learn
> multiplication."

Switch to `reports/figures/timeline.png`.

> "Second finding: rates drifted up all year, about 7% annualised. And `market_index`
> turns out to be a daily market level with a clean seven-day cycle — its daily mean
> correlates 0.70 with what's left after distance and equipment are accounted for."

## 0:50 — Data quality (35s)

Screen share: `reports/figures/outliers.png`.

> "Four issues. 292 weights come through negative — sign errors, the magnitudes are
> normal, so I take the absolute value. A few hundred missing weights and market
> indexes — weight I leave as NaN because the booster handles it natively, market
> index I fill with that day's level.
>
> The big one is this chart: divide each rate by the median rate on the same lane and
> you get three separate clumps. A tight core at 1×, and two blobs at a fifth and at
> four times. That's ~1.3% of labels scrambled. I screen them twice — lane median
> first, then the model's own residuals at four robust sigmas, which catches the ones
> on thin lanes. That's worth 1.15 points of MAPE.
>
> And `quote_signal` is a decoy. It looks predictive on raw averages, but that's the
> corrupted rows sitting in its tails — control for distance and market and the signal
> is gone. Adding it made the holdout worse, so it's out."

## 1:25 — Model and why (30s)

> "`log(rate) = log(distance) + drift(date) + booster(features)`.
>
> Gradient-boosted trees — HistGradientBoosting — because the relationships are
> non-linear, it handles missing values natively, and it trains in seconds.
>
> The important part is the drift term. A tree can't extrapolate: feed it a date and
> it flattens at the last day it saw, which would make November and December a copy of
> October. So I fit a straight line to the day-level residual, subtract it before
> training, add it back when predicting. That alone is worth 1.25 points of MAPE."

## 1:55 — Validation and split (35s)

> "A random K-fold would leak: loads on nearby days share a market level, so the model
> would see the future and the score would flatter itself.
>
> Instead — rolling origin. Train on everything before a cut-off, score the next 61
> days, repeat at three cut-offs: July, August, September. That's the real task, three
> times.
>
> Result: 1.8% MAPE, $44 MAE, against 4.1% for a lane-median-dollars-per-mile
> benchmark. And for scale — predicting *inside* the training period with a random
> split gives 1.6%. So the forecast is within 0.2 points of the noise floor of this
> data. There isn't much left on the table."

## 2:30 — Code walkthrough (30s)

Screen share the repo tree.

> "Six modules. `data.py` loads and repairs. `features.py` builds the 15 features —
> note coordinates rather than city names, because 8 cities and 736 lanes in the
> validation set never appear in training. `model.py` is the interesting one:
> `screen_and_drift` does the two-pass label screen and measures the drift off
> cross-fitted residuals, then `fit` trains the booster on the detrended target.
> `backtest.py` is the rolling-origin harness, `ablation.py` proves each piece earns
> its place, `predict.py` writes the two output files.
>
> December comes out at $841 to $865 on that fixed lane — the weekly cycle plus the
> drift. Thanks for watching."
