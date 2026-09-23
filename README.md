# Freight Rate Prediction — Spotter ML Assessment

Predicts `posted_rate` for the 12,000 loads in `data/validation.csv` (November–December
2025) from 48,000 labelled loads covering January–October 2025.

**Headline:** on two-month-ahead holdouts the model lands at **1.8% MAPE / $44 MAE**
on non-corrupted labels — 2.3× better than a lane-median $/mile benchmark, and within
a hair of the noise floor of the data (1.6% MAPE when predicting *within* the training
period).

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.10+ , no compiled dependencies beyond scikit-learn.

## Run

```bash
# 1. exploration figures + data-quality summary        -> reports/figures/
python -m src.eda

# 2. rolling-origin backtest (the honest score)        -> reports/backtest_results.csv
python -m src.backtest

# 2b. what each piece of the pipeline is worth         -> reports/ablation_results.csv
python -m src.ablation

# 3. train on everything, write both deliverables
python -m src.predict                                  # -> validation_predictions.csv
                                                       #    data/december_chart_inputs.csv

# 4. provided scorer: validates both files, draws the chart
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
```

Step 3 takes about a minute, the ablation about four; nothing else takes more than a few seconds.

## What the model does

`log(posted_rate) = log(distance) + drift(date) + booster(features)`

| Piece | Why |
| --- | --- |
| `log(distance)` as a fixed offset | Rate is ~linear in distance (r = 0.91). Modelling `$/mile` instead of `$` lets the booster spend its capacity on *how expensive a mile is* rather than re-learning the multiplication. |
| `drift(date)` — a straight line fitted on the day-level residual | Rates drifted up ~7%/year. Trees cannot extrapolate past the last date they saw, so the drift is removed before fitting and added back when predicting. This is what makes November and December forecasts instead of copies of October. |
| `HistGradientBoostingRegressor` | Non-linear, handles NaN natively, no scaling or one-hot needed, trains in seconds. Beat a linear/lane-median baseline by 2.3× and deeper/shallower variants made no difference. |

15 features: distance and log distance, weight (+ missing flag), equipment, the daily
market index, both endpoint coordinates, great-circle distance, circuity, bearing and
route midpoint. Coordinates rather than city names matter — 8 cities and 736 lanes in
the validation set never appear in training.

Labels are screened twice before fitting (~1.7% dropped): once against the median rate
of the same lane + equipment, then once against the model's own residuals at 4 robust
sigmas. See `reports/report.docx` for why.

## Layout

```
src/config.py      paths, feature list, hyper-parameters
src/data.py        loading + cleaning (sign-flipped weights, missing values, coordinates)
src/features.py    feature engineering
src/model.py       label screening, drift estimation, the booster
src/backtest.py    rolling-origin evaluation
src/ablation.py    single-factor ablation - what each piece of the pipeline is worth
src/predict.py     final fit + the two output files
src/eda.py         exploration figures
reports/           report.docx + report.pdf, figures, backtest and ablation results,
                   Loom talking points
```

## Outputs

* `validation_predictions.csv` — 12,000 rows of `load_id,predicted_rate`
* `data/december_chart_inputs.csv` — `predicted_rate` filled for all 31 December days
* `scorer_results/candidate_december.png` — produced by the provided `score.py`
