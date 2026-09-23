"""Rolling-origin backtest.

The real task is a two-month-ahead forecast: the labelled data stops on 2025-10-31
and the loads to price run from 2025-11-01 to 2025-12-31. A random K-fold would let
the model peek at the future, so the split used here copies the real one - train on
everything before a cut-off date, score the next 61 days - at three cut-offs.

Holdout rows are reported twice: over every row, and over the rows whose label
passes the same corruption screen used in training. The first number is dominated by
the ~1.3% corrupted labels (no model can predict a label that was scrambled after the
fact); the second is what the model can actually be held to.

    python -m src.backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EPOCH, REPORTS
from .data import clean, daily_market, load_train, load_validation
from .features import build
from .model import RateModel, lane_screen, metrics

CUTOFFS = ["2025-07-01", "2025-08-01", "2025-09-01"]
HORIZON_DAYS = 61


def lane_median_baseline(fit: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """Business-as-usual benchmark: the median $/mile of the lane x equipment."""
    rate_per_mile = fit["posted_rate"] / fit["distance"]
    lane = rate_per_mile.groupby([fit["pickup"], fit["delivery"], fit["equipment"]]).median()
    equipment = rate_per_mile.groupby(fit["equipment"]).median()
    keys = zip(holdout["pickup"], holdout["delivery"], holdout["equipment"])
    per_mile = np.array([lane.get(k, equipment[k[2]]) for k in keys])
    return per_mile * holdout["distance"].to_numpy()


def run() -> pd.DataFrame:
    train = load_train()
    frame = build(clean(train), daily_market(train, load_validation()))
    frame["corrupt_label"] = ~lane_screen(frame)

    rows = []
    for cutoff in CUTOFFS:
        cut = pd.Timestamp(cutoff)
        fit = frame[frame["date"] < cut]
        holdout = frame[(frame["date"] >= cut) & (frame["date"] < cut + pd.Timedelta(days=HORIZON_DAYS))]
        good = ~holdout["corrupt_label"].to_numpy()
        actual = holdout["posted_rate"].to_numpy()

        model = RateModel(epoch=EPOCH).fit(fit)
        predicted = model.predict(holdout)
        baseline = lane_median_baseline(fit[~fit["corrupt_label"]], holdout)

        for name, values in (("lane median $/mile", baseline), ("gradient boosting", predicted)):
            rows.append({
                "cutoff": cutoff, "model": name, "n": len(holdout),
                **{f"{k} (all)": v for k, v in metrics(actual, values).items()},
                **{f"{k} (clean)": v for k, v in metrics(actual[good], values[good]).items()},
            })
        print(f"trained to {cut.date()}  drift/day {model.drift[1]:+.6f}  "
              f"labels kept {model.kept.mean():.1%}  holdout {len(holdout):,} rows", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    results = run()
    show = ["cutoff", "model", "MAE (clean)", "MAPE% (clean)", "RMSE (clean)", "R2 (clean)",
            "bias (clean)", "MAE (all)", "RMSE (all)"]
    pd.set_option("display.width", 200)
    print("\n=== two-month-ahead holdouts ===")
    print(results[show].round(3).to_string(index=False))
    print("\n=== averaged over the three cut-offs ===")
    print(results.groupby("model")[show[2:]].mean().round(3).to_string())

    REPORTS.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(REPORTS / "backtest_results.csv", index=False)
    print(f"\nwritten to {REPORTS / 'backtest_results.csv'}")


if __name__ == "__main__":
    main()
