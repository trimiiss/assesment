"""Ablation: what each piece of the pipeline is worth.

Same rolling-origin protocol as ``src.backtest``, with exactly one piece of the
shipped model changed at a time. The numbers quoted in the report come from here.

    python -m src.ablation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .backtest import CUTOFFS, HORIZON_DAYS
from .config import CATEGORICAL, EPOCH, FEATURES, HGB_PARAMS, REPORTS
from .data import clean, daily_market, load_train, load_validation
from .features import build
from .model import RateModel, lane_screen, metrics


class LaneScreenOnly(RateModel):
    """Stage 1 of the label screen only - no residual pass."""

    def fit(self, frame):
        frame = frame.reset_index(drop=True)
        self.screen_and_drift(frame)  # for the drift
        stage1 = lane_screen(frame)
        self.kept = stage1
        final = frame.loc[stage1]
        target = self._target(final) - self._drift_offset(final["t"])
        self.booster = self._new_booster().fit(final[FEATURES], target)
        return self


class NoLabelScreen(RateModel):
    """Every label trusted, drift handling unchanged."""

    def fit(self, frame):
        frame = frame.reset_index(drop=True)
        self.screen_and_drift(frame)
        self.kept = np.ones(len(frame), bool)
        target = self._target(frame) - self._drift_offset(frame["t"])
        self.booster = self._new_booster().fit(frame[FEATURES], target)
        return self


class DateAsFeature(RateModel):
    """Label screening unchanged, but the date is a plain feature instead of a drift."""

    columns = FEATURES + ["t"]

    def fit(self, frame):
        frame = frame.reset_index(drop=True)
        final = self.screen_and_drift(frame)
        self.drift = np.array([0.0, 0.0])
        self.booster = HistGradientBoostingRegressor(
            categorical_features=[self.columns.index(c) for c in CATEGORICAL], **HGB_PARAMS
        ).fit(final[self.columns], self._target(final))
        return self

    def predict(self, frame):
        return np.exp(self.booster.predict(frame[self.columns]) + frame["log_distance"].to_numpy())


class NoTimeAtAll(DateAsFeature):
    """No drift and no date - the model prices an average day."""

    columns = FEATURES


VARIANTS = {
    "shipped model": RateModel,
    "no residual label screen": LaneScreenOnly,
    "no label screening at all": NoLabelScreen,
    "date as a plain feature": DateAsFeature,
    "no time information": NoTimeAtAll,
}


def main() -> None:
    train = load_train()
    frame = build(clean(train), daily_market(train, load_validation()))
    frame["corrupt_label"] = ~lane_screen(frame)

    rows = []
    for name, factory in VARIANTS.items():
        for cutoff in CUTOFFS:
            cut = pd.Timestamp(cutoff)
            fit = frame[frame["date"] < cut]
            holdout = frame[(frame["date"] >= cut) & (frame["date"] < cut + pd.Timedelta(days=HORIZON_DAYS))]
            good = ~holdout["corrupt_label"].to_numpy()
            predicted = factory(epoch=EPOCH).fit(fit).predict(holdout)
            rows.append({"variant": name, "cutoff": cutoff,
                         **metrics(holdout["posted_rate"].to_numpy()[good], predicted[good])})
        print(f"{name:28s} done", flush=True)

    results = pd.DataFrame(rows)
    summary = results.groupby("variant", sort=False)[["MAE", "MAPE%", "RMSE", "bias"]].mean()
    pd.set_option("display.width", 160)
    print("\n=== averaged over the three two-month-ahead cut-offs (clean labels) ===")
    print(summary.round(3).to_string())

    REPORTS.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(REPORTS / "ablation_results.csv", index=False)
    print(f"\nwritten to {REPORTS / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
