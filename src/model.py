"""The freight rate model.

Shape of the model
------------------
``log(posted_rate) = log(distance) + drift(t) + booster(features)``

* ``log(distance)`` is an offset, not something the booster has to learn, so the
  booster only has to explain the *rate per mile*.
* ``drift(t)`` is a straight line fitted on the day-level residual. Rates drifted up
  ~6% between January and October and a tree cannot extrapolate past the last day it
  saw, so the drift is taken out before fitting and added back at prediction time.
  That is what lets the model price November and December instead of repeating
  October.
* the booster is a ``HistGradientBoostingRegressor`` on the remaining, time-free
  structure: distance, equipment, weight, geography and the daily market index.

Labels are screened twice before fitting because ~1.3% of ``posted_rate`` values are
corrupted (3-6x too high or 0.2-0.4x too low).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .config import (
    CATEGORICAL,
    FEATURES,
    HGB_PARAMS,
    LANE_RATIO_HIGH,
    LANE_RATIO_LOW,
    RESIDUAL_MAD_K,
    TREND_WINDOWS,
)


def lane_screen(frame: pd.DataFrame) -> np.ndarray:
    """Stage 1: flag labels far away from the median rate of the same lane+equipment."""
    median = frame.groupby(["pickup", "delivery", "equipment"])["posted_rate"].transform("median")
    ratio = frame["posted_rate"] / median
    return ratio.between(LANE_RATIO_LOW, LANE_RATIO_HIGH).to_numpy()


def robust_sigma(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def fit_drift(dates: pd.Series, residual: np.ndarray, epoch: pd.Timestamp) -> np.ndarray:
    """Average of straight lines fitted on the day-level residual over several windows.

    A single window is a coin flip - a short one chases noise, a long one is dragged
    down by January. Averaging the three fits in ``TREND_WINDOWS`` was consistently
    the most accurate and least biased option in the rolling backtest.
    """
    daily = pd.Series(residual).groupby(dates.to_numpy()).mean()
    days = (pd.DatetimeIndex(daily.index) - epoch).days.to_numpy(dtype=float)
    last = days.max()

    coefficients = []
    for window in TREND_WINDOWS:
        mask = np.ones(len(days), bool) if window is None else days > last - window
        design = np.column_stack([np.ones(mask.sum()), days[mask]])
        coefficients.append(np.linalg.lstsq(design, daily.to_numpy()[mask], rcond=None)[0])
    return np.mean(coefficients, axis=0)


@dataclass
class RateModel:
    params: dict = field(default_factory=lambda: dict(HGB_PARAMS))
    epoch: pd.Timestamp = pd.Timestamp("2025-01-01")

    booster: HistGradientBoostingRegressor | None = None
    drift: np.ndarray | None = None
    kept: np.ndarray | None = None
    log_sigma: float | None = None

    # ------------------------------------------------------------------ helpers
    def _new_booster(self, **overrides) -> HistGradientBoostingRegressor:
        params = dict(self.params)
        params.update(overrides)
        return HistGradientBoostingRegressor(
            categorical_features=[FEATURES.index(c) for c in CATEGORICAL], **params
        )

    def _drift_offset(self, t) -> np.ndarray:
        return self.drift[0] + self.drift[1] * np.asarray(t, dtype=float)

    def _target(self, frame: pd.DataFrame) -> np.ndarray:
        """Rate per mile in log space."""
        return np.log(frame["posted_rate"].to_numpy()) - frame["log_distance"].to_numpy()

    def _out_of_fold(self, frame: pd.DataFrame, target: np.ndarray, folds: int = 4) -> np.ndarray:
        prediction = np.empty(len(frame))
        for train_idx, test_idx in KFold(folds, shuffle=True, random_state=0).split(frame):
            booster = self._new_booster(max_iter=300)
            booster.fit(frame.iloc[train_idx][FEATURES], target[train_idx])
            prediction[test_idx] = booster.predict(frame.iloc[test_idx][FEATURES])
        return target - prediction

    def screen_and_drift(self, frame: pd.DataFrame, stage1: np.ndarray | None = None):
        """Screen the labels and measure the drift; returns the rows worth fitting on.

        Cross-fitted residuals do double duty here: their day-level mean is the drift,
        and their spread exposes the corrupted labels that the lane screen missed
        (those sitting on lanes too thin for a median to mean anything).
        """
        stage1 = lane_screen(frame) if stage1 is None else stage1
        screened = frame.loc[stage1].reset_index(drop=True)

        residual = self._out_of_fold(screened, self._target(screened))
        self.drift = fit_drift(screened["date"], residual, self.epoch)

        detrended = residual - self._drift_offset(screened["t"])
        self.log_sigma = robust_sigma(detrended)
        stage2 = np.abs(detrended - np.median(detrended)) < RESIDUAL_MAD_K * self.log_sigma

        self.kept = np.zeros(len(frame), bool)
        self.kept[np.flatnonzero(stage1)[stage2]] = True
        return screened.loc[stage2]

    # ---------------------------------------------------------------------- fit
    def fit(self, frame: pd.DataFrame) -> "RateModel":
        frame = frame.reset_index(drop=True)
        final = self.screen_and_drift(frame)
        target = self._target(final) - self._drift_offset(final["t"])
        self.booster = self._new_booster().fit(final[FEATURES], target)
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("call fit() first")
        log_rate = (
            self.booster.predict(frame[FEATURES])
            + frame["log_distance"].to_numpy()
            + self._drift_offset(frame["t"])
        )
        return np.exp(log_rate)


def metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt((error ** 2).mean())),
        "MAPE%": float((np.abs(error) / actual).mean() * 100),
        "R2": float(1 - (error ** 2).sum() / ((actual - actual.mean()) ** 2).sum()),
        "bias": float((predicted / actual).mean()),
    }
