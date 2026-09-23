"""Loading and cleaning of the freight data sets.

Cleaning decisions (all of them are justified in reports/report.md):

* ``weight`` carries sign-flipped values (-47,500 .. -8,000 lb). They are read as
  data-entry errors and restored with ``abs``.
* ``weight`` and ``market_index`` have a few hundred missing values each. Weight is
  left as NaN - the gradient booster routes missing values natively - and flagged.
  ``market_index`` is a daily market level plus row noise, so a missing value is
  filled with that day's mean, taken over train *and* validation features.
* ``posted_rate`` contains ~1.3% deliberately corrupted labels. They are screened in
  ``model.py`` rather than here, because the screen is part of fitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DECEMBER_CSV, TRAIN_CSV, VALIDATION_CSV

EARTH_RADIUS_MILES = 3958.8


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV, parse_dates=["date"])


def load_validation() -> pd.DataFrame:
    return pd.read_csv(VALIDATION_CSV, parse_dates=["date"])


def load_december() -> pd.DataFrame:
    return pd.read_csv(DECEMBER_CSV, parse_dates=["date"])


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    inner = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(inner))


def daily_market(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Mean market_index / quote_signal per calendar day.

    Both frames are used because this only touches *features*, which are given for
    every load we have to score - no label information crosses the split.
    """
    features = pd.concat(
        [train.drop(columns=["posted_rate"], errors="ignore"), validation],
        ignore_index=True,
    )
    return (
        features.groupby("date")
        .agg(market_index_day=("market_index", "mean"), quote_signal_day=("quote_signal", "mean"))
        .reset_index()
    )


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon) lookup, needed for the December rows which ship no coordinates."""
    parts = []
    for frame in frames:
        parts.append(frame[["pickup", "pickup_lat", "pickup_lon"]].rename(
            columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}))
        parts.append(frame[["delivery", "delivery_lat", "delivery_lon"]].rename(
            columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}))
    return pd.concat(parts, ignore_index=True).drop_duplicates("city").set_index("city")


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Repair the known data-quality defects in a feature frame."""
    out = frame.copy()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].abs()
    return out
