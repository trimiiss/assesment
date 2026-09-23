"""Feature engineering.

The model predicts a *rate per mile* in log space, so the feature set is built to
describe how expensive a mile is: how long the haul is, what is being pulled, where
in the country it runs, and what the market was doing that day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EPOCH, EQUIPMENT_CODES
from .data import haversine_miles


def build(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Return `frame` with every column in ``config.FEATURES`` plus helper columns."""
    out = frame.merge(daily, on="date", how="left")

    out["log_distance"] = np.log(out["distance"])
    out["equipment_code"] = out["equipment"].map(EQUIPMENT_CODES).astype("int8")

    # market_index is a daily level with a little row-level noise on top; a missing
    # value is replaced by that day's level.
    out["market_index"] = out["market_index"].fillna(out["market_index_day"])

    out["haversine"] = haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    # How much longer the billed mileage is than the straight line. Short city pairs
    # are floored at 70 miles, which this makes visible to the model.
    out["circuity"] = out["distance"] / out["haversine"].clip(lower=1.0)
    out["bearing"] = np.degrees(
        np.arctan2(out["delivery_lon"] - out["pickup_lon"], out["delivery_lat"] - out["pickup_lat"])
    )
    out["mid_lat"] = (out["pickup_lat"] + out["delivery_lat"]) / 2
    out["mid_lon"] = (out["pickup_lon"] + out["delivery_lon"]) / 2

    out["t"] = (out["date"] - EPOCH).dt.days
    return out
