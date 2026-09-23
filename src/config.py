"""Paths, constants and model hyper-parameters for the freight rate model."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TRAIN_CSV = DATA / "train_test.csv"
VALIDATION_CSV = DATA / "validation.csv"
TEMPLATE_CSV = DATA / "validation_predictions_template.csv"
DECEMBER_CSV = DATA / "december_chart_inputs.csv"

PREDICTIONS_CSV = ROOT / "validation_predictions.csv"
REPORTS = ROOT / "reports"

# Day 0 of the time index. Every date feature is expressed as days since here.
EPOCH = pd.Timestamp("2025-01-01")

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}

# Columns handed to the gradient booster, in a fixed order.
FEATURES = [
    "log_distance",
    "distance",
    "weight",
    "weight_missing",
    "equipment_code",
    "market_index",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "haversine",
    "circuity",
    "bearing",
    "mid_lat",
    "mid_lon",
]
CATEGORICAL = ["equipment_code"]

HGB_PARAMS = dict(
    loss="squared_error",
    max_iter=700,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)

# Stage 1 label screen: posted_rate / median(posted_rate) on the same lane+equipment.
LANE_RATIO_LOW = 0.5
LANE_RATIO_HIGH = 2.2
# Stage 2 label screen: |log residual| in robust sigmas.
RESIDUAL_MAD_K = 4.0

# Linear trend of the day-level residual is fitted on these look-back windows
# (None = every day available) and the three fits are averaged.
TREND_WINDOWS = (None, 150, 120)
