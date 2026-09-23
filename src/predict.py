"""Train on every labelled load and write the two deliverables.

    python -m src.predict

Writes ``validation_predictions.csv`` (12,000 rows, load_id + predicted_rate) and
fills the ``predicted_rate`` column of ``data/december_chart_inputs.csv``.

The December rows only carry pickup, delivery, distance, equipment, weight and date,
so two inputs are reconstructed:

* coordinates - looked up from the city table both data sets share;
* market_index - the given daily level for that December day, averaged over the
  validation loads that ship on it. It is a feature, published for every day of
  December, so using it leaks nothing; it is also what makes the chart move with the
  weekly market cycle instead of being a flat line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DECEMBER_CSV, EPOCH, PREDICTIONS_CSV, TEMPLATE_CSV
from .data import city_coordinates, clean, daily_market, load_december, load_train, load_validation
from .features import build
from .model import RateModel


def december_frame(december: pd.DataFrame, coordinates: pd.DataFrame) -> pd.DataFrame:
    """Give the December rows the columns the feature builder expects."""
    out = december.copy()
    for side in ("pickup", "delivery"):
        out[f"{side}_lat"] = out[side].map(coordinates["lat"])
        out[f"{side}_lon"] = out[side].map(coordinates["lon"])
    missing = out[["pickup_lat", "delivery_lat"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"no coordinates for {sorted(set(out.loc[missing, 'pickup']))}")
    out["market_index"] = np.nan  # filled from the daily level in features.build
    return out


def main() -> None:
    train, validation, december = load_train(), load_validation(), load_december()
    daily = daily_market(train, validation)
    coordinates = city_coordinates(train, validation)

    model = RateModel(epoch=EPOCH).fit(build(clean(train), daily))
    print(f"trained on {model.kept.sum():,} of {len(train):,} labelled loads "
          f"({1 - model.kept.mean():.2%} screened out as corrupted labels)")
    print(f"drift {model.drift[1] * 30:+.3%} per 30 days, residual sigma {model.log_sigma:.4f} in log space")

    # ---- 12,000 validation loads -------------------------------------------------
    predicted = model.predict(build(clean(validation), daily))
    template = pd.read_csv(TEMPLATE_CSV)
    output = template[["load_id"]].merge(
        pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": predicted.round(2)}),
        on="load_id", how="left",
    )
    if output["predicted_rate"].isna().any() or len(output) != 12_000:
        raise ValueError("validation predictions do not line up with the template")
    output.to_csv(PREDICTIONS_CSV, index=False)
    print(f"wrote {PREDICTIONS_CSV.name}: {len(output):,} rows, "
          f"mean ${output.predicted_rate.mean():,.2f}, "
          f"range ${output.predicted_rate.min():,.2f}-${output.predicted_rate.max():,.2f}")

    # ---- 31 fixed December rows --------------------------------------------------
    frame = build(clean(december_frame(december, coordinates)), daily)
    december["predicted_rate"] = model.predict(frame).round(2)
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    december.to_csv(DECEMBER_CSV, index=False)
    print(f"wrote {DECEMBER_CSV.name}: 31 rows, "
          f"${december.predicted_rate.min():,.2f}-${december.predicted_rate.max():,.2f}")


if __name__ == "__main__":
    main()
