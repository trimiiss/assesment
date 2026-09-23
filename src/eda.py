"""Exploration figures and the numbers quoted in the report.

    python -m src.eda

Writes PNGs to reports/figures/ and prints a short data-quality summary.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import REPORTS
from .data import clean, daily_market, load_train, load_validation
from .features import build
from .model import lane_screen

FIGURES = REPORTS / "figures"
INK, ACCENT, MUTED = "#064A56", "#C9722F", "#9DAFB3"


def _style(axis, title, ylabel, xlabel=None):
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    axis.set_ylabel(ylabel)
    if xlabel:
        axis.set_xlabel(xlabel)
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(MUTED)


def figure_timeline(train, validation, path):
    """Daily $/mile over the labelled period, with the straight-line drift."""
    daily = train.groupby("date").apply(
        lambda g: (g.posted_rate / g.distance).median(), include_groups=False)
    days = (daily.index - daily.index[0]).days.to_numpy(dtype=float)
    slope, intercept = np.polyfit(days, daily.to_numpy(), 1)

    figure, axis = plt.subplots(figsize=(11, 4.0), dpi=160)
    axis.plot(daily.index, daily.values, color=INK, lw=0.7, alpha=0.35, label="daily median")
    axis.plot(daily.index, daily.rolling(7, center=True).mean(), color=INK, lw=2.2,
              label="7-day mean")
    axis.plot(daily.index, intercept + slope * days, color=ACCENT, lw=2.0, ls="--",
              label=f"drift {slope * 365 / daily.mean():+.1%} per year")
    axis.axvline(pd.Timestamp("2025-10-31"), color=MUTED, ls=":", lw=1.4)
    axis.text(pd.Timestamp("2025-10-28"), daily.min(), "labels stop  ", color="#455A60",
              fontsize=9, ha="right")
    _style(axis, "Rate per mile drifts upwards all year - a tree cannot extrapolate that",
           "$ per mile")
    axis.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def figure_weekly(train, validation, path):
    """The market index is a daily level with a clean 7-day cycle, published through December."""
    market = pd.concat([train, validation]).groupby("date").market_index.mean()
    recent = market.loc["2025-12-01":"2025-12-28"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=160)
    by_dow = market.groupby(market.index.dayofweek).mean()
    axes[0].bar(range(7), by_dow.values, color=INK, alpha=0.85)
    axes[0].set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    axes[0].set_ylim(0.85, 1.15)
    _style(axes[0], "Market index by day of week (full year)", "mean market index")
    axes[1].plot(recent.index, recent.values, color=ACCENT, lw=2.0, marker="o", ms=3)
    _style(axes[1], "...and it is given for every December day", "market index")
    axes[1].tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def figure_distance(train, path):
    """Rate per mile decays with length of haul and shifts with equipment."""
    frame = train.assign(rpm=train.posted_rate / train.distance)
    frame = frame[lane_screen(train)]
    bands = pd.cut(frame.distance, [0, 250, 500, 750, 1000, 1500, 2000, 2500, 3500])
    table = frame.groupby([bands, "equipment"], observed=True).rpm.mean().unstack()

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.9), dpi=160)
    sample = frame.sample(4000, random_state=0)
    axes[0].scatter(sample.distance, sample.posted_rate, s=3, alpha=0.25, color=INK)
    _style(axes[0], "Rate is close to linear in distance", "posted rate ($)", "distance (miles)")
    for column, color in zip(table.columns, [INK, ACCENT, "#5A8F99"]):
        axes[1].plot([i.mid for i in table.index], table[column], marker="o", color=color, label=column)
    axes[1].legend(frameon=False, fontsize=9)
    _style(axes[1], "...but the price of a mile is not constant", "$ per mile", "distance (miles)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def figure_outliers(train, path):
    """~1.3% of labels are multiples or fractions of the lane's normal rate."""
    median = train.groupby(["pickup", "delivery", "equipment"]).posted_rate.transform("median")
    ratio = train.posted_rate / median
    figure, axis = plt.subplots(figsize=(6.2, 3.8), dpi=160)
    axis.hist(np.log10(ratio), bins=140, color=INK)
    axis.set_yscale("log")
    for edge in (0.5, 2.2):
        axis.axvline(np.log10(edge), color=ACCENT, ls="--", lw=1.4)
    axis.set_xticks(np.log10([0.2, 0.5, 1, 2.2, 5]), ["0.2x", "0.5x", "1x", "2.2x", "5x"])
    _style(axis, "Corrupted labels sit far outside the lane's normal rate",
           "loads (log scale)", "posted rate / median rate on the same lane + equipment")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    train, validation = load_train(), load_validation()

    figure_timeline(train, validation, FIGURES / "timeline.png")
    figure_weekly(train, validation, FIGURES / "weekly.png")
    figure_distance(train, FIGURES / "distance.png")
    figure_outliers(train, FIGURES / "outliers.png")
    print(f"figures written to {FIGURES}")

    good = lane_screen(train)
    daily = daily_market(train, validation)
    frame = build(clean(train), daily)
    print("\n--- data quality ---")
    print(f"train {len(train):,} loads {train.date.min().date()} to {train.date.max().date()}; "
          f"validation {len(validation):,} loads {validation.date.min().date()} to {validation.date.max().date()}")
    print(f"negative weights: {(train.weight < 0).sum()} train / {(validation.weight < 0).sum()} validation")
    print(f"missing weight: {train.weight.isna().sum()} / {validation.weight.isna().sum()}; "
          f"missing market_index: {train.market_index.isna().sum()} / {validation.market_index.isna().sum()}")
    print(f"labels outside 0.5x-2.2x of the lane median: {(~good).sum()} ({(~good).mean():.2%})")
    print(f"cities: {train.pickup.nunique()} in train, {validation.pickup.nunique()} in validation, "
          f"{len(set(validation.pickup) - set(train.pickup))} never seen in training")
    lanes_t = set(zip(train.pickup, train.delivery))
    lanes_v = set(zip(validation.pickup, validation.delivery))
    print(f"lanes: {len(lanes_t):,} in train, {len(lanes_v):,} in validation, {len(lanes_v - lanes_t):,} unseen")
    print(f"within-day sd of market_index {frame.groupby('date').market_index.std().mean():.4f} "
          f"vs across-day sd {frame.groupby('date').market_index.mean().std():.4f}")

    clean_train = train[good]
    rpm = clean_train.posted_rate / clean_train.distance
    print("\n--- rate per mile by equipment (screened labels) ---")
    print(rpm.groupby(clean_train.equipment).agg(["mean", "median", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
