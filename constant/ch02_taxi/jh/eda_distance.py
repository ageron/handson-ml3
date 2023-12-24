#! /usr/bin/env python
# Copyright 2023 O1 Software Network. MIT licensed.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from constant.ch02_taxi.jh.etl import discard_outlier_rows
from constant.ch02_taxi.jh.features import COMPRESSED_DATASET, add_pickup_dow_hour

MAX_ELAPSED = 125 * 60  # 125 minutes, ~ two hours


def eda_distance(df: pd.DataFrame, ax: Axes, num_rows: int = 100_000) -> None:
    df = discard_outlier_rows(df)[:num_rows]
    df = add_pickup_dow_hour(df)

    sns.regplot(
        data=df[df.elapsed > 60],
        x="distance",
        y="elapsed",
        ax=ax,
        line_kws=dict(color="purple"),
    )
    ax.set_ylim(0, MAX_ELAPSED)


def eda_min_time(df: pd.DataFrame, ax: Axes) -> None:
    df["distance_km"] = (df.distance / 1000).apply(int)
    df = df.groupby("distance_km").elapsed.min().reset_index()

    # 13.3 m/s is 30 mph
    sns.regplot(
        data=df,
        x="distance_km",
        y="elapsed",
        ax=ax,
        line_kws=dict(color="purple"),
    )
    ax.set_ylim(0, MAX_ELAPSED)
    plt.tight_layout()
    plt.show()


def main(in_file: Path = COMPRESSED_DATASET, num_rows: int = 100_000) -> None:
    df = pd.read_parquet(in_file)[:num_rows]
    df = discard_outlier_rows(df)
    df = add_pickup_dow_hour(df)
    _, axes = plt.subplots(1, 2)

    eda_distance(df, axes[0])
    eda_min_time(df, axes[1])


if __name__ == "__main__":
    main()
