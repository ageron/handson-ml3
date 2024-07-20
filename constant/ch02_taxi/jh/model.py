#! /usr/bin/env python
# Copyright 2023 O1 Software Network. MIT licensed.
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from constant.ch02_taxi.jh.etl import discard_outlier_rows
from constant.ch02_taxi.jh.features import COMPRESSED_DATASET, add_pickup_dow_hour
from constant.util.path import temp_dir


def _get_df() -> pd.DataFrame:
    in_file: Path = COMPRESSED_DATASET
    df = pd.read_parquet(in_file)
    df = discard_outlier_rows(df)
    df = add_pickup_dow_hour(df)
    return df


def linear_model() -> None:
    df = _get_df()
    model = LinearRegression()

    model.fit(np.array(df.distance).reshape(-1, 1), df.elapsed)

    m_per_s = 1 / model.coef_[0]
    print(f"{m_per_s:.1f} m/s")
    assert model.intercept_ > 0  # Sigh! Ideally it would go through the origin.
    print("model.score():", model.score(df[["distance"]], df.elapsed))
    sns.scatterplot(
        data=df,
        x="distance",
        y="elapsed",
        alpha=0.2,
    )
    # now plot the regression line
    xs = np.linspace(0, 40_000, 100)
    ys = model.predict(pd.DataFrame(xs))
    plt.plot(xs, ys, color="red")
    plt.title("Taxi Ride  Distance (m) vs. Elapsed time (s)")
    plt.savefig(temp_dir() / "constant/distance_vs_duration.png")
    plt.show()


def tree_model(num_rows: int = 10_000) -> None:
    df = _get_df()
    df = df[df.elapsed <= 3600]
    df = df[df.distance <= 30_000][: 2 * num_rows]
    model = XGBRegressor()
    y_train = df.elapsed[:num_rows]
    informative_cols = ["distance", "dow", "hour"]
    print("fitting...")

    model.fit(np.array(df[informative_cols][:num_rows]), y_train)

    p = pd.DataFrame({"distance": df.distance, "actual_elapsed": df.elapsed})
    sns.scatterplot(data=p, x="distance", y="actual_elapsed", alpha=0.3, color="red")

    assert 9_929 == len(df.distance)
    df = df[:num_rows]
    print("\n")
    print(pd.DataFrame({"elapsed": model.predict(df[informative_cols])}))
    p = pd.DataFrame(
        {
            "distance": df.distance,
            "elapsed": model.predict(df[informative_cols]),
        }
    )
    sns.scatterplot(data=p, x="distance", y="elapsed", alpha=0.8, color="purple")
    plt.show()


if __name__ == "__main__":
    tree_model()
    # linear_model()
