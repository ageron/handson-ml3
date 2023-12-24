#! /usr/bin/env streamlit run --server.runOnSave true
# Copyright 2023 O1 Software Network. MIT licensed.

import warnings
from pathlib import Path

import matplotlib.figure as mpfig
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from constant.ch02_taxi.jh.features import COMPRESSED_DATASET

warnings.filterwarnings(
    "ignore",
    message="is_categorical_dtype is deprecated and will be removed in a future version.",
)


@st.cache_data
def _tight_bbox(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        True
        & (-74.05 < df.dropoff_longitude)
        & (df.dropoff_longitude < -73.8)
        & (40.7 < df.dropoff_latitude)
        & (df.dropoff_latitude < 40.85)
    ]


def eda_time(df: pd.DataFrame, num_rows: int = 200_000) -> None:
    df = _tight_bbox(df)[:num_rows]
    show_dropoff_locations(df)


def show_dropoff_locations(df: pd.DataFrame) -> None:
    display_hour = st.slider("hour", 0, 23, 6)
    st.write(_plot_fig(df[df.hour == display_hour]))


@st.cache_data
def _plot_fig(df: pd.DataFrame) -> mpfig.Figure:
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="dropoff_longitude",
        y="dropoff_latitude",
        size=2,
        alpha=0.08,
        color="purple",
    )
    ax.set_xlim(df.dropoff_longitude.min(), df.dropoff_longitude.max())
    ax.set_ylim(df.dropoff_latitude.min(), df.dropoff_latitude.max())
    return fig


def main(in_file: Path = COMPRESSED_DATASET) -> None:
    eda_time(pd.read_parquet(in_file))


if __name__ == "__main__":
    main()
