#! /usr/bin/env python
# Copyright 2023 O1 Software Network. MIT licensed.

from pathlib import Path

import pandas as pd
from beartype import beartype

from constant.ch02_taxi.jh.features import COMPRESSED_DATASET


@beartype
def train_duration_model(df: pd.DataFrame) -> None:
    """Train a model to predict trip duration."""
    print(df)


def main(in_file: Path = COMPRESSED_DATASET) -> None:
    train_duration_model(pd.read_parquet(in_file))


if __name__ == "__main__":
    main()
