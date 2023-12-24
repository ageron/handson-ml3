#! /usr/bin/env _TYPER_STANDARD_TRACEBACK=1 SQLALCHEMY_WARN_20=1 python
# Copyright 2023 O1 Software Network. MIT licensed.


import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import sqlalchemy as sa
import typer
from geopy.distance import distance
from ruamel.yaml import YAML

from constant.ch02_taxi.jh.features import (
    COMPRESSED_DATASET,
    add_direction,
    add_pickup_dow_hour,
    add_tlc_zone,
    grand_central_nyc,
)
from constant.util.path import constant, temp_dir
from constant.util.timing import timed

CONFIG_FILE = constant() / "ch02_taxi/jh/taxi.yml"


class Etl:
    """Extract, transform, and load Kaggle taxi data into a SQLite trip table."""

    def __init__(self, db_file: Path, decorator: Callable[[Any], Any] = timed) -> None:
        self.folder = db_file.parent.resolve()
        self.engine = sa.create_engine(f"sqlite:///{db_file}", echo=False)

        for method_name in dir(self) + dir(Etl):
            attr = getattr(self, method_name)
            if re.match(r"<class '(function|method)'>", str(type(attr))):
                wrapped = decorator(attr)
                setattr(self, method_name, wrapped)

    def create_table(self, in_csv: Path) -> None:
        date_cols = ["pickup_datetime", "dropoff_datetime"]
        df = pd.read_csv(in_csv, parse_dates=date_cols)
        df = self._discard_unhelpful_columns(df)
        df["distance"] = 0.0
        self._find_distance(df)  # This costs 6 minutes for 1.46 M rows (4200 row/sec)
        df = discard_outlier_rows(df)
        df = add_direction(df)
        df = add_pickup_dow_hour(df)
        df.to_parquet(COMPRESSED_DATASET, index=False)  #  JIC -- will overwrite later
        df = add_tlc_zone(df)

        one_second = "1s"  # trim meaningless milliseconds from observations
        for col in date_cols:
            df[col] = df[col].dt.round(one_second)

        with self.engine.begin() as sess:
            sess.execute(sa.text("DROP TABLE  IF EXISTS  trip"))
            sess.execute(self.ddl)
        df = df.drop(columns=["pickup_pt", "dropoff_pt"])
        print(df)
        df.to_sql("trip", self.engine, if_exists="append", index=False)

        df.to_parquet(COMPRESSED_DATASET, index=False)

        # self._write_yaml_bbox(df)

    def _discard_unhelpful_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        delayed = _round(df[df.store_and_fwd_flag == "Y"])
        delayed.to_csv(self.folder / "outlier_delayed.csv", index=False)

        df = df.drop(columns=["vendor_id"])  # TPEP: Creative Mobile or Verifone
        df = df.drop(columns=["store_and_fwd_flag"])  # only Y when out-of-range
        return df

    ddl = sa.text(
        """
    CREATE TABLE trip (
        id                 TEXT  PRIMARY KEY,
        pickup_datetime    DATETIME,
        dropoff_datetime   DATETIME,
        passenger_count    INTEGER,
        pickup_longitude   FLOAT,
        pickup_latitude    FLOAT,
        dropoff_longitude  FLOAT,
        dropoff_latitude   FLOAT,
        trip_duration      INTEGER,
        distance           FLOAT,
        direction          BIGINT,
        dow                INTEGER,
        hour               INTEGER,
        elapsed            FLOAT,
        pickup_borough     TEXT,
        pickup_zone        TEXT,
        dropoff_borough    TEXT,
        dropoff_zone       TEXT
    )
    """
    )

    @classmethod
    def _distance(cls, row: pd.Series) -> float:
        from_ = row.pickup_latitude, row.pickup_longitude
        to = row.dropoff_latitude, row.dropoff_longitude
        meters: float = distance(from_, to).m
        return round(meters, 2)

    @classmethod
    def _find_distance(cls, df: pd.DataFrame) -> pd.DataFrame:
        df["distance"] = df.apply(lambda row: cls._distance(row), axis=1)
        return df

    # Discard distant locations, e.g. the Verifone / Automation Anywhere
    # meter service center in San Jose, CA, locations in the North Atlantic.
    # SERVICE_RADIUS = 60_000  # 37 miles
    SERVICE_RADIUS = 150_000  # 93 miles

    def _write_yaml_bbox(self, df: pd.DataFrame, out_file: Path = CONFIG_FILE) -> None:
        ul, lr = self._get_bbox(df)

        d = dict(
            db_file=self.engine.url.database,
            grand_central_nyc=grand_central_nyc,
            service_radius=self.SERVICE_RADIUS,
            trip_count=len(df),
            ul=ul,
            lr=lr,
        )
        yaml = YAML()
        yaml.dump(d, Path(out_file))

    @staticmethod
    def _get_bbox(
        df: pd.DataFrame, precision: int = 3  # decimal places of precision
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        dec = precision  # need a short identifier, to prevent line wrap
        n_lat = round(max(df.pickup_latitude.max(), df.dropoff_latitude.max()), dec)
        s_lat = round(min(df.pickup_latitude.min(), df.dropoff_latitude.min()), dec)
        w_lng = round(min(df.pickup_longitude.min(), df.dropoff_longitude.min()), dec)
        e_lng = round(max(df.pickup_longitude.max(), df.dropoff_longitude.max()), dec)
        up, lf = map(float, (n_lat, w_lng))
        lw, rt = map(float, (s_lat, e_lng))
        ul = up, lf
        lr = lw, rt
        return ul, lr


def _round(df: pd.DataFrame, precision: int = 4) -> pd.DataFrame:
    cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    t = pd.DataFrame()
    for col in cols:
        t[col] = df[col].round(precision)
    df = df.drop(columns=cols)
    return pd.concat([df, t], axis=1)


def _read_yaml_bbox(
    in_file: Path = CONFIG_FILE,
) -> tuple[tuple[float, float], tuple[float, float]]:
    d = YAML().load(Path(in_file))
    return d["ul"], d["lr"]


def discard_outlier_rows(df: pd.DataFrame) -> pd.DataFrame:
    FOUR_HOURS = 4 * 60 * 60  # Somewhat commonly we see 86400 second "trips".
    long_trips = _round(df[df.trip_duration >= FOUR_HOURS])
    long_trips.to_csv(temp_dir() / "constant/outlier_long_trips.csv", index=False)
    # Discard trips where cabbie forgot to turn off the meter.
    df = df[df.trip_duration < FOUR_HOURS]

    ul, lr = _read_yaml_bbox()
    df = df[
        True
        & (lr[0] < df.pickup_latitude)
        & (df.pickup_latitude < ul[0])
        & (ul[1] < df.pickup_longitude)
        & (df.pickup_longitude < lr[1])
        & (lr[0] < df.dropoff_latitude)
        & (df.dropoff_latitude < ul[0])
        & (ul[1] < df.dropoff_longitude)
        & (df.dropoff_longitude < lr[1])
    ]

    assert df.passenger_count.max() <= 9
    return df


def main(in_csv: Path) -> None:
    Etl(in_csv.parent / "taxi.db").create_table(in_csv)


if __name__ == "__main__":
    typer.run(main)
