#! /usr/bin/env python
# Copyright 2023 O1 Software Network. MIT licensed.
"""Feature augmentation."""

import warnings
from collections import Counter
from time import time
from typing import Generator

import geopandas as gpd
import numpy as np
import pandas as pd
from cartopy.geodesic import Geodesic

from constant.util.path import temp_dir
from constant.util.timing import timed

warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0")


COMPRESSED_DATASET = temp_dir() / "constant/trip.parquet"

wgs84: Geodesic = Geodesic()

# This is very near both the median pickup and median dropoff point,
# Bryant Park behind the lions at the NYPL.
# Distance from pickup to Grand Central can help with removing outliers.
grand_central_nyc = 40.752, -73.978


def add_pickup_dow_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week and hour-of-day features."""
    df.loc[:, ("dow",)] = df["pickup_datetime"].copy().dt.dayofweek  # Monday=0
    df.loc[:, ("hour",)] = df["pickup_datetime"].copy().dt.hour
    return df


def _direction(row: pd.Series) -> float:
    """Return azimuth of a single trip."""
    begin = (row["pickup_latitude"], row["pickup_longitude"])
    end = (row["dropoff_latitude"], row["dropoff_longitude"])
    bearing, _ = azimuth(begin, end)
    return round(bearing)


@timed
def add_direction(df: pd.DataFrame) -> pd.DataFrame:
    df["direction"] = df.apply(lambda row: _direction(row), axis=1)
    return df


def azimuth(
    begin_lat_lng: tuple[float, float], end_lat_lng: tuple[float, float]
) -> tuple[float, float]:
    """Return the azimuth angle when heading from BEGIN to END, and also distance."""
    begin_lng_lat = np.array(list(reversed(begin_lat_lng)))
    end_lng_lat = np.array(list(reversed(end_lat_lng)))
    meters, degrees, _ = map(int, wgs84.inverse(begin_lng_lat, end_lng_lat)[0])
    return degrees, meters


# wgs84:
_tlc_zone_shapes = gpd.read_file(temp_dir() / "constant/taxi_zones.zip").to_crs(
    "EPSG:2263"
)


@timed
def add_tlc_zone(df: pd.DataFrame) -> pd.DataFrame:
    wgs_84 = "EPSG:4326"
    df["pickup_pt"] = gpd.points_from_xy(df.pickup_longitude, df.pickup_latitude)
    gdf = gpd.GeoDataFrame(df, geometry="pickup_pt", crs=wgs_84).to_crs(epsg=2263)
    assert "EPSG:2263" == gdf.crs, gdf.crs  # https://epsg.io/2263 NAD83 / New York L.I.
    assert "EPSG:2263" == _tlc_zone_shapes.crs, _tlc_zone_shapes.crs
    assert gdf.crs.is_projected
    assert _tlc_zone_shapes.crs.is_projected

    t0 = time()
    joined_pu = gdf.sjoin_nearest(
        _tlc_zone_shapes, how="left", distance_col="join_distance"
    )
    elapsed = time() - t0
    elapsed > 1 and print(f"joined_pu: {elapsed:.3f} seconds")
    assert joined_pu.join_distance.quantile(0.9992) == 0
    joined_pu = joined_pu[joined_pu.join_distance == 0]
    df["pickup_borough"] = joined_pu.borough
    df["pickup_zone"] = joined_pu.zone

    # and now attend to the destination:
    df["dropoff_pt"] = gpd.points_from_xy(df.dropoff_longitude, df.dropoff_latitude)
    gdf = gpd.GeoDataFrame(df, geometry="dropoff_pt", crs=wgs_84).to_crs(epsg=2263)
    joined_dr = gdf.sjoin_nearest(
        _tlc_zone_shapes, how="left", distance_col="join_distance"
    )
    if len(df) > 1:  # Ignore the Logan test.
        assert joined_dr.join_distance.quantile(0.997) == 0

    joined_dr = joined_dr[joined_dr.join_distance == 0]
    df["dropoff_borough"] = joined_dr.borough
    df["dropoff_zone"] = joined_dr.zone
    return df


def _get_borough_pairs(df: pd.DataFrame) -> Generator[tuple[str, str], None, None]:
    for _, row in df.iterrows():
        yield (row.pickup_borough, row.dropoff_borough)


def get_borough_matrix(df: pd.DataFrame) -> Counter:
    return Counter(sorted(_get_borough_pairs(df)))


def _get_zone_pairs(df: pd.DataFrame) -> Generator[tuple[str, str], None, None]:
    for _, row in df.iterrows():
        yield (row.pickup_zone, row.dropoff_zone)


def get_zone_matrix(df: pd.DataFrame) -> Counter:
    return Counter(sorted(_get_zone_pairs(df)))
