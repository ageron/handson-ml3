# Copyright 2023 O1 Software Network. MIT licensed.
import logging
from time import strftime


def logging_basic_config(level=logging.INFO):
    tz = strftime("%z")
    fmt = f"%(asctime)s.%(msecs)03d{tz} %(levelname)s %(relativeCreated)5d %(name)s  %(message)s"
    logging.basicConfig(level=level, datefmt="%Y-%m-%d %H:%M:%S", format=fmt)
