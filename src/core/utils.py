import json
import logging
import re

import numpy as np
import pandas as pd


def get_logger(log_level: str, logger_name: str = "default") -> logging.Logger:
    """Creates and configures a logger with the specified name and log level.

    Parameters
    ----------
    log_level : str
        Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
    logger_name : str
        Name of the logger

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}

    if log_level not in level_map:
        raise ValueError(f"Invalid log level: {log_level}")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level_map[log_level])
    formatter = logging.Formatter("%(asctime)s - %(levelname)s [%(name)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class CustomJSONEncoder(json.JSONEncoder):
    """Class for serializing timestamps."""

    def default(self, o):
        """Override the default method to serialize timestamps.

        Parameters:
        ----------
        o
            The object to serialize.

        Returns:

            str: The serialized object.
        """

        if isinstance(o, pd.Timestamp):
            if o.tzinfo is not None:
                # Serialize with timezone information
                return o.isoformat()
            else:
                # Serialize without timezone information
                return o.strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(o, pd.Timedelta):
            # Serialize Timedelta as a string in "Xd", "Xh", "Xm", etc.
            seconds = o.total_seconds()
            if seconds % 86400 == 0:  # 86400 seconds in a day
                return f"{int(seconds // 86400)}d"
            elif seconds % 3600 == 0:
                return f"{int(seconds // 3600)}h"
            elif seconds % 60 == 0:
                return f"{int(seconds // 60)}m"
            else:
                return f"{seconds}s"

        if isinstance(o, np.float32):
            return float(o)

        return super().default(o)


def timestamp_decoder(obj):
    """Convert strings back to pd.Timestamp or pd.Timedelta during JSON decoding.

    Handles nested structures like dictionaries, lists, and tuples.
    """

    # Regular expression to match Timedelta strings (e.g., "4h", "15min", "1d")
    timedelta_pattern = re.compile(r"^(\d+)([a-z]+)$")

    def process_value(value):
        """Process individual value to convert to Timestamp or Timedelta."""
        if isinstance(value, str):
            match = timedelta_pattern.match(value)
            if match:
                return pd.Timedelta(value)
            try:
                return pd.Timestamp(value, tz="utc")
            except ValueError:
                return value
        elif isinstance(value, dict):
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [process_value(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(process_value(item) for item in value)
        return value

    return process_value(obj)
