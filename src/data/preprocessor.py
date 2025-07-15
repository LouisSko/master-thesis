from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import logging
from autogluon.timeseries import TimeSeriesDataFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


def read_smard_data(
    file_paths: List[Path],
    selected_time_series: Optional[List[str]] = None,
    cols_to_drop: Optional[List[str]] = None,
    freq: Union[str, pd.DateOffset] = "1h",
) -> Tuple[TimeSeriesDataFrame, Dict[int, str]]:
    """
    Reads and processes SMARD data from multiple CSV files into a unified long-format TimeSeriesDataFrame.

    Note
    ----
    The timestamps column indicates the end of each hourly/quarterhourly interval.
    For example, "23:00 01-01-2025" represents the hour from 22:00 to 23:00 on January 1st, 2025.
    This means timestamps are backward-facing (the value belongs to the preceding hour)."

    Parameters
    ----------
    file_paths : List[Path]
        List of file paths pointing to SMARD CSV data files.
    selected_time_series : Optional[List[str]], default=None
        List of column names to retain. If None, all columns are used.
    cols_to_drop : Optional[List[str]], default=None
        List of column names to drop from the data.

    Returns
    -------
    Tuple[TimeSeriesDataFrame, Dict[int, str]]
        - A long-format TimeSeriesDataFrame with columns ['timestamp', 'item_id', 'target'].
        - A mapping from integer item_ids to original time series names.
    """

    freq = pd.tseries.frequencies.to_offset(freq).freqstr

    logging.info("Reading SMARD data...")

    # Step 1: Read and concatenate all CSVs
    dfs = []
    for file_path in file_paths:
        logging.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path, sep=";", decimal=".", thousands=",", na_values=["-"], parse_dates=["Start date", "End date"], date_format="%b %d, %Y %I:%M %p")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Total rows after concat: {len(df)}")

    # Step 2: Drop unnecessary columns
    df.drop(columns=["Start date"], inplace=True, errors="ignore")

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        logging.info(f"Dropped columns: {cols_to_drop}")

    if selected_time_series:
        logging.info(f"Filtering columns: {selected_time_series}")
        columns_to_keep = ["End date"] + selected_time_series
        df = df[columns_to_keep]
        logging.info(f"Columns retained: {df.columns.tolist()}")

    # Step 3: Format & reshape
    df.rename(columns={"End date": "timestamp"}, inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", keep="first", inplace=True)

    df = df.melt(id_vars="timestamp", var_name="item_id", value_name="target")
    logging.info(f"Reshaped DataFrame: {df.shape[0]} rows")

    # Step 4: Factorize item_id
    df["item_id"], mapping = pd.factorize(df["item_id"])
    mapping = {i: name for i, name in enumerate(mapping)}

    logging.info(f"Mapped {len(mapping)} unique time series.")

    ts_df = TimeSeriesDataFrame(df)

    # check frequency of the df
    if ts_df.freq != freq:
        logging.info("Frequency of data '%s' does not match defined frequency '%s'.", ts_df.freq, freq)
        ts_df = ts_df.convert_frequency(freq)
        logging.info("Data resampled to %s", ts_df.freq)

    # Step 5: Log memory usage
    mem_mb = ts_df.memory_usage(deep=True).sum() / 1024**2
    logging.info(f"Final DataFrame memory usage: {mem_mb:.2f} MB")

    return ts_df, mapping


def read_exchange_rates_data(files_dir: Union[str, Path] = Path("data/exchange_rates/")) -> Tuple[TimeSeriesDataFrame, Dict[int, str]]:
    """
    Reads and processes exchange rate data from FRED into a unified long-format TimeSeriesDataFrame.

    Parameters
    ----------
    files_dir : Union[str, Path]
        Path to the directory containing exchange rate CSV files.

    Returns
    -------
    Tuple[TimeSeriesDataFrame, Dict[int, str]]
        - A long-format TimeSeriesDataFrame with columns ['timestamp', 'item_id', 'target'].
        - A mapping from integer item_ids to currency pairs.
    """

    if isinstance(files_dir, str):
        files_dir = Path(files_dir)

    logging.info("Reading exchange rates data...")

    freq = pd.tseries.offsets.BusinessDay(1)

    # Step 1: Read and preprocess each file
    ex_rates = {}
    for file_path in sorted(files_dir.iterdir()):
        if not file_path.name.endswith(".csv"):
            continue

        logging.info(f"Reading file: {file_path}")
        df = pd.read_csv(file_path)
        df["timestamp"] = pd.to_datetime(df["observation_date"])
        df.drop(columns=["observation_date"], inplace=True)
        df.set_index("timestamp", inplace=True)

        symbol = file_path.stem.removeprefix("DEX")

        if symbol.endswith("US"):
            df = 1 / df
            symbol = symbol[2:] + symbol[:2]

        df.columns = [symbol]
        ex_rates[symbol] = df

    # Step 2: Concatenate series
    df = pd.concat(ex_rates.values(), axis=1).reset_index()
    logging.info(f"Total rows after concatenation: {len(df)}")

    # Step 3: Reshape to long format
    df = df.melt(id_vars="timestamp", var_name="item_id", value_name="target")
    logging.info(f"Reshaped DataFrame: {df.shape[0]} rows")

    # Step 4: Factorize item_id
    df["item_id"], mapping = pd.factorize(df["item_id"])
    mapping = {i: name for i, name in enumerate(mapping)}

    logging.info(f"Mapped {len(mapping)} unique exchange rate series.")


    ts_df = TimeSeriesDataFrame(df)

    # check frequency of the df
    if ts_df.freq != freq:
        logging.info("Frequency of data '%s' does not match defined frequency '%s'.", ts_df.freq, freq)
        ts_df = ts_df.convert_frequency(freq)
        logging.info("Data resampled to %s", ts_df.freq)

    # Step 5: Log memory usage
    mem_mb = ts_df.memory_usage(deep=True).sum() / 1024**2
    logging.info(f"Final DataFrame memory usage: {mem_mb:.2f} MB")

    return ts_df, mapping
