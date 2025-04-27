from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


def read_smard_data(file_paths: List[Path], selected_time_series: Optional[List[str]] = None, cols_to_drop: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Reads and processes SMARD data from multiple CSV files into a unified long-format DataFrame.

    Data downloaded from: https://www.smard.de/home/downloadcenter/download-marktdaten/
    Information regarding switch to summer time: https://www.smard.de/home/die-zeitumstellung-auf-smard-216438

    Parameters
    ----------
    file_paths : List[Path]
        A list of file paths pointing to SMARD CSV data files.
    selected_time_series : Optional[List[str]], default=None
        A list of column names (time series) to retain. If None, all columns are used.
    cols_to_drop : Optional[List[str]], defaults=None
        A list of column names which should be dropped.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, str]]
        A tuple containing:
        - A long-format DataFrame with columns ['timestamp', 'item_id', 'target'] and integer-encoded item IDs.
        - A dictionary mapping integer item IDs to their original time series names.
    """

    logging.info("Starting to read SMARD data files...")

    # Step 1: Read and concatenate all CSV files
    dfs = []
    for file_path in file_paths:
        logging.info(f"Reading file: {file_path}")
        df_temp = pd.read_csv(file_path, sep=";", decimal=",", thousands=".", parse_dates=["Datum von", "Datum bis"], na_values=["-"], dayfirst=True)
        dfs.append(df_temp)

    df = pd.concat(dfs)
    logging.info(f"Total number of rows after concatenation: {len(df)}")

    # Step 2: Keep only selected time series
    df = df.drop(columns="Datum bis")

    if cols_to_drop:
        df.drop(cols_to_drop)
        logging.info(f"Dropped the following columns: {cols_to_drop}")

    if selected_time_series:
        logging.info(f"Selected time series: {selected_time_series}")
        selected_time_series.append("Datum von")
        df = df[selected_time_series].copy()
        logging.info(f"Columns retained: {df.columns.tolist()}")
    else:
        logging.info("No timeseries selected. Keeping all columns.")

    # Step 3: Preprocess and reshape
    df = df.rename(columns={"Datum von": "timestamp"})
    df = df.sort_values(by="timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    logging.info(f"Number of rows after removing duplicates: {len(df)}")

    df = df.melt(id_vars="timestamp", var_name="item_id", value_name="target")
    logging.info(f"Dataframe got melted. Number of rows: {len(df)}")

    # Step 4: Encode item_id
    df["item_id"], mapping = pd.factorize(df["item_id"])
    mapping = {i: m for i, m in zip(range(0, len(mapping)), mapping)}
    logging.info(f"Encoded {len(mapping)} unique time series.")

    logging.info(f"Mapped the time series to unique ids: {mapping}")
    # logging.info(("\n".join(f"{str(k)}: {v}" for k, v in mapping.items())))

    # Step 5: Log final memory usage
    size_gb = df.memory_usage(deep=True).sum() / (1024**2)
    logging.info(f"Final DataFrame size: {size_gb:.2f} MB")

    logging.info("Finished reading and preprocessing SMARD data.")

    return df, mapping
