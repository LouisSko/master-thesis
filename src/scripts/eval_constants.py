import numpy as np
import pandas as pd
from src.postprocessors.mle import PostprocessorMLE
from src.postprocessors.qr import PostprocessorQR
from src.postprocessors.eqc import PostprocessorEQC
from pathlib import Path
from src.data.preprocessor import read_smard_data, read_exchange_rates_data

lead_times = np.arange(1, 192 + 1).tolist()
quantiles = np.round(np.arange(0.1, 1, 0.1), 1).tolist()
test_start_date = pd.Timestamp("2023-01-01")
postprocessors = [PostprocessorMLE, PostprocessorQR, PostprocessorEQC]
postprocessor_kwargs = [{"transformer": None, "n_jobs": 4}, {"transformer": None, "n_jobs": 4}, {"n_jobs": 4}]

output_dir_day_ahead_prices = Path("./results/day_ahead_prices/pipeline/")
output_dir_electricity_consumption = Path("./results/electricity_consumption/pipeline/")
output_dir_exchange_rates = Path("./results/exchange_rates/pipeline/")


electricity_consumption_config = {
    "freq": "15 min",
    "val_window_size": pd.DateOffset(years=1),  # potentially months=3 to be consistent
    "test_window_step": 96,  # 4 * 24 -> sliding window evaluation every 96 timesteps, corresponds to one day
    "output_dir": Path("./results/electricity_consumption/pipeline/"),
    "seasonal_period": 672,  # 4 * 24 * 7 -> seasonal period same minute last week
    "data": read_smard_data(
        file_paths=[
            "data/electricity_consumption/Actual_consumption_201501010000_202001010000_Quarterhour.csv",
            "data/electricity_consumption/Actual_consumption_202001010000_202506120000_Quarterhour.csv",
        ],
        selected_time_series=["grid load [MWh] Original resolutions", "Residual load [MWh] Original resolutions"],
        freq="15 min",
    )[0],
}

day_ahead_prices_config = {
    "freq": "1 h",
    "val_window_size": pd.DateOffset(years=1),
    "test_window_step": 24,  # sliding window evaluation every 24 timesteps, corresponds to one day
    "output_dir": Path("./results/electricity_consumption/pipeline/"),
    "seasonal_period": 168,  # 24 * 7 -> same day of the previous week
    "data": read_smard_data(
        file_paths=["data/day_ahead_prices/Day-ahead_prices_201501010000_202001010000_Hour.csv", "data/day_ahead_prices/Day-ahead_prices_202001010000_202506120000_Hour.csv"],
        selected_time_series=[
            "Belgium [€/MWh] Original resolutions",
            "Denmark 1 [€/MWh] Original resolutions",
            "Denmark 2 [€/MWh] Original resolutions",
            "France [€/MWh] Original resolutions",
            "Netherlands [€/MWh] Original resolutions",
            "Norway 2 [€/MWh] Original resolutions",
            "Sweden 4 [€/MWh] Original resolutions",
            "Switzerland [€/MWh] Original resolutions",
            "Czech Republic [€/MWh] Original resolutions",
            "Slovenia [€/MWh] Original resolutions",
            "Hungary [€/MWh] Original resolutions",
        ],
        freq="1 h",
    )[0],
}

exchange_rate_config = {
    "freq": "B",  # business day
    "val_window_size": pd.DateOffset(years=5),
    "test_window_step": 1,  # 4 * 24 -> sliding window evaluation every timestep, corresponds to one day
    "output_dir": Path("./results/electricity_consumption/pipeline/"),
    "seasonal_period": 1,  # no reasonable setting
    "data": read_exchange_rates_data(files_dir="data/exchange_rates/")[0],
}
