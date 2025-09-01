import numpy as np
import pandas as pd
from src.postprocessors.mle import PostprocessorMLE
from src.postprocessors.qr import PostprocessorQR, PostprocessorFastQR
from src.postprocessors.eqc import PostprocessorEQC
from pathlib import Path
from src.data.preprocessor import read_smard_data, read_exchange_rates_data

lead_times = np.arange(1, (64*10) + 1).tolist()
quantiles = np.round(np.arange(0.1, 1, 0.1), 1).tolist()
test_start_date = pd.Timestamp("2023-01-01")
postprocessors = [PostprocessorEQC, PostprocessorFastQR, PostprocessorMLE]
postprocessor_kwargs = [
    {"n_jobs": 2, "name": "PP_Offset"},
    {"n_jobs": 2, "name": "PP_QuantReg"},
    {"n_jobs": 2, "name": "PP_Gauss"},
    ]

# whether to evaluate using auto calibration or not
auto_calibration = False # or False
auto_determine_val_set = False # or False

# Path to this script
script_dir = Path(__file__).resolve().parent

# Project root (adjust depending on where scripts/ sits)
project_root = script_dir.parent.parent


def get_electricity_consumption_config():
    return {
        "freq": "15 min",
        "val_window_size": pd.DateOffset(years=1),  # potentially months=3 to be consistent
        "test_window_step": 96,  # 4 * 24 -> sliding window evaluation every 96 timesteps, corresponds to one day
        "calibration_window_step": 96,
        "output_dir": project_root / "results/electricity_consumption/pipeline/",
        "seasonal_period": 672,  # 4 * 24 * 7 -> seasonal period same minute last week
        "data": read_smard_data(
            file_paths=[
                project_root / "data/electricity_consumption/Actual_consumption_201501010000_202001010000_Quarterhour.csv",
                project_root / "data/electricity_consumption/Actual_consumption_202001010000_202509010000_Quarterhour.csv",
            ],
            selected_time_series=["grid load [MWh] Original resolutions", "Residual load [MWh] Original resolutions"],
            freq="15 min",
        )[0],
    }


def get_day_ahead_prices_config():
    return {
        "freq": "1 h",
        "val_window_size": pd.DateOffset(years=1),
        "test_window_step": 24,  # sliding window evaluation every 24 timesteps, corresponds to one day
        "calibration_window_step": 24,
        "output_dir": project_root / "results/day_ahead_prices/pipeline/",
        "seasonal_period": 168,  # 24 * 7 -> same day of the previous week
        "data": read_smard_data(
            file_paths=[
                project_root / "data/day_ahead_prices/Day-ahead_prices_201501010000_202001010000_Hour.csv",
                project_root / "data/day_ahead_prices/Day-ahead_prices_202001010000_202509010000_Hour.csv",
            ],
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


def get_exchange_rate_config():
    return {
        "freq": "B",  # business day
        "val_window_size": pd.DateOffset(years=5),
        "test_window_step": 1,  # -> sliding window evaluation every timestep, corresponds to one day
        "calibration_window_step": 1,
        "output_dir": project_root / "results/exchange_rates/pipeline/",
        "seasonal_period": 1,  # no reasonable setting
        "data": read_exchange_rates_data(files_dir=project_root / "data/exchange_rates/")[0],
    }
