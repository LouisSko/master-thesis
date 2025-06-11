import numpy as np
import pandas as pd
from src.postprocessors.mle import PostprocessorMLE
from src.postprocessors.qr import PostprocessorQR
from src.postprocessors.eqc import PostprocessorEQC
from pathlib import Path

lead_times = np.arange(1, 192 + 1).tolist()
quantiles = np.round(np.arange(0.1, 1, 0.1), 1).tolist()
test_start_date = pd.Timestamp("2023-01-01")
postprocessors = [PostprocessorMLE, PostprocessorQR, PostprocessorEQC]
postprocessor_kwargs = [{"transformer": None, "n_jobs": 4}, {"transformer": None, "n_jobs": 4}, {"n_jobs": 4}]

output_dir_day_ahead_prices = Path("./results/day_ahead_prices/pipeline/")
output_dir_electricity_consumption = Path("./results/electricity_consumption/pipeline/")
output_dir_exchange_rates = Path("./results/exchange_rates/pipeline/")
