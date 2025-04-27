# Quantile Forecasting with Pretrained Time Series Transformers

This project provides a flexible pipeline for univariate time series **quantile forecasting**.

It supports:
- Fine-tuning of the **Chronos Bolt** model (including full fine-tuning, last-layer fine-tuning, and LoRA).
- Integration of other time series models and custom postprocessors.
- End-to-end (E2E) testing of forecasting pipelines.
- Comprehensive evaluation with metrics such as **CRPS scores**, **quantile scores**, **reliability diagrams**, **PIT histograms**, and more.


## Setup

### 1. Create Virtual Environment
```bash
python3.12 -m venv .venv
```

### 2. Activate Environment
```bash
source .venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```


## Basic Usage

### Create a pipeline consisting of models and postprocessors
```python
from autogluon.timeseries import TimeSeriesDataFrame
from src.data.preprocessor import read_smard_data
from autogluon.timeseries import TimeSeriesDataFrame

df, mapping = read_smard_data(file_paths=["data/Realisierter_Stromverbrauch_201501010000_202101010000_Stunde.csv",
                                          "data/Realisierter_Stromverbrauch_202101010000_202504240000_Stunde.csv"],
                              selected_time_series=["Netzlast [MWh] Berechnete Auflösungen", 
                                                    "Residuallast [MWh] Berechnete Auflösungen"])


data = TimeSeriesDataFrame(df)

from src.predictors.chronos import Chronos
from src.postprocessors.mle import PostprocessorMLE
from src.postprocessors.qr import PostprocessorQR
from src.pipeline.pipeline import ForecastingPipeline

pipeline = ForecastingPipeline(model=Chronos, 
                               model_kwargs={"pretrained_model_name_or_path": "amazon/chronos-bolt-tiny", "device_map": "mps", "lead_times": lead_times, "freq": freq, "finetuning_type": "full"}, 
                               postprocessors=[PostprocessorMLE, PostprocessorQR],
                               output_dir=Path("./results/pipeline/chronos"))
```

### Train, Predict, Postprocess 

```python
data_train, data_val, data_test = pipeline.split_data(data=data, 
                                                      test_start_date=pd.Timestamp  ("01-01-2022"), 
                                                      train_window_size=None,
                                                      test_window_size=None,
                                                      val_window_size=pd.DateOffset(years=1))

pipeline.train(data_train)
pipeline.train_postprocessors(calibration_data=data_val)

predictions = pipeline.predict(data_test=data_test, data_previous_context=data_val)
predictions = pipeline.apply_postprocessing(predictions)

# save and reload pipeline
pipeline.save()
pipeline = ForecastingPipeline.from_pretrained(path="./results/pipeline/nearest_neighbour")
```

### Evaluation of Predictions on Test Dataset

Predictions are stored as `PredictionLeadTimes` instances, allowing you to easily compute various evaluation metrics:

- `get_crps()`
- `get_quantile_scores()`
- `get_empirical_coverage_rates()`
- `get_pit_histogram()`
- `get_reliability_diagram()`

```python
# The raw, non-postprocessed predictions
predictions["Chronos"].get_crps(mean_lead_times=True, mean_time=True)

# Postprocessed predictions
predictions["PostprocessorMLE"].get_crps(mean_lead_times=True, mean_time=True)
predictions["PostprocessorQR"].get_crps(mean_lead_times=True, mean_time=True)
```

### Or Backtest Performance End-to-End
```python
predictions = pipeline.backtest(data=data,
                                test_start_date=pd.Timestamp("01-01-2022"),
                                rolling_window_eval=False,
                                train=True,
                                val_window_size=None,
                                train_window_size=None,
                                test_window_size=None,
                                calibration_based_on="train",
                                save_results=True)                              
```