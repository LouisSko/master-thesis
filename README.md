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
pipeline = ForecastingPipeline(model=Chronos, 
                               model_kwargs={"pretrained_model_name_or_path": "amazon/chronos-bolt-tiny", "device_map": "mps", "lead_times": lead_times, "freq": freq, "finetuning_type": "full"}, 
                               postprocessors=[PostprocessorMLE, PostprocessorQR],
                               output_dir=Path("./results/pipeline/chronos"))
```

### Train, Predict, Postprocess 

```python
# data needs to be provided in the autogluon TimeSeriesDataFrame format
data_train, data_val, data_test = pipeline.split_data(data=data, 
                                                      test_start_date=pd.Timestamp  ("01-01-2022"), 
                                                      train_window_size=None,
                                                      test_window_size=None,
                                                      val_window_size=pd.DateOffset(years=1))

pipeline.train(data_train)
pipeline.train_postprocessors(calibration_data=data_val)
pipeline.save()

predictions = pipeline.predict(data_test=data_test, data_previous_context=data_val)
predictions = pipeline.apply_postprocessing(predictions)
```

### Or backtest performance E2E
```python
results = pipeline.backtest(data=data,
                            test_start_date=pd.Timestamp("01-01-2022"),
                            rolling_window_eval=False,
                            train=True,
                            val_window_size=None,
                            train_window_size=None,
                            test_window_size=None,
                            calibration_based_on="train",
                            save_results=True)
```
