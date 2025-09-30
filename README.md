# Postprocessing and Fine-Tuning Methods for Improving Long-Term Probabilistic Forecasts of Time Series Foundation Models  
**(Master's Thesis Project)**

This repository provides a flexible framework for **univariate time series quantile forecasting**.  
It focuses on improving the performance of large time series foundation models (like **Chronos Bolt**) through **fine-tuning** and **postprocessing** techniques.

---

## 📦 Relevant Repositories

- [chronos-forecasting](https://github.com/LouisSko/chronos-forecasting/tree/feature/sampling) – Chronos usage and fine-tuning (branch: `feature/sampling`)
- [tirex-microservice](https://github.com/LouisSko/tirex-microservice) – Optional TiRex microservice integration
- [gift-eval](https://github.com/LouisSko/gift-eval) – Benchmarking and evaluation framework for GIFT-Eval (branch: `chronos`)

---

## 🚀 Features

- Fine-tuning of **Chronos and Chronos Bolt** for any forecast horizon  
- Integration of external models (e.g., **AutoGluon**)
- Easy to use post-hoc calibration techniques
- End-to-end (E2E) testing of forecasting pipelines  
- Comprehensive evaluation metrics:
  - Continuous Ranked Probability Score (**CRPS**)
  - Quantile scores
  - Reliability diagrams
  - Probability Integral Transform (**PIT**) histograms
  - And more

---

## ⚙️ Setup

### 1) Clone the Repository

```bash
git clone git@github.com:LouisSko/mt-probabilistic-forecasting-framework.git
cd mt-probabilistic-forecasting-framework
```

### 2) Create and Activate a Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3) Upgrade pip (recommended)

```bash
pip install --upgrade pip
```

### 4) Install Requirements

```bash
pip install -r requirements.txt
```

---

## 📈 Chronos Bolt Fine-Tuning & Evaluation

### 5) Clone the Chronos Forecasting Repository (branch: `feature/sampling`)

```bash
git clone --branch feature/sampling git@github.com:LouisSko/chronos-forecasting.git
```

### 6) Install Chronos in Editable Mode with Training Dependencies

```bash
cd chronos-forecasting
pip install --editable ".[training]"
```

### Getting Started
use the minium_working_example.ipynb to get started.


### 7) Run Example Evaluation

```bash
cd /path/to/mt-probabilistic-forecasting-framework
export PYTHONPATH=$(pwd)  # Set repo root as Python path

python src/scripts/evaluate.py --dataset exchange_rates
python src/scripts/evaluate.py --dataset electricity_consumption
python src/scripts/evaluate.py --dataset day_ahead_prices
```

All individual predictions and models are saved. If you want to run this make sure, to have 100 Gb of available storage. 

---

## 🔌 Optional: TiRex Integration

If you want to integrate **TiRex** with the forecasting framework, follow the instructions here:  
👉 [tirex-microservice](https://github.com/LouisSko/tirex-microservice)

---

## 📊 Results

- Results are stored in the `/results` directory.
- During evaluation, large `predictions.joblib` files were created (~100 GB).  
  Only high-level `.json` result summaries are uploaded to GitHub.
- Many evaluation plots depend on the full `joblib` prediction files.
- Model-specific settings (e.g., `max_calibration_samples`) can be configured in `src/scripts/eval_constants.py`.

---

# 📉 GIFT-Eval Benchmarking

Run models on GIFT-Eval benchmark

### 1) Clone the `gift-eval` Repository (branch: `chronos`)

```bash
cd /path/to/mt-probabilistic-forecasting-framework
git clone --branch chronos git@github.com:LouisSko/gift-eval.git
```

### 2) Install Required Dependencies inside the existing environmen

```bash
cd gift-eval
pip install -e .
```

### 3) Download the GIFT-Eval Dataset and Create a `.env` File

```bash
# Set the path where you want to store the dataset
PATH_TO_SAVE="/absolute/path/to/save"

# Download dataset
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir "$PATH_TO_SAVE"

# Create a .env file for environment variable loading
echo "GIFT_EVAL=$PATH_TO_SAVE" > .env

# If your environment doesn't auto-load .env, export it manually:
# export $(cat .env | xargs)
```

### 4) Run the Benchmark

```bash
cd /path/to/mt-probabilistic-foreacsting-framework/gift-eval
python notebooks/chronos-custom_ft_ensemble.py
```

---

## 📁 Project Structure

```
mt-probabilistic-forecasting-framework/
├─ src/                     # Core framework code
├─ results/                 # Evaluation results on core datasets
├─ GIFT-Eval-results/       # Benchmark results
└─ notebooks/               # Analysis notebooks
└─ data/                    # Contains data for the core datasets
└─ archive/                 # Archive
```


