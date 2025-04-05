import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy.optimize import minimize
from timeseries import PredictionLeadTimes, PredictionLeadTime, TabularDataFrame
import torch
from tqdm import tqdm
from typing import Tuple


def neg_log_likelihood(params: list, M: np.ndarray, IQR: np.ndarray, y: np.ndarray):
    """Computes the negative log-likelihood for a normal distribution, parameterized
    by median (M) and interquartile range (IQR), used for maximum likelihood estimation.

    Parameters
    ----------
    params : list
        List of parameters [a, b, c, d] where:
        - mu = a + b * M
        - sigma = c + d * IQR
    M : np.ndarray
        Median predictions.
    IQR : np.ndarray
        Interquartile ranges of predictions (e.g., 0.9 quantile - 0.1 quantile).
    y : np.ndarray
        Observed target values.

    Returns
    -------
    float
        Negative log-likelihood value.
    """

    a, b, c, d = params
    mu = a + b * M
    sigma = c + d * IQR
    nll = -np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    return nll


def estimate_init_params(m: np.ndarray, iqr: np.ndarray, y_mu: np.ndarray, y_sigma: np.ndarray) -> Tuple[float, float, float, float]:
    """Estimates initial parameters [a, b, c, d] using linear regression for mean and std.

    Parameters
    ----------
    m : np.ndarray
        Median values.
    iqr : np.ndarray
        Interquartile ranges.
    y_mu : np.ndarray
        Observed means (log target).
    y_sigma : np.ndarray
        Observed standard deviations of log target.

    Returns
    -------
    Tuple[float, float, float, float]
        Initial parameter estimates [a, b, c, d] for mu and sigma formulas.
    """

    # mean = a + b * Median
    x_mu = sm.add_constant(m)
    model_mu = sm.OLS(y_mu, x_mu).fit()
    a_init, b_init = model_mu.params

    # std = c + d * IQR
    x_sigma = sm.add_constant(iqr)
    model_sigma = sm.OLS(y_sigma, x_sigma).fit()
    c_init, d_init = model_sigma.params

    return a_init, b_init, c_init, d_init


def postprocess_mle(predictions_train: PredictionLeadTimes, predictions_test: PredictionLeadTimes) -> PredictionLeadTimes:
    """Applies Maximum Likelihood Estimation (MLE) to postprocess quantile predictions.
    Fits a normal distribution in log space using median and IQR-based modeling.

    Parameters
    ----------
    predictions_train : PredictionLeadTimes
        Training predictions used to estimate parameters for each item time series.
    predictions_test : PredictionLeadTimes
        Test predictions for which the distribution is fitted and quantiles are predicted.

    Returns
    -------
    PredictionLeadTimes
        Updated predictions for the test set with quantiles generated via MLE.
    """

    lead_time = list(predictions_train.results.keys())
    postprocessing_results = {}

    ignore_first_n_train_entries = 500

    for lt in tqdm(lead_time):
        # Process training data
        quantiles = predictions_test.results[lt].quantiles
        freq = predictions_test.results[lt].freq

        predictions_item_ids = []

        # Postprocess each time series separately
        for item_id in predictions_test.results[lt].data.item_ids:
            # Get train and test data
            predictions_train_df = predictions_train.results[lt].to_dataframe(item_id=item_id).iloc[ignore_first_n_train_entries:].dropna()
            predictions_test_df = predictions_test.results[lt].to_dataframe(item_id=item_id)

            # Calculate log and avoid negative values by adding a constant - tried out multiple options here
         
            # this does not work well because there might still be negative values in the test set
            # min_val = predictions_train_df[quantiles + ["target"]].min().min()
            # epsilon = max(1, -min_val + 1)

            # works better
            # min_val = predictions_train_df[quantiles + ["target"]].max().max()
            # epsilon = max(1, min_val)
            epsilon = 100_0000

            log_df_train = np.log(predictions_train_df[quantiles + ["target"]] + epsilon)
            log_df_test = np.log(predictions_test_df[quantiles] + epsilon)

            # Prepare data for MLE
            log_df_train["std_target"] = log_df_train["target"].rolling(20, min_periods=20, center=True).std()
            log_df_train = log_df_train.dropna()
            M_train = log_df_train[0.5].values  # Median prediction
            M_test = log_df_test[0.5].values
            y_mu_train = log_df_train["target"].values
            y_sigma_train = log_df_train["std_target"].values
            IQR_test = (log_df_test[0.9] - log_df_test[0.1]).values
            IQR_train = (log_df_train[0.9] - log_df_train[0.1]).values  # IQR

            # Estimate initial parameters
            initial_params = estimate_init_params(M_train, IQR_train, y_mu_train, y_sigma_train)

            # Minimize negative log-likelihood
            result = minimize(fun=neg_log_likelihood, args=(M_train, IQR_train, y_mu_train), x0=initial_params, method="Nelder-Mead", options={"maxiter": 10000})

            if not result.success:
                raise ValueError(f"Optimization failed for lead time {lt}")

            # Extract results
            a, b, c, d = result.x

            # Predict mean and std for test data
            mu_test = a + b * M_test
            sigma_test = c + d * IQR_test

            # Generate predicted quantiles in log space
            log_predictions = stats.norm.ppf(np.array(quantiles).reshape(-1, 1), loc=mu_test, scale=sigma_test).T

            # Convert back to original scale and store results
            predictions_item_ids.append(np.exp(log_predictions) - epsilon)

        # Create numpy array
        predictions_item_ids = np.vstack(predictions_item_ids)

        # Store results
        postprocessing_results[lt] = PredictionLeadTime(
            lead_time=lt, predictions=torch.tensor(predictions_item_ids), quantiles=quantiles, freq=freq, data=predictions_test.results[lt].data
        )

    return PredictionLeadTimes(results=postprocessing_results)


def postprocess_quantreg(predictions_train: PredictionLeadTimes, predictions_test: PredictionLeadTimes) -> PredictionLeadTimes:
    """Postprocess predictions using quantile regression"""

    lead_time = list(predictions_test.results.keys())
    postprocessing_results = {}

    ignore_first_n_train_entries = 500

    for lt in tqdm(lead_time):

        quantiles = predictions_test.results[lt].quantiles
        freq = predictions_test.results[lt].freq

        # prepare predictions for postprocessing
        predictions_train_df = predictions_train.results[lt].to_dataframe().iloc[ignore_first_n_train_entries:].dropna()
        predictions_test_df = predictions_test.results[lt].to_dataframe()
        cols_rename = {q: f"feature_{q}" for q in quantiles}
        predictions_test_df = predictions_test_df.rename(columns=cols_rename)
        predictions_train_df = predictions_train_df.rename(columns=cols_rename)
        cols_to_keep = list(cols_rename.values()) + ["target"]
        test_data = TabularDataFrame(predictions_test_df[cols_to_keep])
        train_data = TabularDataFrame(predictions_train_df[cols_to_keep])

        # store results
        qr_models = {}
        test_results = {}

        # fit a quantile regression for each quantile and make predictions on test dataset
        for q in quantiles:

            x_train = np.log(train_data[f"feature_{q}"].values.reshape(-1, 1))
            y_train = np.log(train_data["target"].values)
            x_test = np.log(test_data[f"feature_{q}"].values.reshape(-1, 1))

            # Add constant for intercept
            x_train = sm.add_constant(x_train)
            x_test = sm.add_constant(x_test)

            # Fit quantile regression model
            model = sm.QuantReg(y_train, x_train)
            qr_models[q] = model.fit(q=q, max_iter=2000)

            # Predict on test data
            predictions = qr_models[q].predict(x_test)
            test_results[q] = np.exp(predictions)

        postprocessing_results[lt] = PredictionLeadTime(lead_time=lt, predictions=torch.tensor(np.array(list(test_results.values())).T), freq=freq, data=test_data)

    return PredictionLeadTimes(results=postprocessing_results)
