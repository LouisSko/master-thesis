import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from postprocessing import neg_log_likelihood
from scipy.optimize import minimize
from timeseries import PredictionLeadTimes, PredictionLeadTime
import torch


def neg_log_likelihood(params: list, M: np.ndarray, IQR: np.ndarray, y: np.ndarray):
    """Define negative log-likelihood function for maximum likelihood estimation"""

    a, b, c, d = params
    mu = a + b * M
    sigma = c + d * IQR
    nll = -np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    return nll


def postprocess_mle(predictions_train: PredictionLeadTimes, predictions_test: PredictionLeadTimes) -> PredictionLeadTimes:
    """Postprocessing of predictions by estimating the normal distribution."""

    lead_time = list(predictions_train.results.keys())
    results_nrml = {}

    for lt in lead_time:
        # Process training data
        predictions_train_lt = predictions_train.results[lt]
        quantile_levels = predictions_train_lt.quantiles

        # TODO: get rid of the first 100 entries, improve this.
        df_train = predictions_train_lt.to_dataframe().dropna().iloc[100:].copy()
        log_df_train = np.log(df_train[quantile_levels + ["target"]])

        # Prepare data for MLE
        M_train = log_df_train[0.5].values  # Model's median prediction
        IQR_train = (log_df_train[0.9] - log_df_train[0.1]).values  # Model's IQR
        y_train = log_df_train["target"].values  # Observed log Y

        # estimate initial parameters with linear regresssion
        # mean=a+b*Median
        x_mu = sm.add_constant(M_train)
        model_mu = sm.OLS(y_train, x_mu).fit()
        a_init, b_init = model_mu.params
        # std=c+d*IQR
        log_df_train["std_target"] = log_df_train["target"].groupby("item_id").rolling(20, min_periods=20, center=True).std().droplevel(0)
        log_df_train = log_df_train.dropna()
        iqr = log_df_train[0.9] - log_df_train[0.1]
        x_sigma = sm.add_constant(iqr)
        y_sigma = log_df_train["std_target"]
        model_sigma = sm.OLS(y_sigma, x_sigma).fit()
        c_init, d_init = model_sigma.params

        initial_params = [a_init, b_init, c_init, d_init]
        result = minimize(fun=neg_log_likelihood, args=(M_train, IQR_train, y_train), x0=initial_params, method="Nelder-Mead", options={"maxiter": 10000})

        if not result.success:
            raise ValueError("Optimization failed for lead time {}".format(lt))

        a, b, c, d = result.x

        # Prepare test data
        prediction_test_lt = predictions_test.results[lt]
        df_test = prediction_test_lt.to_dataframe().iloc[100:].copy()
        log_df_test = np.log(df_test[quantile_levels + ["target"]])

        # Predict mean and std for test data
        M_test = log_df_test[0.5].values
        IQR_test = (log_df_test[0.9] - log_df_test[0.1]).values

        mu_test = a + b * M_test
        sigma_test = c + d * IQR_test

        # Generate predicted quantiles
        log_predictions = stats.norm.ppf(
            np.array(quantile_levels).reshape(-1, 1),
            loc=mu_test,
            scale=sigma_test,
        ).T

        predictions = np.exp(log_predictions)  # Convert back to original scale

        # Store results
        results_nrml[lt] = PredictionLeadTime(
            lead_time=lt, predictions=torch.tensor(predictions), quantiles=quantile_levels, freq=prediction_test_lt.freq, data=prediction_test_lt.data.iloc[100:]
        )

    return PredictionLeadTimes(results=results_nrml)
