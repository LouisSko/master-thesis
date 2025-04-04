import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from scipy.optimize import minimize
from timeseries import PredictionLeadTimes, PredictionLeadTime, TabularDataFrame
import torch
from tqdm import tqdm


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

    ignore_first_n_train_entries = 500

    for lt in tqdm(lead_time):
        # Process training data
        quantiles = predictions_test.results[lt].quantiles
        freq = predictions_test.results[lt].freq

        # TODO: might want handle ignore_first_n_train_entries differently
        predictions_train_df = predictions_train.results[lt].to_dataframe().iloc[ignore_first_n_train_entries:].dropna()
        log_df_train = np.log(predictions_train_df[quantiles + ["target"]])

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
        predictions_test_df = predictions_test.results[lt].to_dataframe()
        log_df_test = np.log(predictions_test_df[quantiles])

        # Predict mean and std for test data
        M_test = log_df_test[0.5].values
        IQR_test = (log_df_test[0.9] - log_df_test[0.1]).values

        mu_test = a + b * M_test
        sigma_test = c + d * IQR_test

        # Generate predicted quantiles
        log_predictions = stats.norm.ppf(
            np.array(quantiles).reshape(-1, 1),
            loc=mu_test,
            scale=sigma_test,
        ).T

        # Convert back to original scale
        predictions = np.exp(log_predictions)

        # Store results
        results_nrml[lt] = PredictionLeadTime(lead_time=lt, predictions=torch.tensor(predictions), quantiles=quantiles, freq=freq, data=predictions_test.results[lt].data)

    return PredictionLeadTimes(results=results_nrml)


def postprocess_quantreg(predictions_train: PredictionLeadTimes, predictions_test: PredictionLeadTimes) -> PredictionLeadTimes:
    """Postprocess predictions using quantile regression"""

    lead_time = list(predictions_test.results.keys())
    results_qr = {}

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

        results_qr[lt] = PredictionLeadTime(lead_time=lt, predictions=torch.tensor(np.array(list(test_results.values())).T), freq=freq, data=test_data)

    return PredictionLeadTimes(results=results_qr)
