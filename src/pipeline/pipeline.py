import torch
from typing import Dict, List, Optional, Type, Union, Literal, Tuple
from src.core.timeseries_evaluation import ForecastCollection, TimeSeriesForecast, HorizonForecast, TabularDataFrame, DIR_BACKTESTS, DIR_MODELS, DIR_POSTPROCESSORS, ITEMID, TARGET
from src.core.base import AbstractPostprocessor, AbstractPredictor, load_class_from_path
from src.core.utils import CustomJSONEncoder
from autogluon.timeseries import TimeSeriesDataFrame
import pandas as pd
from src.core.base import AbstractPipeline
from pathlib import Path
import json
from typing import Type
import joblib
import logging

PIPELINE_CONFIG_FILE_NAME = "pipeline_config.json"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s")


class ForecastingPipeline(AbstractPipeline):
    def __init__(
        self,
        model: Type[AbstractPredictor],
        model_kwargs: Dict,
        postprocessors: Optional[List[Type[AbstractPostprocessor]]] = None,
        postprocessor_kwargs: Optional[List[Dict]] = None,
        output_dir: Optional[Union[Path, str]] = None,
    ):
        super().__init__(model, model_kwargs, postprocessors, postprocessor_kwargs, output_dir)

        # define storage directory
        self.pipeline_dir_models = self.output_dir / DIR_MODELS
        self.pipeline_dir_postprocessors = self.output_dir / DIR_POSTPROCESSORS
        self.pipeline_dir_backtests = self.output_dir / DIR_BACKTESTS

        # add output directory
        self.postprocessors = postprocessors
        self.model = model
        self.model_kwargs = model_kwargs

        # Need to be fitted
        self.predictor = None
        self.postprocessor_dict: Dict[str, AbstractPostprocessor] = {}

        # create predictor and postprocessor
        self.create_predictor()
        self.create_postprocessors()

    def create_predictor(self):
        self.model_kwargs.update({"output_dir": self.pipeline_dir_models})
        self.predictor = self.model(**self.model_kwargs)

    def create_postprocessors(self):
        if self.postprocessors is None:
            raise ValueError("No postprocessors specified.")

        if self.postprocessor_kwargs is None:
            self.postprocessor_kwargs = [{} for p in self.postprocessors]

        assert len(self.postprocessor_kwargs) == len(self.postprocessors)

        for p_kwarg in self.postprocessor_kwargs:
            p_kwarg.update({"output_dir": self.pipeline_dir_postprocessors})

        self.postprocessor_dict: Dict[str, AbstractPostprocessor] = {}
        for pp, kwargs in zip(self.postprocessors, self.postprocessor_kwargs):
            pp_instance = pp(**kwargs)
            self.postprocessor_dict.update({pp_instance.class_name: pp_instance})

    def save(self) -> None:
        """Save the pipeline configuration to a JSON file and store predictors and postprocessors as joblib."""

        logging.info("Saving Pipeline to specified output directory: %s", self.output_dir)

        config = self.get_init_params()

        config_file_path = self.output_dir / "pipeline_config.json"
        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=4, cls=CustomJSONEncoder)
        logging.info("Pipeline configuration saved to: %s", config_file_path)

        self.predictor.save()

        if self.postprocessor_dict is not None:
            for postprocessor in self.postprocessor_dict.values():
                postprocessor.save()

        logging.info('Pipeline saved successfully. Reload Pipeline using: ForecastingPipeline.from_pretrained("%s")', self.output_dir)

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "ForecastingPipeline":
        """
        Load a ForecastingPipeline from a saved directory.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the directory containing the saved pipeline configuration, model, and postprocessors.

        Returns
        -------
        ForecastingPipeline
            The loaded ForecastingPipeline instance.
        """
        pipeline_dir = Path(path)

        with open(pipeline_dir / PIPELINE_CONFIG_FILE_NAME, "r") as f:
            config: dict = json.load(f)

        # Load model class
        config["model"] = load_class_from_path(config["model"])

        # Handle special fields in model_kwargs
        model_kwargs = config.get("model_kwargs", {})
        if "freq" in model_kwargs:  # TODO: also support DateOffset
            model_kwargs["freq"] = pd.Timedelta(model_kwargs["freq"])
        config["model_kwargs"] = model_kwargs

        # Load postprocessor classes
        config["postprocessors"] = [load_class_from_path(pp) for pp in config.get("postprocessors", [])]

        # Recreate the pipeline
        pipeline = ForecastingPipeline(
            model=config["model"],
            model_kwargs=config["model_kwargs"],
            postprocessors=config["postprocessors"],
            postprocessor_kwargs=config["postprocessors_kwargs"],
            output_dir=pipeline_dir,
        )

        # Load predictor
        models = list((pipeline_dir / DIR_MODELS).glob("*.joblib"))
        if not models:
            logging.info("No models found in %s. Skipping model loading.", pipeline_dir / DIR_MODELS)
        else:
            pipeline.predictor = joblib.load(models[0])

        # Load postprocessors
        postprocessor_files = list((pipeline_dir / DIR_POSTPROCESSORS).glob("*.joblib"))
        if not postprocessor_files:
            logging.info("No postprocessors found in %s. Skipping postprocessor loading.", pipeline_dir / DIR_POSTPROCESSORS)
        else:
            for pp_file in postprocessor_files:
                name = pp_file.stem
                pipeline.postprocessor_dict[name] = joblib.load(pp_file)

        return pipeline

    def backtest(
        self,
        data: Union[TimeSeriesDataFrame, TabularDataFrame],
        test_start_date: pd.Timestamp,
        test_end_date: Optional[pd.Timestamp] = None,
        rolling_window_eval: bool = False,
        train_window_size: Optional[pd.DateOffset] = None,
        val_window_size: Optional[pd.DateOffset] = None,
        test_window_size: Optional[pd.DateOffset] = None,
        train: bool = False,
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]] = None,
        save_results: bool = False,
    ) -> Dict[str, ForecastCollection]:
        """
        Run a backtest over the specified time period.

        Parameters
        ----------
        data : Union[TimeSeriesDataFrame, TabularDataFrame]
            The dataset to use for training and prediction
        test_start_date : pd.Timestamp
            Start date of the test set.
        test_end_date : Optional[pd.Timestamp], optional
            End date of the test set. Defaults to last available timestamp.
        rolling_window_eval : bool, optional
            Whether to perform rolling window evaluation. Defaults to False.
        train_window_size : Optional[pd.DateOffset], optional
            Size of the training window. Defaults to None.
        val_window_size : Optional[pd.DateOffset], optional
            Size of the validation window. Defaults to None.
        test_window_size : Optional[pd.DateOffset], optional
            Size of the test window. Defaults to None.
        train : bool, optional
            Whether to train the model during backtesting. Defaults to False.
        calibration_based_on : Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]], optional
            Strategy for calibrating postprocessors. Defaults to None.
        save_results : bool
            Whether to save backtest results. Defaults to False.

        Returns
        -------
        ForecastCollection
            The predictions over the backtest period.
        """

        logging.info("Start E2E backtesting...")

        results = {}
        info = {}

        if test_end_date is None:
            test_end_date = data.index.get_level_values("timestamp").max()

        if rolling_window_eval:
            if not test_window_size:
                logging.info("test_window_size not specified. Set it to 1 year as default")
                test_window_size = pd.DateOffset(years=1)

            while test_start_date < test_end_date:
                results[test_start_date], info[test_start_date] = self._train_predict_postprocess(
                    data, test_start_date, train, train_window_size, val_window_size, test_window_size, calibration_based_on, evaluate=True
                )
                test_start_date += test_window_size

            results = self._merge_results(results)
        else:
            results, info = self._train_predict_postprocess(data, test_start_date, train, train_window_size, val_window_size, test_window_size, calibration_based_on, evaluate=True)

        if save_results:
            backtest_params = {
                "test_start_date": test_start_date,
                "test_end_date": test_end_date,
                "rolling_window_eval": rolling_window_eval,
                "train_window_size": train_window_size,
                "val_window_size": val_window_size,
                "test_window_size": test_window_size,
                "train": train,
                "calibration_based_on": calibration_based_on,
                "additional_info": info,
            }
            self._save_backtest_results(results, backtest_params)

        return results, info

    def _save_backtest_results(self, results: Dict[str, ForecastCollection], backtest_params: Dict) -> None:
        """Save backtest results and config."""

        logging.info("Storing backtest results...")

        for method, result in results.items():
            save_path = self.pipeline_dir_backtests / method

            create_dir(save_path)

            # Add information
            eval_config_info = {}
            eval_config_info = {"applied_postprocessor": None if method == "raw" else method}
            eval_config_info.update(result.get_crps(mean_time=True).to_dict())
            eval_config_info.update(result.get_empirical_coverage_rates(mean_lead_times=True).to_dict())
            eval_config_info.update(result.get_quantile_scores(mean_lead_times=True).to_dict())
            config = self.get_config(backtest_params, eval_config_info)

            # Save config
            config_path = save_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4, cls=CustomJSONEncoder)
            logging.info("Saved backtest configuration for `%s` including evaluation results to: %s.", method, config_path)
            # Save predictions
            result.save(save_path / "predictions.joblib")

    def split_data(
        self,
        data: Union[TimeSeriesDataFrame, TabularDataFrame],
        test_start_date: pd.Timestamp,
        train_window_size: Optional[pd.DateOffset],
        val_window_size: Optional[pd.DateOffset],
        test_window_size: Optional[pd.DateOffset],
    ) -> Tuple[Union[TimeSeriesDataFrame, TabularDataFrame], Union[TimeSeriesDataFrame, TabularDataFrame, None], Union[TimeSeriesDataFrame, TabularDataFrame]]:
        """
        Split data into training, validation, and testing sets.

        Parameters
        ----------
        data : Union[TimeSeriesDataFrame, TabularDataFrame]
            The dataset to use for training and prediction
        test_start_date : pd.Timestamp
            Start date of the test set.
        train_window_size : Optional[pd.DateOffset]
            Size of the training window.
        val_window_size : Optional[pd.DateOffset]
            Size of the validation window.
        test_window_size : Optional[pd.DateOffset]
            Size of the test window.

        Returns
        -------
        Tuple[data_train, data_val, data_test]
            Split datasets for training, validation (optional), and testing.
        """
        logging.info("Starting data split operation.")

        data_val = None
        logging.debug("Splitting data based on the test start date: %s", test_start_date)
        data_train, data_test = data.split_by_time(test_start_date)
        logging.debug("Training set and testing set created.")

        if val_window_size:
            logging.debug("Splitting training data for validation with window size: %s", val_window_size)
            data_train, data_val = data_train.split_by_time(test_start_date - val_window_size)
            logging.debug("Validation set created.")

        if train_window_size:
            logging.debug("Adjusting training data based on window size: %s", train_window_size)
            _, data_train = data_train.split_by_time(test_start_date - train_window_size)
            logging.debug("Training set adjusted.")

        if test_window_size:
            logging.debug("Adjusting test set based on window size: %s", test_window_size)
            data_test, _ = data_test.split_by_time(test_start_date + test_window_size)
            logging.debug("Test set adjusted.")

        logging.info("Data split operation completed successfully.")

        return data_train, data_val, data_test

    def train(
        self,
        data_train: Union[TimeSeriesDataFrame, TabularDataFrame],
        data_val: Optional[Union[TimeSeriesDataFrame, TabularDataFrame]] = None,
    ) -> Dict:
        """
        Train the predictor.

        Parameters
        ----------
        data_train : Union[TimeSeriesDataFrame, TabularDataFrame]
            The training data.
        data_val : Optional[Union[TimeSeriesDataFrame, TabularDataFrame]], optional
            Optional validation data. Defaults to None.

        Returns
        -------
        Dict
            Training information
        """

        info = {}

        start_time = pd.Timestamp.now()
        logging.info("Starting training process from %s to %s", data_train.index.get_level_values("timestamp").min(), data_train.index.get_level_values("timestamp").max())

        # Check if validation data is provided and log accordingly
        if data_val is not None:
            logging.info("Validation data from %s to %s", data_val.index.get_level_values("timestamp").min(), data_val.index.get_level_values("timestamp").max())
        else:
            logging.info("No validation data provided.")

        # Initialize the predictor
        logging.info("Initializing predictor with model: %s", self.model.__name__)
        self.create_predictor()

        # Fit the predictor with training (and optional validation) data
        logging.info("Fitting predictor to the training data...")
        self.predictor.fit(data_train, data_val)
        info["predictor_execution_time"] = self.predictor.fit_execution_time

        end_time = pd.Timestamp.now()
        logging.info("Pipeline training completed in %s seconds.", (end_time - start_time).total_seconds())

        return info

    def predict(
        self,
        data_test: Union[TimeSeriesDataFrame, TabularDataFrame],
        data_previous_context: Optional[Union[TimeSeriesDataFrame, TabularDataFrame]] = None,
    ) -> Dict[str, ForecastCollection]:
        """
        predict on the test data.

        Parameters
        ----------
        data_test : Union[TimeSeriesDataFrame, TabularDataFrame]
            The test dataset.
        data_previous_context : Union[TimeSeriesDataFrame, TabularDataFrame]
            The previous context data. This is used by some predictors.
        Returns
        -------
        Dict[str, ForecastCollection]
            Dictionary with raw predictions.
        """

        start_time = pd.Timestamp.now()
        logging.info("Starting prediction for test data from %s to %s", data_test.index.get_level_values("timestamp").min(), data_test.index.get_level_values("timestamp").max())

        # Run the prediction
        logging.info("Running prediction using the model: %s", self.predictor.__class__.__name__)
        predictions = self.predictor.predict(data=data_test, previous_context_data=data_previous_context)
        logging.info("Prediction completed successfully.")

        end_time = pd.Timestamp.now()
        logging.info("Prediction completed in %s seconds.", (end_time - start_time).total_seconds())

        return {self.predictor.__class__.__name__: predictions}

    def train_postprocessors(
        self, calibration_data: Union[TimeSeriesDataFrame, TabularDataFrame], previous_context_data: Union[TimeSeriesDataFrame, TabularDataFrame, None] = None
    ) -> Dict:
        """
        Fit the postprocessors based on calibration data.

        Parameters
        ----------
        calibration_data : Union[TimeSeriesDataFrame, TabularDataFrame]
            The calibration data. Used to fit the postprocessor.
        """

        # TODO: this function could return multiple information regarding the training
        info = {}

        start_time = pd.Timestamp.now()
        logging.info(
            "Starting postprocessor training with calibration data from %s to %s",
            calibration_data.index.get_level_values("timestamp").min(),
            calibration_data.index.get_level_values("timestamp").max(),
        )

        # Check if postprocessors are available
        if not self.postprocessors:
            logging.error("No postprocessors specified. Cannot proceed with training.")
            raise ValueError("No postprocessors specified.")

        # Generate calibration predictions
        logging.info("Running prediction on calibration data using the model: %s", self.predictor.__class__.__name__)
        calibration_predictions = self.predictor.predict(calibration_data, previous_context_data)
        logging.info("Calibration predictions completed successfully.")

        # initialize postprocessors
        self.create_postprocessors()

        info["postprocessors_execution_time"] = {}

        # Fit postprocessors
        for name, postprocessor in self.postprocessor_dict.items():

            # Fit postprocessor on calibration data
            logging.info("Fitting postprocessor: %s", name)
            postprocessor.fit(data=calibration_predictions)
            info["postprocessors_execution_time"][name] = postprocessor.fit_execution_time
            logging.info("Successfully fitted postprocessor: %s", name)

        end_time = pd.Timestamp.now()
        logging.info("Postprocessors training completed in %s seconds.", (end_time - start_time).total_seconds())

        return info

    def apply_postprocessing(self, predictions: Dict[str, ForecastCollection]) -> Dict[str, ForecastCollection]:
        """
        Apply postprocessing to predictions.

        Parameters
        ----------
        predictions : Dict[str, ForecastCollection]
            The predictions. Keys correspond to the utilized model, e.g. `Chronos` or `QuantileRegression`

        Returns
        -------
        Dict[str, ForecastCollection]
            Dictionary with processed predictions.
        """
        if not self.postprocessor_dict:
            logging.info("No postprocessors configured. Returning raw predictions.")
            return predictions

        logging.info("Applying postprocessing to predictions...")
        for name, postprocessor in self.postprocessor_dict.items():
            logging.info("Postprocessing predictions using postprocessor: %s", name)

            # Apply the postprocessing and store predictions as additional key value pair
            predictions[name] = postprocessor.postprocess(data=predictions[self.predictor.__class__.__name__])
            logging.info("Postprocessing complete for %s", name)

        logging.info("Postprocessing completed for all models.")
        return predictions

    def _train_predict_postprocess(
        self,
        data: Union[TimeSeriesDataFrame, TabularDataFrame],
        test_start_date: pd.Timestamp,
        train: bool,
        train_window_size: Optional[pd.DateOffset],
        val_window_size: Optional[pd.DateOffset],
        test_window_size: Optional[pd.DateOffset],
        calibration_based_on: Optional[Union[Literal["val", "train", "train_val"], pd.DateOffset]],
        evaluate: bool = False,
    ) -> Tuple[Dict[str, ForecastCollection], Dict]:
        """Train, predict, and postprocess wrapper for internal backtesting."""

        info = {}

        data_train, data_val, data_test = self.split_data(data, test_start_date, train_window_size, val_window_size, test_window_size)

        if evaluate:
            logging.info("evaluate is set to true. Removing all item_ids which contain only nans in target column.")
            data_test = data_test[data_test.groupby(level=ITEMID)[TARGET].transform(lambda x: not x.isna().all())].copy()
            if len(data_test) == 0:
                raise ValueError("Test data is empty after dropping all rows with target=nan. Check data.")
        if train:
            info["model"] = self.train(data_train, data_val)
        else:
            logging.info("Skipping model training because `train=False`.")

        predictions = self.predict(data_test, data.split_by_time(data_test.index.get_level_values("timestamp").min())[0])
        # TODO: save predictions directly
        if self.postprocessors is not None:
            if calibration_based_on == "val":
                calibration_data = data_val
            elif calibration_based_on == "train":
                calibration_data = data_train
            elif calibration_based_on == "train_val":
                calibration_data = pd.concat([data_train, data_val]).sort_index()
            else:
                raise ValueError(f"Invalid calibration_based_on: {calibration_based_on}")

            info["postprocessors"] = self.train_postprocessors(calibration_data)
            predictions = self.apply_postprocessing(predictions)

        return predictions, info

    def _merge_results(self, backtest_results: Dict[pd.Timestamp, Dict[str, ForecastCollection]]) -> Dict[str, ForecastCollection]:

        all_predictors = {key for result in backtest_results.values() for key in result}

        logging.info("Merge results for each of the following predictors/postprocessors %s", all_predictors)

        merged_results = {}
        for predictor_name in all_predictors:
            merged_results[predictor_name] = self.merge_predictor_backtest(backtest_results, predictor_name)

        logging.info("Merge completed.")

        return merged_results

    def merge_predictor_backtest(
        self,
        backtest_results: Dict[pd.Timestamp, Dict[str, ForecastCollection]],
        predictor_name: str,
    ) -> ForecastCollection:
        """
        Merge a single predictor's backtest windows into one ForecastCollection.

        Parameters
        ----------
        backtest_results : Dict[pd.Timestamp, Dict[str, ForecastCollection]]
            mapping from window-start date to all predictors' ForecastCollection.
        predictor_name : str
            which predictor to merge.

        Returns
        -------
            A ForecastCollection whose TimeSeriesForecasts have all windows stitched together.
        """
        # 1) sort by window-start date
        dates = sorted(backtest_results.keys())

        # 2) grab each window's ForecastCollection for this predictor
        per_window_fc = [backtest_results[d][predictor_name] for d in dates]

        # 3) identify all item_ids
        item_ids = per_window_fc[0].get_item_ids()

        merged_ts: Dict[int, TimeSeriesForecast] = {}

        for item_id in item_ids:
            # 4) collect this item’s TimeSeriesForecast from each window
            ts_list = [fc.get_time_series_forecast(item_id) for fc in per_window_fc]

            lead_times = ts_list[0].get_lead_times()
            quantiles = ts_list[0].quantiles
            freq = ts_list[0].freq

            # 5) concatenate underlying dataframes
            data_concat = pd.concat([ts.data for ts in ts_list])

            # 6) for each lead time, stack predictions
            lead_time_forecasts: Dict[int, HorizonForecast] = {}
            for lt in lead_times:
                # gather the raw prediction tensors in window order
                preds = torch.vstack([ts.get_lead_time_forecast(lt).predictions for ts in ts_list])
                # build a new HorizonForecast
                lead_time_forecasts[lt] = HorizonForecast(lead_time=lt, predictions=preds)

            # 7) re-create the merged TimeSeriesForecast
            merged_ts[item_id] = TimeSeriesForecast(
                item_id=item_id,
                lead_time_forecasts=lead_time_forecasts,
                data=data_concat,
                quantiles=quantiles,
                freq=freq,
            )

        return ForecastCollection(item_ids=merged_ts)
