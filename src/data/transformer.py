from src.core.base import AbstractDataTransformer
import numpy as np
import logging
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from typing import Optional, Literal, Union, Any
from sklearn.preprocessing import PowerTransformer, FunctionTransformer


def log_transform(x, epsilon):
    return np.log(x + epsilon)


def log_inverse(x, epsilon):
    return np.exp(x) - epsilon


def identity(x):
    return x


class DataTransformer(AbstractDataTransformer):
    """
    A flexible transformer supporting various methods:
      - 'yeo-johnson' or 'box-cox' via sklearn PowerTransformer
      - 'log' with an additive epsilon
      - 'arcsinh' (inverse hyperbolic sine)
      - None or unrecognized => identity

    Preserves pandas DataFrame structure when passed.
    If a fitted transformer was trained on a single feature, and new data has multiple columns,
    it applies the transformation column-wise.
    """

    def __init__(self, method: Optional[Literal["yeo-johnson", "box-cox", "log", "arcsinh"]] = None, epsilon: float = 1e-8, **power_kwargs: Any):
        self.method = method
        self.epsilon = epsilon

        if method in {"yeo-johnson", "box-cox"}:
            self.transformer = PowerTransformer(method=method, **power_kwargs)
            self.requires_fit = True

        elif method == "log":
            self.transformer = FunctionTransformer(
                func=lambda x: log_transform(x, self.epsilon),
                inverse_func=lambda x: log_inverse(x, self.epsilon),
                validate=False,
            )
            self.requires_fit = False

        elif method == "arcsinh":
            self.transformer = FunctionTransformer(
                func=np.arcsinh,
                inverse_func=np.sinh,
                validate=False,
            )
            self.requires_fit = False

        else:
            self.transformer = FunctionTransformer(
                func=identity,
                inverse_func=identity,
                validate=False,
            )
            self.requires_fit = False

        logging.info("Set data transformation: %s", method)

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """Fit the transformer if required (only for PowerTransformer)."""
        if self.requires_fit:
            arr, _ = self._to_array(X)
            self.transformer.fit(arr)

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Apply the forward transformation, with column-wise support for single-feature fits."""
        arr, meta = self._to_array(X)

        # If fitted on single feature but multiple columns provided, apply column-wise transformation
        if self.requires_fit and hasattr(self.transformer, "n_features_in_") and arr.ndim == 2 and arr.shape[1] != self.transformer.n_features_in_:
            transformed = np.empty_like(arr, dtype=float)
            for i in range(arr.shape[1]):
                col = arr[:, [i]]
                transformed[:, i] = self.transformer.transform(col).flatten()
        else:
            transformed = self.transformer.transform(arr)
        return self._to_output(transformed, meta)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Fit (if needed) and transform in one step."""
        arr, meta = self._to_array(X)
        if self.requires_fit:
            transformed = self.transformer.fit_transform(arr)
        else:
            transformed = self.transformer.transform(arr)
        return self._to_output(transformed, meta)

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """Invert the transformation, with column-wise support for single-feature fits."""
        arr, meta = self._to_array(X)
        if self.requires_fit and hasattr(self.transformer, "n_features_in_") and arr.ndim == 2 and arr.shape[1] != self.transformer.n_features_in_:
            inv = np.empty_like(arr, dtype=float)
            for i in range(arr.shape[1]):
                col = arr[:, [i]]
                inv[:, i] = self.transformer.inverse_transform(col).flatten()
        else:
            inv = self.transformer.inverse_transform(arr)
        return self._to_output(inv, meta)

    def _to_array(self, X: Union[np.ndarray, pd.DataFrame]) -> tuple:
        if isinstance(X, pd.DataFrame):
            return X.values, {"columns": X.columns, "index": X.index}

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X, None

    def _to_output(self, arr: np.ndarray, meta: Optional[dict]) -> Union[np.ndarray, pd.DataFrame]:
        if meta:
            return pd.DataFrame(arr, columns=meta["columns"], index=meta["index"])
        return arr
