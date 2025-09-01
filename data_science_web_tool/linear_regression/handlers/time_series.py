from collections import deque
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from preprocessing.models import Data


class LinearRegressionBase:
    SUPPORTED_COLUMN_TYPES = (
        "int64",
        "float64",
    )

    def __init__(
        self,
        data: Data,
        column_name: str,
        lag_size: int,
        max_tree_depth: int | None = None,
        forecast_horizon: int | None = None,
        train_percentage: int | None = None,
        validation_percentage: int | None = None,
        test_percentage: int | None = None,
        target_mode: Literal["delta", "direct"] = "delta",
    ):
        self.data = data
        self.column_name = column_name
        self.column_name_lagged = self.column_name + "_lagged"
        self.lag_size = lag_size
        self.max_tree_depth = max_tree_depth
        self.forecast_horizon = forecast_horizon or 0
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage
        self.target_mode = target_mode
        print()

    def _forecast_future_values(
        self,
        full_df: pd.DataFrame,
        model: DecisionTreeRegressor | LinearRegression,
    ) -> list:
        """
        Tries to predict n future value by predicting n-1 future value and appending it to the prediction list.
        """
        future_predictions = []
        if not self.forecast_horizon:
            return future_predictions

        p = int(self.lag_size)
        history = deque(full_df[self.column_name].iloc[-p:].astype(float).tolist(), maxlen=p)
        for _ in range(self.forecast_horizon):
            x_in = np.array([[history[0]]], dtype=float)
            y = float(model.predict(x_in)[0])
            if self.target_mode == "delta":
                next_level = history[0] + y
            else:
                next_level = y

            history.append(next_level)
            future_predictions.append(next_level)

        return future_predictions

    def _get_model_data_sets(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
        """
        Returns training, validation and test data sets by getting data percentages selected by the user.
        """
        total_rows: int = len(df)

        train_size = round(total_rows * self.train_percentage / 100)
        val_size = round(total_rows * self.validation_percentage / 100)
        if train_size + val_size > total_rows:
            val_size = total_rows - train_size

        # Now define the index bounds
        train_end = train_size
        val_end = train_size + val_size

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        return train_df, val_df, test_df

    def _get_model_predictions_and_statistics(
        self,
        df: pd.DataFrame,
        model: LinearRegression | DecisionTreeRegressor,
        keys_prefix: str = "",
    ) -> tuple[pd.DataFrame, dict]:
        # model predicts the differences, value = lag + delta
        X = df[[self.column_name_lagged]]
        raw_pred = model.predict(X)
        if self.target_mode == "delta":
            predictions = df[self.column_name_lagged] + raw_pred
        else:
            predictions = pd.Series(raw_pred, index=df.index)

        actual = df[self.column_name]
        return predictions, {
            keys_prefix + "r_2": r2_score(actual, predictions),
            keys_prefix + "mse": mean_squared_error(actual, predictions),
            keys_prefix + "mae": mean_absolute_error(actual, predictions),
            keys_prefix + "rmse": root_mean_squared_error(actual, predictions),
            keys_prefix + "mape": mean_absolute_percentage_error(actual, predictions),
        }


class LinearRegressionTimeSeriesHandler(LinearRegressionBase):
    """
    Linear regression time series handler with recursive forecast support.
    """

    def handle(self):
        df: pd.DataFrame = self.data.get_df()
        df[self.column_name_lagged] = df[self.column_name].shift(self.lag_size)
        df["values_diff"] = df[self.column_name] - df[self.column_name_lagged]
        df.dropna(inplace=True)
        # df.reset_index(inplace=True)

        train_df, val_df, test_df = self._get_model_data_sets(df)
        model_metadata, forecast = self._linear_regression(
            df,
            train_df,
            val_df,
            test_df,
        )
        return model_metadata, forecast

    def _linear_regression(
        self,
        full_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple:
        x = train_df[[self.column_name_lagged]]
        y = train_df["values_diff"] if self.target_mode == "delta" else train_df[self.column_name]

        # Train the model to predict values differences
        model = LinearRegression()
        model.fit(x, y)

        # Get statistics for validation and train sets
        val_predictions, val_statistics = self._get_model_predictions_and_statistics(val_df, model, keys_prefix="val_")
        test_predictions, test_statistics = self._get_model_predictions_and_statistics(
            test_df,
            model,
            keys_prefix="test_",
        )
        model_metadata = {
            "train_values": train_df[self.column_name_lagged],
            "val_predictions": val_predictions,
            "val_statistics": val_statistics,
            "test_predictions": test_predictions,
            "test_statistics": test_statistics,
            # y = m * x + b
            "slope": model.coef_[0],  # m
            "intercept": model.intercept_,  # b
        }

        future_forecast = self._forecast_future_values(full_df, model)
        return model_metadata, future_forecast


class RegressionTreeTimeSeriesHandler(LinearRegressionBase):
    """
    Regression tree time series handler.
    """

    def handle(self):
        df: pd.DataFrame = self.data.get_df()
        df[self.column_name_lagged] = df[self.column_name].shift(self.lag_size)
        df["values_diff"] = df[self.column_name] - df[self.column_name_lagged]
        df.dropna(inplace=True)
        # df.reset_index(inplace=True)

        train_df, val_df, test_df = self._get_model_data_sets(df)
        model_metadata, forecast = self._regression_tree(
            df,
            train_df,
            val_df,
            test_df,
        )
        return model_metadata, forecast

    def _regression_tree(
        self,
        full_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple:
        x = train_df[[self.column_name_lagged]]
        y = train_df["values_diff"] if self.target_mode == "delta" else train_df[self.column_name]

        model = DecisionTreeRegressor(
            max_depth=self.max_tree_depth,
            random_state=42,  # so it's possible to reproduce results
            min_samples_leaf=10,
        )
        model.fit(x, y)

        # Get statistics for validation and train sets
        val_predictions, val_statistics = self._get_model_predictions_and_statistics(val_df, model, keys_prefix="val_")
        test_predictions, test_statistics = self._get_model_predictions_and_statistics(
            test_df,
            model,
            keys_prefix="test_",
        )
        model_metadata = {
            "train_values": train_df[self.column_name_lagged],
            "val_predictions": val_predictions,
            "val_statistics": val_statistics,
            "test_predictions": test_predictions,
            "test_statistics": test_statistics,
            "val_tree_levels": self.get_tree_step_series(val_df, model),
            "test_tree_levels": self.get_tree_step_series(test_df, model),
        }

        future_forecast = self._forecast_future_values(full_df, model)
        return model_metadata, future_forecast

    def get_tree_step_series(
        self,
        df: pd.DataFrame,
        model: DecisionTreeRegressor,
    ) -> pd.Series:
        """
        Gets tree levels.
        """
        X = df[[self.column_name_lagged]]
        y_hat = pd.Series(model.predict(X), index=df.index, name="tree_pred_raw")
        if self.target_mode == "delta":
            levels = df[self.column_name_lagged] + y_hat
        else:
            levels = y_hat

        levels.name = "tree_levels"
        return levels
