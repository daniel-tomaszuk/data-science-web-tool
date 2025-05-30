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
    ):
        self.data = data
        self.column_name = column_name
        self.column_name_lagged = self.column_name + "_lagged"
        self.lag_size = lag_size
        self.max_tree_depth = max_tree_depth
        self.forecast_horizon = forecast_horizon or 0

    def _forecast_future_values(self, df: pd.DataFrame, model: DecisionTreeRegressor | LinearRegression) -> list:
        """
        Tries to predict n future value by predicting n-1 future value and appending it to the prediction list.
        """
        future_predictions = []
        if not self.forecast_horizon:
            return future_predictions

        last_prediction = df[self.column_name_lagged].iloc[-1]
        for _ in range(self.forecast_horizon):
            input_df = pd.DataFrame([[last_prediction]], columns=[self.column_name_lagged])
            next_pred = model.predict(input_df)[0]
            future_predictions.append(next_pred)
            last_prediction = next_pred

        return future_predictions


class LinearRegressionTimeSeriesHandler(LinearRegressionBase):
    """
    Linear regression time series handler with recursive forecast support.
    """

    def handle(self):
        df: pd.DataFrame = self.data.get_df()
        df[self.column_name_lagged] = df[self.column_name].shift(self.lag_size)
        df.dropna(inplace=True)
        predictions, statistics, forecast = self._linear_regression(df)
        return predictions, statistics, forecast

    def _linear_regression(self, df: pd.DataFrame) -> tuple:
        x = df[[self.column_name_lagged]]
        y = df[self.column_name]

        model = LinearRegression()
        model.fit(x, y)
        y_predictions = model.predict(x)
        statistics = {
            "r_2": r2_score(y, y_predictions),
            "mse": mean_squared_error(y, y_predictions),
            "mae": mean_absolute_error(y, y_predictions),
            "rmse": root_mean_squared_error(y, y_predictions),
            "mape": mean_absolute_percentage_error(y, y_predictions),

            # y = m * x + b
            "slope": model.coef_[0],  # m
            "intercept": model.intercept_,  # b
        }
        future_predictions = self._forecast_future_values(df, model)
        return y_predictions, statistics, future_predictions


class RegressionTreeTimeSeriesHandler(LinearRegressionBase):
    """
    Regression tree time series handler.
    """

    def handle(self):
        df: pd.DataFrame = self.data.get_df()
        df[self.column_name_lagged] = df[self.column_name].shift(self.lag_size)
        df.dropna(inplace=True)
        predictions, statistics, forecast = self._regression_tree(df)
        return predictions, statistics, forecast

    def _regression_tree(self, df: pd.DataFrame) -> tuple:
        x = df[[self.column_name_lagged]]
        y = df[self.column_name]

        model = DecisionTreeRegressor(max_depth=self.max_tree_depth)
        model.fit(x, y)
        y_predictions = model.predict(x)
        statistics = {
            "r_2": r2_score(y, y_predictions),
            "mse": mean_squared_error(y, y_predictions),
            "mae": mean_absolute_error(y, y_predictions),
            "rmse": root_mean_squared_error(y, y_predictions),
            "mape": mean_absolute_percentage_error(y, y_predictions),
        }
        future_predictions = self._forecast_future_values(df, model)
        return y_predictions, statistics, future_predictions
