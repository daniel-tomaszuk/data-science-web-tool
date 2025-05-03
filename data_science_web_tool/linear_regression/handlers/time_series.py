import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from preprocessing.models import Data


class LinearRegressionTimeSeriesHandler:
    """
    Linear regression time series handler.
    """

    SUPPORTED_COLUMN_TYPES = (
        "int64",
        "float64",
    )

    def __init__(self, data: Data, column_name: str, lag_size: int):
        self.data = data
        self.column_name = column_name
        self.column_name_lagged = self.column_name + "_lagged"
        self.lag_size = lag_size

    def handle(self):
        df: pd.DataFrame = self.data.get_df()
        df[self.column_name_lagged] = df[self.column_name].shift(self.lag_size)
        df.dropna(inplace=True)
        predictions, statistics = self._linear_regression(df)
        return predictions, statistics

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
        }
        return y_predictions, statistics
