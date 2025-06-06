from django.db import models

from linear_regression.handlers.time_series import LinearRegressionTimeSeriesHandler
from linear_regression.handlers.time_series import RegressionTreeTimeSeriesHandler
from preprocessing.models import Data


class LinearRegressionTimeSeriesResult(models.Model):
    SUPPORTED_HANDLERS = {
        "linear_regression": LinearRegressionTimeSeriesHandler,
        "regression_tree": RegressionTreeTimeSeriesHandler,
    }
    MODEL_TYPES_CHOICES = tuple(
        (model_type, model_type.replace("_", " ").title())
        for model_type in SUPPORTED_HANDLERS.keys()
    )
    data = models.ForeignKey(
        Data,
        on_delete=models.CASCADE,
        related_name="linear_regression_timeseries_results",
    )

    model_type = models.CharField()
    target_column = models.CharField()
    lag_size = models.IntegerField(blank=True, null=True)
    max_tree_depth = models.IntegerField(blank=True, null=True)
    forecast_horizon = models.IntegerField(blank=True, null=True)
    predictions = models.JSONField(blank=True, null=True)
    forecast = models.JSONField(blank=True, null=True)

    r_2 = models.FloatField(blank=True, null=True, help_text="R2 score")
    mse = models.FloatField(blank=True, null=True, help_text="Mean squared error")
    mae = models.FloatField(blank=True, null=True, help_text="Mean absolute error")
    rmse = models.FloatField(blank=True, null=True, help_text="Root mean square error")
    mape = models.FloatField(
        blank=True,
        null=True,
        help_text="Mean absolute percentage error",
    )
    slope = models.FloatField(blank=True, null=True)
    intercept = models.FloatField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_statistics(self) -> dict:
        return {
            "r_2": round(self.r_2, 4),
            "mse": round(self.mse, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4),
            "slope": round(self.slope, 4) if self.slope else None,
            "intercept": round(self.intercept, 4) if self.intercept else None,
        }

    def __str__(self):
        return (
            f"Linear Regression TimeSeries Result {self.data} -> {self.target_column}"
        )
