from django.db import models

from linear_regression.handlers.time_series import LinearRegressionTimeSeriesHandler
from linear_regression.handlers.time_series import RegressionTreeTimeSeriesHandler
from preprocessing.models import Data


class LinearRegressionTimeSeriesResult(models.Model):
    REGRESSION_TREE_MODEL = "regression_tree"
    LINEAR_REGRESSION_MODEL = "linear_regression"

    DIRECT_TARGET_MODE = "direct"
    DELTA_TARGET_MODE = "delta"
    SUPPORTED_HANDLERS = {
        "linear_regression": LinearRegressionTimeSeriesHandler,
        "regression_tree": RegressionTreeTimeSeriesHandler,
    }
    MODEL_TYPES_CHOICES = tuple(
        (model_type, model_type.replace("_", " ").title()) for model_type in SUPPORTED_HANDLERS.keys()
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
    target_mode = models.CharField(blank=True, null=True, default=DELTA_TARGET_MODE)

    train_values = models.JSONField(blank=True, null=True, help_text="Train data set")
    val_predictions = models.JSONField(blank=True, null=True, help_text="Validation data set predictions")
    test_predictions = models.JSONField(blank=True, null=True, help_text="Test data set predictions")

    forecast = models.JSONField(blank=True, null=True)

    train_percentage = models.FloatField(blank=True, null=True)
    validation_percentage = models.FloatField(blank=True, null=True)
    test_percentage = models.FloatField(blank=True, null=True)

    val_tree_levels = models.JSONField(blank=True, null=True)
    test_tree_levels = models.JSONField(blank=True, null=True)

    val_r_2 = models.FloatField(blank=True, null=True, help_text="R2 score for validation data")
    val_mse = models.FloatField(blank=True, null=True, help_text="Mean squared error for validation data")
    val_mae = models.FloatField(blank=True, null=True, help_text="Mean absolute error for validation data")
    val_rmse = models.FloatField(blank=True, null=True, help_text="Root mean square error for validation data")
    val_mape = models.FloatField(
        blank=True,
        null=True,
        help_text="Mean absolute percentage error for validation data",
    )

    test_r_2 = models.FloatField(blank=True, null=True, help_text="R2 score for test data")
    test_mse = models.FloatField(blank=True, null=True, help_text="Mean squared error for test data")
    test_mae = models.FloatField(blank=True, null=True, help_text="Mean absolute error for test data")
    test_rmse = models.FloatField(blank=True, null=True, help_text="Root mean square error for test data")
    test_mape = models.FloatField(
        blank=True,
        null=True,
        help_text="Mean absolute percentage error for test data",
    )

    slope = models.FloatField(blank=True, null=True)
    intercept = models.FloatField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_statistics(self, statistics_type: str) -> dict:
        statistics_keys = ("r_2", "mse", "mae", "rmse", "mape")
        statistics = {
            "slope": round(self.slope, 4) if self.slope else None,
            "intercept": round(self.intercept, 4) if self.intercept else None,
        }

        for key in statistics_keys:
            key: str = statistics_type + "_" + key
            statistics[key] = getattr(self, key, None)
            if statistics[key]:
                statistics[key] = round(statistics[key], 4)

        return statistics

    def __str__(self):
        return f"Linear Regression TimeSeries Result {self.data} -> {self.target_column}"
