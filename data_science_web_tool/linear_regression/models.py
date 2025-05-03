from django.db import models

from preprocessing.models import Data


class LinearRegressionTimeSeriesResult(models.Model):
    data = models.ForeignKey(
        Data,
        on_delete=models.CASCADE,
        related_name="linear_regression_timeseries_results",
    )

    target_column = models.CharField()
    lag_size = models.IntegerField(blank=True, null=True)
    predictions = models.JSONField(blank=True, null=True)

    r_2 = models.FloatField(blank=True, null=True, help_text="R2 score")
    mse = models.FloatField(blank=True, null=True, help_text="Mean squared error")
    mae = models.FloatField(blank=True, null=True, help_text="Mean absolute error")
    rmse = models.FloatField(blank=True, null=True, help_text="Root mean square error")
    mape = models.FloatField(
        blank=True,
        null=True,
        help_text="Mean absolute percentage error",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_statistics(self) -> dict:
        return {
            "r_2": round(self.r_2, 4),
            "mse": round(self.mse, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4),
        }

    def __str__(self):
        return (
            f"Linear Regression TimeSeries Result {self.data} -> {self.target_column}"
        )
