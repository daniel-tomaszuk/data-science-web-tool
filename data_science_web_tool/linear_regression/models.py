from django.db import models

from preprocessing.models import Data


class LinearRegressionTimeSeriesResult(models.Model):
    data = models.ForeignKey(
        Data,
        on_delete=models.CASCADE,
        related_name="linear_regression_timeseries_results",
    )

    target_column = models.CharField()
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
            "r_2": self.r_2,
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
        }
