from django.db import models
from garch.handlers.garch_time_series import ArchTimeSeriesHandler
from garch.handlers.garch_time_series import GarchTimeSeriesHandler

from preprocessing.models import Data


class GarchResult(models.Model):
    ARCH_MODEL = "arch"
    GARCH_MODEL = "garch"

    SUPPORTED_HANDLERS = {
        ARCH_MODEL: ArchTimeSeriesHandler,
        GARCH_MODEL: GarchTimeSeriesHandler,
    }
    MODEL_TYPES_CHOICES = tuple(
        (model_type, model_type.replace("_", " ").title()) for model_type in SUPPORTED_HANDLERS.keys()
    )
    data = models.ForeignKey(
        Data,
        on_delete=models.CASCADE,
        related_name="garch_results",
    )

    model_type = models.CharField()
    summary = models.CharField(blank=True, null=True)
    target_column = models.CharField()
    forecast_horizon = models.IntegerField(blank=True, null=True)

    forecast = models.JSONField(blank=True, null=True)
    train_percentage = models.FloatField(blank=True, null=True)
    validation_percentage = models.FloatField(blank=True, null=True)
    test_percentage = models.FloatField(blank=True, null=True)

    raw_data_engle_arch_test_results = models.JSONField(blank=True, null=True)
    model_fit_engle_arch_test_results = models.JSONField(blank=True, null=True)

    raw_data_ljung_box_test_results = models.JSONField(blank=True, null=True)
    raw_data_ljung_box_test_results_squared = models.JSONField(blank=True, null=True)
    model_fit_ljung_box_test_results = models.JSONField(blank=True, null=True)
    model_fit_ljung_box_test_results_squared = models.JSONField(blank=True, null=True)

    p_mean_equation_lags = models.IntegerField(default=0)
    q_variance_equation_lags = models.IntegerField(null=True, blank=True)

    model_result_resid = models.JSONField(blank=True, null=True)
    model_result_conditional_volatility = models.JSONField(blank=True, null=True)

    acf_lags = models.IntegerField(default=36)
    tests_lags = models.IntegerField(default=5)

    # Validation Set Statistics
    val_vol_mse = models.FloatField(
        null=True,
        blank=True,
        help_text="Validation set: Mean Squared Error between predicted variance and squared returns.",
    )
    val_vol_mae = models.FloatField(
        null=True,
        blank=True,
        help_text="Validation set: Mean Absolute Error between predicted variance and squared returns.",
    )
    val_vol_rmse = models.FloatField(
        null=True,
        blank=True,
        help_text="Validation set: Root Mean Squared Error between predicted variance and squared returns.",
    )
    val_vol_qlike = models.FloatField(
        null=True,
        blank=True,
        help_text="Validation set: QLIKE loss between predicted variance and squared returns (lower is better).",
    )
    val_mean_r2 = models.FloatField(
        null=True, blank=True, help_text="Validation set: R-squared score for predicted vs. actual returns."
    )
    val_mean_mse = models.FloatField(
        null=True, blank=True, help_text="Validation set: Mean Squared Error for predicted vs. actual returns."
    )
    val_mean_mae = models.FloatField(
        null=True, blank=True, help_text="Validation set: Mean Absolute Error for predicted vs. actual returns."
    )
    val_mean_rmse = models.FloatField(
        null=True, blank=True, help_text="Validation set: Root Mean Squared Error for predicted vs. actual returns."
    )
    val_mean_mape = models.FloatField(
        null=True,
        blank=True,
        help_text="Validation set: Mean Absolute Percentage Error for predicted vs. actual returns.",
    )

    # Test Set Statistics
    test_vol_mse = models.FloatField(
        null=True, blank=True, help_text="Test set: Mean Squared Error between predicted variance and squared returns."
    )
    test_vol_mae = models.FloatField(
        null=True, blank=True, help_text="Test set: Mean Absolute Error between predicted variance and squared returns."
    )
    test_vol_rmse = models.FloatField(
        null=True,
        blank=True,
        help_text="Test set: Root Mean Squared Error between predicted variance and squared returns.",
    )
    test_vol_qlike = models.FloatField(
        null=True,
        blank=True,
        help_text="Test set: QLIKE loss between predicted variance and squared returns (lower is better).",
    )
    test_mean_r2 = models.FloatField(
        null=True, blank=True, help_text="Test set: R-squared score for predicted vs. actual returns."
    )
    test_mean_mse = models.FloatField(
        null=True, blank=True, help_text="Test set: Mean Squared Error for predicted vs. actual returns."
    )
    test_mean_mae = models.FloatField(
        null=True, blank=True, help_text="Test set: Mean Absolute Error for predicted vs. actual returns."
    )
    test_mean_rmse = models.FloatField(
        null=True, blank=True, help_text="Test set: Root Mean Squared Error for predicted vs. actual returns."
    )
    test_mean_mape = models.FloatField(
        null=True, blank=True, help_text="Test set: Mean Absolute Percentage Error for predicted vs. actual returns."
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_statistics(self, statistics_type: str) -> dict:
        statistics_keys = ("r_2", "mse", "mae", "rmse", "mape")
        statistics = {}
        for key in statistics_keys:
            key: str = statistics_type + "_" + key
            statistics[key] = getattr(self, key, None)
            if statistics[key]:
                statistics[key] = round(statistics[key], 4)

        return statistics

    def __str__(self):
        return f"Linear Regression TimeSeries Result {self.data} -> {self.target_column}"
