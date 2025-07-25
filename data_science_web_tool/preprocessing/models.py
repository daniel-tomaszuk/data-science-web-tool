import io

import pandas as pd
from django.db import models
from django.urls import reverse
from django.utils.html import format_html

from preprocessing.data_sources_handlers.csv_source_handler import CsvDataSourceHandler
from preprocessing.data_sources_handlers.yfinance_source_handler import YFinanceDataSourceHandler
from preprocessing.plot_handlers.histogram import HistogramPlotHandler
from preprocessing.plot_handlers.line_plot import LinePlotHandler
from preprocessing.statistics_tests_handlers.adf_test import ADFTestHandler


class Data(models.Model):
    NUMERICAL_TYPES = (
        "int64",
        "float64",
    )
    SUPPORTED_COLUMN_TYPES = (
        "int64",
        "float64",
        "object",
        "bool",
        "datetime64[ns]",
        "category",
    )
    SUPPORTED_PLOT_TYPES = {
        "sns.lineplot": LinePlotHandler,
        "sns.histplot": HistogramPlotHandler,
    }

    class Meta:
        verbose_name_plural = "data"

    name = models.CharField(max_length=256, help_text="Name of the data source.")
    description = models.TextField(null=True, blank=True, help_text="Additional information about the data source.")
    data = models.JSONField(null=True, blank=True)
    data_index = models.JSONField(
        null=True,
        blank=True,
        default=list,
        help_text="DataFrame indexes defined by the user. Must match possible column names.",
    )
    data_columns = models.JSONField(
        null=True,
        blank=True,
        default=dict,
        help_text="Found DataFrame columns.",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.data_columns:
            df: pd.DataFrame = self.get_df()
            self.data_columns = {str(key): str(value) for key, value in df.dtypes.to_dict().items()}
        super().save(*args, **kwargs)

    def get_admin_change_url(self) -> str:
        url = reverse("admin:preprocessing_data_change", args=[self.pk])
        return format_html('<a href="{}">{}</a>', url, self.name)

    def get_statistics(self):
        df: pd.DataFrame = self.get_df()
        if df.empty:
            return {}

        # Capture info() output as a string
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue().replace("\n", "<br>")
        return {
            "describe": df.describe().T.to_html(
                classes="table table-bordered table-sm",
            ),
            "head": df.head(5).to_html(classes="table table-bordered table-sm"),
            "info": info_text,
        }

    def get_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        df = df.astype(self.data_columns)
        if self.data_index:
            df.set_index(self.data_index, inplace=True)
            df = df.sort_index()
        return df

    def __str__(self) -> str:
        return f"Data `{self.name}` created at {self.created_at}"


class DataUpload(models.Model):
    FILE_TYPE_CSV = "csv"
    YFINANACE = "yfinance"
    DATA_SOURCE_CHOICES = (
        (FILE_TYPE_CSV, FILE_TYPE_CSV),
        (YFINANACE, YFINANACE),
    )
    DATA_SOURCE_TYPE_PROCESSORS = {
        "csv": CsvDataSourceHandler,
        "yfinance": YFinanceDataSourceHandler,
    }

    file_name = models.CharField(max_length=512)
    file_type = models.CharField(max_length=32, choices=DATA_SOURCE_CHOICES)
    description = models.TextField(null=True, blank=True)

    file = models.FileField(upload_to="uploads/")
    data = models.OneToOneField(
        Data,
        related_name="data_upload",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if self.file and not self.file_name:
            self.file_name = self.file.name

        if self.file and self.data is None:
            file_handler = self.DATA_SOURCE_TYPE_PROCESSORS.get(self.file_type)
            if not file_handler:
                raise TypeError(f"File type '{self.file_type}' is not supported")

            data = file_handler(self.file).load_data()
            self.data = Data.objects.create(
                name=self.file_name,
                description=self.description,
                data=data,
            )
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Data {self.file} uploaded at {self.created_at.isoformat()}"


class DataTestResult(models.Model):
    SUPPORTED_TEST_HANDLERS = {
        "adf": ADFTestHandler,
    }

    data = models.ForeignKey(Data, on_delete=models.CASCADE)
    results = models.JSONField(null=True, blank=True)

    test_type = models.CharField(null=True, blank=True)
    target_column = models.CharField(null=True, blank=True)
    max_augmentation_count = models.IntegerField(null=True, blank=True)
    test_version = models.CharField(null=True, blank=True)
    differentiate_count = models.IntegerField(null=True, blank=True)

    raw_data_pp_test_results = models.JSONField(null=True, blank=True)
    raw_data_kpss_test_results = models.JSONField(null=True, blank=True)
    diff_data_pp_test_results = models.JSONField(null=True, blank=True)
    diff_data_kpss_test_results = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
