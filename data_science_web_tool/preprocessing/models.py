import base64
import io
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from django.db import models
from django.urls import reverse
from django.utils.html import format_html
from preprocessing.data_sources_handlers.csv_source_handler import CsvDataSourceHandler


class Data(models.Model):
    SUPPORTED_COLUMN_TYPES = (
        "int64",
        "float64",
        "object",
        "bool",
        "datetime64[ns]",
        "categorical",
    )
    SUPPORTED_PLOT_TYPES = "sns.lineplot"

    class Meta:
        verbose_name_plural = "data"

    name = models.CharField(max_length=256, help_text="Name of the data source.")
    description = models.TextField(
        null=True, blank=True, help_text="Additional information about the data source."
    )
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
        df: pd.DataFrame = self.get_df()
        self.data_columns = {
            str(key): str(value) for key, value in df.dtypes.to_dict().items()
        }
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
            "head": df.head(10).to_html(classes="table table-bordered table-sm"),
            "info": info_text,
        }

    def get_histogram(self, column_name: str):
        s = time.perf_counter()
        if not column_name:
            return

        df: pd.DataFrame = pd.DataFrame(self.data)

        df["Date"] = pd.to_datetime(df["Date"])
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
        df.set_index("Date", inplace=True)
        df = df.sort_index()

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df[column_name])
        plt.xlabel("Date")
        plt.ylabel(column_name)
        plt.title(f"Histogram of {column_name}")
        plt.grid(True)

        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()
        e = time.perf_counter()

        print("IMAGE TIME", e - s)
        return f"data:image/png;base64,{base64_img}", column_name

    def get_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        df = df.astype(self.data_columns)
        return df

    def __str__(self) -> str:
        return f"Data `{self.name}` created at {self.created_at}"


class DataUpload(models.Model):
    FILE_TYPE_CSV = "csv"
    FILE_TYPE_CHOICES = ((FILE_TYPE_CSV, FILE_TYPE_CSV),)

    FILE_TYPE_PROCESSORS = {
        "csv": CsvDataSourceHandler,
    }

    file_name = models.CharField(max_length=512)
    file_type = models.CharField(max_length=32, choices=FILE_TYPE_CHOICES)
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
        self.file_name = self.file.name
        if self.file and self.data is None:
            file_handler = self.FILE_TYPE_PROCESSORS.get(self.file_type)
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


class DataStatistics(models.Model):
    data = models.ForeignKey(
        Data,
        null=True,
        blank=True,
        related_name="statistics",
        on_delete=models.CASCADE,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
