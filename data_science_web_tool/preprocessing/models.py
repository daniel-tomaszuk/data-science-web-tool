import base64
import io
import json
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from django.db import models
from django.urls import reverse
from django.utils.html import format_html
from preprocessing.data_sources_handlers.csv_source_handler import CsvDataSourceHandler


class Data(models.Model):
    class Meta:
        verbose_name_plural = "data"

    name = models.CharField(max_length=256)
    description = models.TextField(null=True, blank=True)

    data = models.JSONField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_admin_change_url(self) -> str:
        url = reverse("admin:preprocessing_data_change", args=[self.pk])
        return format_html('<a href="{}">{}</a>', url, self.name)

    def get_statistics(self):
        s = time.perf_counter()
        print("START", s)
        df: pd.DataFrame = pd.DataFrame(self.data)
        if df.empty:
            return {}
        e = time.perf_counter()
        print("END", e)
        print("load time", e - s)

        # Capture info() output as a string
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_text = buffer.getvalue().replace("\n", "<br>")

        return {
            "describe": df.describe().T.to_html(),
            "head": df.head(10).to_html(),
            "info": info_text,
        }

    def get_histogram(self, column_name: str):
        df: pd.DataFrame = pd.DataFrame(self.data)
        numeric_cols = df.select_dtypes(include=["number"]).columns
        selected_column = numeric_cols[-1] if not numeric_cols.empty else None

        selected_column = df.Temperature
        if not selected_column:
            return

        plt.figure(figsize=(8, 5))
        sns.histplot(df[selected_column], bins=30, kde=True)
        plt.xlabel(selected_column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {selected_column}")
        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()

        return f"data:image/png;base64,{base64_img}", selected_column

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
