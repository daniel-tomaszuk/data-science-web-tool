import itertools
import json

from django import forms
from django.contrib import admin
from django.core.files.uploadedfile import TemporaryUploadedFile
from preprocessing.models import Data
from preprocessing.models import DataUpload


class FileUploadForm(forms.ModelForm):
    class Meta:
        model = DataUpload
        fields = [
            "file_name",
            "file_type",
            "description",
            "file",
        ]

    def clean_file(self) -> TemporaryUploadedFile:
        file: TemporaryUploadedFile = self.cleaned_data["file"]
        if not any(
            file.name.endswith(f".{supported_extension}")
            for supported_extension in list(DataUpload.FILE_TYPE_PROCESSORS.keys())
        ):
            raise forms.ValidationError("Uploaded file type is not supported.")
        return file


@admin.register(DataUpload)
class DataUploadAdmin(admin.ModelAdmin):
    """
    ModelAdmin for DataUpload model.
    """

    form = FileUploadForm
    list_display = ("file_name", "file_type", "description", "file")
    readonly_fields = ("file_name", "created_at", "updated_at")


@admin.register(Data)
class DataAdmin(admin.ModelAdmin):
    """
    ModelAdmin page for Data model.
    """

    DATA_FIELD_PREVIEW_CHARS_COUNT = 500
    list_display = (
        "name",
        "updated_at",
        "created_at",
    )
    fields = (
        "description",
        "data_index",
        "data_columns",
        "short_preview",
        "data_upload",
        "created_at",
        "updated_at",
    )
    readonly_fields = (
        "name",
        "short_preview",
        "data_upload",
        "created_at",
        "updated_at",
    )

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related("data_upload")

    def data_upload(self, obj: Data) -> DataUpload:
        return obj.data_upload

    def short_preview(self, obj: Data):
        if obj.data:
            json_iter = json.JSONEncoder().iterencode(obj.data)
            preview_str = "".join(
                itertools.islice(json_iter, self.DATA_FIELD_PREVIEW_CHARS_COUNT)
            )
            return preview_str
        return "No Data"


class DataStatisticsAdmin(admin.ModelAdmin):
    """
    ModelAdmin page for DataStatistics.
    """
