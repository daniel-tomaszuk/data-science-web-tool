from typing import Callable

import pandas as pd
from arch.unitroot import PhillipsPerron
from django.contrib import messages
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect
from django.views.generic import DetailView
from rest_framework.exceptions import ValidationError
from rest_framework.generics import CreateAPIView
from statsmodels.tsa.stattools import kpss

from preprocessing.models import Data
from preprocessing.models import DataTestResult
from preprocessing.serializers.serializers import CreateStationaryTestRequestSerializer
from preprocessing.serializers.serializers import ImageRetrieveSerializer


class DataDetailView(DetailView):
    model = Data
    template_name = "preprocessing/data/details.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["statistics"] = self.object.get_statistics()
        context["preprocessing_plot_image"] = self.request.session.pop("preprocessing_plot_image", None)
        context["preprocessing_plot_axis_x_name"] = self.request.session.pop("preprocessing_plot_axis_x_name", None)
        context["preprocessing_plot_axis_y_name"] = self.request.session.pop("preprocessing_plot_axis_y_name", None)
        context["preprocessing_group_by_name"] = self.request.session.pop("preprocessing_group_by_name", None)
        context["preprocessing_plot_type"] = self.request.session.pop("preprocessing_plot_type", None)
        context["image_validation_errors"] = self.request.session.pop("image_validation_errors", None)
        context["is_group_by_disabled"] = not bool(
            any(value_type == "category" for value_type in self.object.data_columns.values())
        )
        context["numerical_columns"] = [
            column_name
            for column_name, column_type in self.object.data_columns.items()
            if column_type in Data.NUMERICAL_TYPES
        ]

        last_adf_test = DataTestResult.objects.filter(test_type="adf").order_by("-created_at").first()
        if not last_adf_test:
            return context

        results = last_adf_test.results
        context["last_adf_test_results"] = results if last_adf_test else None
        context["last_adf_test_target_column"] = last_adf_test.target_column
        context["last_adf_test_results_indices"] = list(list(results.values())[0].keys())
        context["last_adf_test_results_max_augmentation_count"] = last_adf_test.max_augmentation_count
        context["last_adf_test_datetime"] = last_adf_test.created_at if last_adf_test else None
        context["last_adf_test_results_differentiate_count"] = last_adf_test.differentiate_count
        context["last_adf_test_results_test_version"] = last_adf_test.test_version
        context["pp_test_results"] = last_adf_test.pp_test_results
        context["kpss_test_results"] = last_adf_test.kpss_test_results

        highlight_indices = []
        for idx, value in results["ADF Test statistic"].items():
            pval_5 = results["Test BG (5 lags) (p-value)"][idx]
            pval_10 = results["Test BG (10 lags) (p-value)"][idx]
            pval_15 = results["Test BG (15 lags) (p-value)"][idx]
            if pval_5 > 0.05 and pval_10 > 0.05 and pval_15 > 0.05:
                highlight_indices.append(idx)

        context["last_adf_test_results_highlight_indices"] = highlight_indices
        return context


class CreateStationaryTestResultsAPIView(CreateAPIView):
    model = Data
    serializer_class = CreateStationaryTestRequestSerializer

    def create(self, request, *args, **kwargs):
        instance: Data = get_object_or_404(Data, id=self.kwargs.get("pk", ""))
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid(raise_exception=False):
            request.session["image_validation_errors"] = serializer.errors
            return redirect("preprocessing:data-detail", pk=instance.id)

        validated_data = serializer.validated_data
        test_handler = DataTestResult.SUPPORTED_TEST_HANDLERS.get(validated_data["test_type"])
        df: pd.DataFrame = instance.get_df()

        time_series = df[validated_data["target_column"]]
        test_handler_instance = test_handler(
            series=time_series.copy(),
            max_aug=validated_data["max_augmentation_count"],
            version=validated_data["test_version"],
            differentiate_count=validated_data["differentiate_count"],
        )
        adf_test_results: pd.DataFrame = test_handler_instance.run()

        pp_test = PhillipsPerron(time_series.copy(), trend="n")
        kpss_stat, kpss_p_value, _, _ = kpss(time_series.copy(), regression="c")

        DataTestResult.objects.create(
            data=instance,
            results=adf_test_results.to_dict(),
            pp_test_results={
                "test_statistic": pp_test.stat,
                "p_value": pp_test.pvalue,
                "summary": str(pp_test.summary()),
            },
            kpss_test_results={"test_statistic": kpss_stat, "p_value": kpss_p_value},
            **validated_data,
        )

        return redirect("preprocessing:data-detail", pk=instance.id)


class ChangeColumnTypeCreateAPIView(CreateAPIView):
    queryset = Data.objects.all()
    serializer_class = None

    def create(self, request, *args, **kwargs):
        """
        Tries to change column type of selected data frame.
        Returns HTTP 400 Bad Request if operation fails.
        """
        instance: Data = get_object_or_404(Data, id=self.kwargs.get("pk", ""))
        column_types = {}
        current_columns = list(instance.data_columns.keys())
        for key, value in request.POST.items():
            if not key.startswith("data_type_"):
                continue

            new_type = request.POST.get(key, "")
            if not new_type or new_type not in Data.SUPPORTED_COLUMN_TYPES:
                messages.error(request, "Invalid column type.")
                return redirect("preprocessing:data-detail", pk=instance.id)

            column_name = key.replace("data_type_", "")
            if column_name not in current_columns:
                messages.error(request, "Invalid column name.")
                return redirect("preprocessing:data-detail", pk=instance.id)

            column_types[column_name] = new_type

        instance.data_columns = column_types
        is_success = True
        try:
            instance.save()
        except Exception as e:
            is_success = False
            messages.error(request, str(e))

        if is_success:
            messages.success(request, "Column types updated successfully!")

        return redirect("preprocessing:data-detail", pk=instance.id)


class ImageCreateAPIView(CreateAPIView):
    queryset = Data.objects.all()
    serializer_class = ImageRetrieveSerializer

    def create(self, request, *args, **kwargs):
        """
        Generate requested plot and return it as an image.
        """
        instance: Data = get_object_or_404(Data, id=self.kwargs.get("pk", ""))
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid(raise_exception=False):
            request.session["image_validation_errors"] = serializer.errors
            return redirect("preprocessing:data-detail", pk=instance.id)

        validated_data = serializer.validated_data

        plot_handler: Callable = instance.SUPPORTED_PLOT_TYPES.get(validated_data["plot_type"])
        if not plot_handler:
            raise ValidationError("Invalid plot type.")

        image: str = plot_handler(
            data=instance,
            axis_x_name=validated_data["axis_x_name"],
            axis_y_name=validated_data["axis_y_name"],
            group_by_name=validated_data.get("group_by_name"),
        ).create_image()
        request.session["preprocessing_plot_image"] = image
        request.session["preprocessing_plot_axis_x_name"] = validated_data["axis_x_name"].lower()
        request.session["preprocessing_plot_axis_y_name"] = validated_data["axis_y_name"].lower()
        request.session["preprocessing_plot_type"] = validated_data["plot_type"]
        request.session["preprocessing_group_by_name"] = validated_data.get("group_by_name", "")
        return redirect("preprocessing:data-detail", pk=instance.id)
