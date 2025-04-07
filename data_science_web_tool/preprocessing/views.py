from typing import Callable

from django.contrib import messages
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect
from django.views.generic import DetailView
from preprocessing.models import Data
from preprocessing.serializers.preprocessing import ImageRetrieveSerializer
from rest_framework.exceptions import ValidationError
from rest_framework.generics import CreateAPIView


class DataDetailView(DetailView):
    model = Data
    template_name = "preprocessing/data/details.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["statistics"] = self.object.get_statistics()
        context["preprocessing_plot_image"] = self.request.session.pop(
            "preprocessing_plot_image", None
        )
        context["preprocessing_plot_axis_x_name"] = self.request.session.pop(
            "preprocessing_plot_axis_x_name", None
        )
        context["preprocessing_plot_axis_y_name"] = self.request.session.pop(
            "preprocessing_plot_axis_y_name", None
        )
        context["preprocessing_group_by_name"] = self.request.session.pop(
            "preprocessing_group_by_name", None
        )
        context["preprocessing_plot_type"] = self.request.session.pop(
            "preprocessing_plot_type", None
        )
        context["image_validation_errors"] = self.request.session.pop(
            "image_validation_errors", None
        )
        return context


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

        plot_handler: Callable = instance.SUPPORTED_PLOT_TYPES.get(
            validated_data["plot_type"]
        )
        if not plot_handler:
            raise ValidationError("Invalid plot type.")

        image: str = plot_handler(
            data=instance,
            axis_x_name=validated_data["axis_x_name"],
            axis_y_name=validated_data["axis_y_name"],
            group_by_name=validated_data["group_by_name"],
        ).create_image()
        request.session["preprocessing_plot_image"] = image
        request.session["preprocessing_plot_axis_x_name"] = validated_data[
            "axis_x_name"
        ].lower()
        request.session["preprocessing_plot_axis_y_name"] = validated_data[
            "axis_y_name"
        ].lower()
        request.session["preprocessing_plot_type"] = validated_data["plot_type"]
        request.session["preprocessing_group_by_name"] = validated_data["group_by_name"]
        return redirect("preprocessing:data-detail", pk=instance.id)
