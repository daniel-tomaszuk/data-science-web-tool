import base64
import io
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect
from django.views.generic import DetailView
from rest_framework.exceptions import ValidationError
from rest_framework.generics import CreateAPIView

from linear_regression.handlers.time_series import LinearRegressionTimeSeriesHandler
from linear_regression.models import LinearRegressionTimeSeriesResult
from linear_regression.serializers.serializers import LinearRegressionTimeSeriesCreateSerializer
from preprocessing.models import Data


class LinearRegressionView(DetailView):
    template_name = "linear_regression/details.html"
    queryset = Data.objects.prefetch_related(
        "linear_regression_timeseries_results"
    ).all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["statistics"] = self.object.get_statistics()
        context["regression_columns_options"] = [
            column_name
            for column_name, column_type in self.object.data_columns.items()
            if column_type in LinearRegressionTimeSeriesHandler.SUPPORTED_COLUMN_TYPES
        ]
        context["model_types"] = LinearRegressionTimeSeriesResult.MODEL_TYPES_CHOICES

        if linear_regression_result := self.object.linear_regression_timeseries_results.order_by(
            "-created_at"
        ).first():
            context["linear_regression_statistics"] = {
                **linear_regression_result.get_statistics(statistics_type="val"),
                **linear_regression_result.get_statistics(statistics_type="test"),
            }
            context["slope"] = round(linear_regression_result.slope, 4) if linear_regression_result.slope else None
            context["intercept"] = (
                round(linear_regression_result.intercept, 4) if linear_regression_result.intercept else None
            )

            base64_image: str = self._create_regression_plot(
                linear_regression_result=linear_regression_result,
            )
            context["base64_image"] = base64_image
            context["linear_regression_target_column"] = (
                linear_regression_result.target_column
            )
            context["linear_regression_lag"] = linear_regression_result.lag_size
            context["used_model_type"] = linear_regression_result.model_type
            context["max_tree_depth"] = linear_regression_result.max_tree_depth or 1
            context["forecast_horizon"] = linear_regression_result.forecast_horizon or 0
            context["train_percentage"] = linear_regression_result.train_percentage
            context["validation_percentage"] = linear_regression_result.validation_percentage
            context["test_percentage"] = linear_regression_result.test_percentage

        return context

    def _create_regression_plot(
        self, linear_regression_result: LinearRegressionTimeSeriesResult
    ):
        df = self.object.get_df()
        target_column: str = linear_regression_result.target_column

        val_predicted_data = linear_regression_result.val_predictions or []
        test_predicted_data = linear_regression_result.test_predictions or []

        forecast_horizon_data = linear_regression_result.forecast or []
        original_data = df[target_column][-(len(df) - linear_regression_result.lag_size):]

        df["Date"] = pd.to_datetime(df["Date"])
        index = list(df.Date[-(len(df) - linear_regression_result.lag_size):])

        plt.figure(figsize=(12, 8))
        if len(original_data):
            sns.lineplot(
                x=index, y=original_data, label=f"Original data of {target_column}"
            )
        if len(val_predicted_data):
            val_index = list(val_predicted_data.keys())
            val_index = index[int(val_index[0]):int(val_index[-1]) + 1]
            val_values = list(val_predicted_data.values())
            sns.lineplot(
                x=val_index, y=val_values, label=f"Validation Predicted data of {target_column}",
            )

        if len(test_predicted_data):
            test_index = list(test_predicted_data.keys())
            test_index = index[int(test_index[0])::]
            test_values = list(test_predicted_data.values())[0:len(test_index)]
            sns.lineplot(
                x=test_index, y=test_values, label=f"Test Predicted data of {target_column}",
            )

        if len(forecast_horizon_data):
            step = (index[-1] - index[-2]) if len(index) > 1 else pd.Timedelta(days=1)
            forecast_index = [index[-1] + step * (i + 1) for i in range(len(forecast_horizon_data))]
            sns.lineplot(
                x=forecast_index,
                y=forecast_horizon_data,
                label=f"Forecast data of {target_column}",
                linestyle="--",
            )

        if linear_regression_result.slope is not None and linear_regression_result.intercept is not None:
            base_date = index[0]
            base_ordinal = base_date.toordinal()

            ordinal_index = [d.toordinal() - base_ordinal for d in index]
            linear_space = np.linspace(min(ordinal_index), max(ordinal_index), 100)
            y_regression_values = linear_regression_result.slope * linear_space + linear_regression_result.intercept
            date_space = [base_date + timedelta(days=int(d)) for d in linear_space]

            sns.lineplot(x=date_space, y=y_regression_values, label="Linear Regression Line", linestyle=":")

        plt.xticks(rotation=45)
        plt.ylim(top=800)
        plt.gcf().autofmt_xdate()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{base64_img}"


class LinearRegressionTimeSeriesCreateAPIView(CreateAPIView):
    queryset = LinearRegressionTimeSeriesResult.objects.all()
    serializer_class = LinearRegressionTimeSeriesCreateSerializer

    def create(self, request, *args, **kwargs):
        """
        Generate requested plot and return it as an image.
        """
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid(raise_exception=False):
            request.session["validation_errors"] = serializer.errors
            return redirect(
                "linear_regression:linear-regression-details",
                id=request.data["object_id"],
            )

        validated_data = serializer.validated_data
        data_instance: Data = get_object_or_404(Data, pk=validated_data["object_id"])
        target_column: str = validated_data["target_column"]
        column_type: str = data_instance.data_columns.get(target_column)

        handler = LinearRegressionTimeSeriesResult.SUPPORTED_HANDLERS.get(
            validated_data.get("model_type")
        )
        if not handler:
            raise ValidationError("Selected model type is not supported.")

        if not column_type or column_type not in handler.SUPPORTED_COLUMN_TYPES:
            raise ValidationError("Selected target column type is not supported.")

        max_tree_depth = validated_data.get("max_tree_depth")
        lag_size = validated_data.get("lag")
        forecast_horizon = validated_data.get("forecast_horizon")

        train_percentage = validated_data["train_percent"]
        validation_percentage = validated_data["val_percent"]
        test_percentage = validated_data["test_percent"]

        regression_handler = handler(
            data=data_instance,
            column_name=target_column,
            lag_size=lag_size,
            max_tree_depth=max_tree_depth,
            forecast_horizon=forecast_horizon,
            train_percentage=train_percentage,
            validation_percentage=validation_percentage,
            test_percentage=test_percentage,
        )
        model_metadata, forecast = regression_handler.handle()
        LinearRegressionTimeSeriesResult.objects.create(
            data=data_instance,
            target_column=target_column,
            forecast=forecast,
            lag_size=lag_size,
            max_tree_depth=max_tree_depth,
            forecast_horizon=forecast_horizon,
            model_type=validated_data["model_type"],
            train_percentage=train_percentage,
            validation_percentage=validation_percentage,
            test_percentage=test_percentage,
            val_predictions=dict(model_metadata["val_predictions"]),
            test_predictions=dict(model_metadata["test_predictions"]),
            slope=model_metadata["slope"],
            intercept=model_metadata["intercept"],
            **model_metadata["val_statistics"],
            **model_metadata["test_statistics"],
        )
        return redirect(
            "linear_regression:linear-regression-details", pk=data_instance.id
        )
