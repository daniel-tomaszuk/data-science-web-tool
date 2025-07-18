import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect
from django.views.generic import DetailView
from rest_framework.exceptions import ValidationError
from rest_framework.generics import CreateAPIView
from statsmodels.tsa.stattools import acf

from garch.handlers.garch_time_series import GarchTimeSeriesHandler
from garch.models import GarchResult
from garch.serializers.serializers import GarchResultCreateSerializer
from preprocessing.models import Data


class GarchResultView(DetailView):
    template_name = "garch/details.html"
    queryset = Data.objects.prefetch_related("garch_results").all()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["statistics"] = self.object.get_statistics()
        context["columns_options"] = [
            column_name
            for column_name, column_type in self.object.data_columns.items()
            if column_type in GarchTimeSeriesHandler.SUPPORTED_COLUMN_TYPES
        ]
        context["model_types"] = GarchResult.MODEL_TYPES_CHOICES
        context["validation_errors"] = self.request.session.pop("validation_errors", {})

        garch_result = self.object.garch_results.order_by("-created_at").first()
        if garch_result:
            df = self.object.get_df()

            log_diff_column_name = f"{garch_result.target_column}_log_diff"
            df[log_diff_column_name] = np.log(df[garch_result.target_column]).diff()
            df.dropna(inplace=True)
            df.reset_index(inplace=True)

            context["target_column"] = garch_result.target_column
            context["used_model_type"] = garch_result.model_type
            context["used_acf_lags"] = garch_result.acf_lags
            context["used_tests_lags"] = garch_result.tests_lags
            context["time_series_with_log_diffs_img"] = self._create_time_series_plot_with_log_diffs(
                df=df,
                target_column=garch_result.target_column,
            )
            context["time_series_acf_log_diffs_img"] = self._create_acf_plot(
                df=df,
                target_column=log_diff_column_name,
                nlags=garch_result.acf_lags,
            )
            context["time_series_acf_log_diffs_img_squared"] = self._create_acf_plot(
                df=df,
                target_column=log_diff_column_name,
                nlags=garch_result.acf_lags,
                squared=True,
            )
            context["time_series_acf_log_diffs_img_model_result"] = self._create_acf_plot(
                df=df,
                model_result_resid=garch_result.model_result_resid,
                model_result_conditional_volatility=garch_result.model_result_conditional_volatility,
                target_column=log_diff_column_name,
                nlags=garch_result.acf_lags,
            )
            context["time_series_acf_log_diffs_img_squared_model_result"] = self._create_acf_plot(
                df=df,
                model_result_resid=garch_result.model_result_resid,
                model_result_conditional_volatility=garch_result.model_result_conditional_volatility,
                target_column=log_diff_column_name,
                nlags=garch_result.acf_lags,
                squared=True,
            )

            context["summary"] = garch_result.summary

            self._set_test_results(garch_result, context)
            context["p_mean_equation_lags"] = garch_result.p_mean_equation_lags
            context["q_variance_equation_lags"] = garch_result.q_variance_equation_lags
            context["forecast_horizon"] = garch_result.forecast_horizon
            context["train_percentage"] = garch_result.train_percentage
            context["validation_percentage"] = garch_result.validation_percentage
            context["test_percentage"] = garch_result.test_percentage
            context["garch_statistics"] = {
                "val_vol_mse": garch_result.val_vol_mse,
                "val_vol_mae": garch_result.val_vol_mae,
                "val_vol_rmse": garch_result.val_vol_rmse,
                "val_vol_qlike": garch_result.val_vol_qlike,
                "val_mean_r2": garch_result.val_mean_r2,
                "val_mean_mse": garch_result.val_mean_mse,
                "val_mean_mae": garch_result.val_mean_mae,
                "val_mean_rmse": garch_result.val_mean_rmse,
                "val_mean_mape": garch_result.val_mean_mape * 100,
                "test_vol_mse": garch_result.test_vol_mse,
                "test_vol_mae": garch_result.test_vol_mae,
                "test_vol_rmse": garch_result.test_vol_rmse,
                "test_vol_qlike": garch_result.test_vol_qlike,
                "test_mean_r2": garch_result.test_mean_r2,
                "test_mean_mse": garch_result.test_mean_mse,
                "test_mean_mae": garch_result.test_mean_mae,
                "test_mean_rmse": garch_result.test_mean_rmse,
                "test_mean_mape": garch_result.test_mean_mape * 100,
            }
            if garch_result.forecast_horizon > 0:
                context["forecast_plot"], context["forecast_plot_zoomed"] = self._create_forecast_plots(
                    df=df,
                    result=garch_result,
                )

        return context

    def _set_test_results(self, garch_result, context: dict):
        # Engle Arch Test
        context["raw_data_engle_arch_test_results"] = self._get_rounded_engle_test_results(
            garch_result.raw_data_engle_arch_test_results
        )
        context["raw_data_engle_arch_test_results_success"] = (
            garch_result.raw_data_engle_arch_test_results.get("p_value", 0) > 0.05
        )
        context["model_fit_engle_arch_test_results"] = self._get_rounded_engle_test_results(
            garch_result.model_fit_engle_arch_test_results
        )
        context["model_fit_engle_arch_test_results_success"] = (
            garch_result.model_fit_engle_arch_test_results.get("p_value", 0) > 0.05
        )

        # Ljung Box Test
        context["raw_data_ljung_box_test_results"] = self._get_rounded_ljung_box_test_results(
            garch_result.raw_data_ljung_box_test_results
        )
        context["raw_data_ljung_box_test_results_success"] = all(
            p_value > 0.05 for p_value in garch_result.raw_data_ljung_box_test_results.get("lb_pvalue", [])
        )
        context["raw_data_ljung_box_test_results_squared"] = self._get_rounded_ljung_box_test_results(
            garch_result.raw_data_ljung_box_test_results_squared
        )
        context["raw_data_ljung_box_test_results_squared_success"] = all(
            p_value > 0.05 for p_value in garch_result.raw_data_ljung_box_test_results_squared.get("lb_pvalue", [])
        )
        context["model_fit_ljung_box_test_results"] = self._get_rounded_ljung_box_test_results(
            garch_result.model_fit_ljung_box_test_results
        )
        context["model_fit_ljung_box_test_results_success"] = all(
            p_value > 0.05 for p_value in garch_result.model_fit_ljung_box_test_results.get("lb_pvalue", [])
        )
        context["model_fit_ljung_box_test_results_squared"] = self._get_rounded_ljung_box_test_results(
            garch_result.model_fit_ljung_box_test_results_squared
        )
        context["model_fit_ljung_box_test_results_squared_success"] = all(
            p_value > 0.05 for p_value in garch_result.model_fit_ljung_box_test_results_squared.get("lb_pvalue", [])
        )

    def _create_time_series_plot_with_log_diffs(self, df: pd.DataFrame, target_column: str) -> str:
        df["Date"] = pd.to_datetime(df["Date"])
        index = list(df.Date)

        # Create figure and main axis
        fig, ax1 = plt.subplots(figsize=(12, 4))

        # Plot the original values on the left axis
        sns.lineplot(
            x=index,
            y=list(df[target_column]),
            ax=ax1,
            label=f"Original values for {target_column}",
            color="blue",
        )
        ax1.set_ylabel(f"Original Values {target_column}", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(
            loc="upper left",
            bbox_to_anchor=(0, 1.12),
        )

        # Create a second y-axis for the log diffs
        ax2 = ax1.twinx()
        sns.lineplot(
            x=index,
            y=list(df[f"{target_column}_log_diff"]),
            ax=ax2,
            label=f"Logarithmic differences for {target_column}",
            color="red",
            alpha=0.5,
            linewidth=1.5,
        )
        ax2.set_ylabel(f"Log Diff (Returns) {target_column}", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(
            loc="upper right",
            bbox_to_anchor=(1, 1.12),
        )

        # Format x-axis
        ax1.set_xlabel("Date")
        plt.xticks(rotation=45)
        fig.autofmt_xdate()
        ax1.grid(True)
        fig.tight_layout()

        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{base64_img}"

    def _create_acf_plot(
        self,
        df: pd.DataFrame,
        model_result_resid: dict | None = None,
        model_result_conditional_volatility: dict | None = None,
        target_column: str = "",
        nlags: int = 36,
        squared: bool = False,
    ) -> str:
        if model_result_resid is None:
            df[target_column].dropna(inplace=True)
            if squared:
                df[f"{target_column}_squared"] = df[target_column] ** 2
                target_column = f"{target_column}_squared"

            acf_values = acf(df[target_column], nlags=nlags)
        else:
            acf_values: pd.Series = pd.Series(model_result_resid) / pd.Series(model_result_conditional_volatility)
            acf_values.dropna(inplace=True)
            if squared:
                acf_values = acf_values**2
            acf_values = acf(acf_values, nlags=nlags)

        plt.figure(figsize=(12, 4))
        plt.stem(range(len(acf_values)), acf_values)
        plt.axhline(y=0, linestyle="-", color="black")
        plt.axhline(y=-1.96 / np.sqrt(len(df[target_column])), linestyle="--", color="gray")
        plt.axhline(y=1.96 / np.sqrt(len(df[target_column])), linestyle="--", color="gray")
        plt.ylim(-0.4, 0.4)

        plt.xlabel("Lag (periods)")
        plt.ylabel("Autocorrelation")
        if squared:
            plt.ylabel("Autocorrelation (squared series)")

        title = "ACF for Log Diffs"
        if squared:
            title += " Squared"

        title += ", 95% confidence interval"
        plt.title(title)

        plt.grid(True)
        plt.tight_layout()

        title = "ACF for Log Diffs"
        if squared:
            title += " Squared"

        title += ", 95% confidence interval"

        plt.title(title)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()
        return f"data:image/png;base64,{base64_img}"

    def _create_forecast_plots(
        self,
        df: pd.DataFrame,
        result: GarchResult,
    ) -> tuple[str, str]:
        forecast_data: dict = result.forecast
        if not forecast_data:
            return "", ""

        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        index = list(df.Date)

        forecast_data["forecast_std"] = np.sqrt(forecast_data["forecast_variance"])
        forecast_index = forecast_data.pop("forecast_index")

        step = (index[-1] - index[-2]) if len(index) > 1 else pd.Timedelta(days=1)
        for i in range(1, min(5, len(index))):
            step = min(step, (index[-i] - index[-i - 1]))

        forecast_index = [index[-1] + step * i for i in range(len(forecast_index))]
        forecast_df: pd.DataFrame = pd.DataFrame(forecast_data, index=forecast_index)

        # +- 2 std visualization
        forecast_ci_upper = forecast_df["forecast_means"] + 2 * forecast_df["forecast_std"]
        forecast_ci_lower = forecast_df["forecast_means"] - 2 * forecast_df["forecast_std"]

        plt.figure(figsize=(12, 4))
        plt.plot(
            index,
            df[[f"{result.target_column}_log_diff"]],
            label="Actual returns",
            color="black",
        )
        plt.plot(
            forecast_index,
            forecast_df.forecast_means,
            label=f"Forecast {result.model_type.upper()}({result.p_mean_equation_lags},{result.q_variance_equation_lags})",
            color="blue",
            linestyle="--",
        )

        plt.fill_between(
            forecast_index, forecast_ci_lower, forecast_ci_upper, color="blue", alpha=0.2, label="95% CI forecast"
        )

        plt.title(
            f"Forecast {result.model_type.upper()}({result.p_mean_equation_lags},{result.q_variance_equation_lags})"
        )
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()
        main_plot: str = f"data:image/png;base64,{base64_img}"

        zoom_start_idx = -len(forecast_df) - int(0.25 * result.forecast_horizon)
        zoom_returns = df.iloc[zoom_start_idx:]

        plt.figure(figsize=(12, 4))
        plt.plot(
            index[zoom_start_idx:],
            zoom_returns[[f"{result.target_column}_log_diff"]],
            label="Actual returns",
            color="black",
        )
        plt.plot(
            forecast_index,
            forecast_df.forecast_means,
            label="Forecast GARCH(1,1)",
            color="blue",
            linestyle="--",
        )
        plt.fill_between(
            forecast_index,
            forecast_ci_lower,
            forecast_ci_upper,
            color="blue",
            alpha=0.2,
            label="95% CI forecast",
        )

        plt.axvline(x=forecast_index[0], color="red", linestyle=":", label="Forecast start")
        plt.title(
            f"Forecast {result.model_type.upper()}({result.p_mean_equation_lags},{result.q_variance_equation_lags}) Zoom"
        )
        plt.xlabel("Date")
        plt.ylabel("Returns")
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        base64_img = base64.b64encode(img_buffer.read()).decode("utf-8")
        plt.close()
        zoom_plot: str = f"data:image/png;base64,{base64_img}"

        return main_plot, zoom_plot

    @staticmethod
    def _get_rounded_ljung_box_test_results(test_results: dict) -> list[tuple]:
        results_list: list[tuple] = list(zip(test_results["index"], test_results["lb_stat"], test_results["lb_pvalue"]))
        return list(
            map(
                lambda result_tuple: (result_tuple[0], round(result_tuple[1], 6), round(result_tuple[2], 6)),
                results_list,
            )
        )

    @staticmethod
    def _get_rounded_engle_test_results(test_results: dict) -> dict:
        return {
            "p_value": round(test_results["p_value"], 6),
            "lm_statistics": round(test_results["lm_statistics"], 2),
        }


class GarchResultCreateAPIView(CreateAPIView):
    queryset = GarchResult.objects.all()
    serializer_class = GarchResultCreateSerializer

    def create(self, request, *args, **kwargs):
        """
        Generate requested plot and return it as an image.
        """
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid(raise_exception=False):
            # TODO: display errors inside template
            request.session["validation_errors"] = serializer.errors
            return redirect(
                "garch:garch-details",
                pk=request.data["object_id"],
            )

        validated_data = serializer.validated_data
        data_instance: Data = get_object_or_404(Data, pk=validated_data["object_id"])
        target_column: str = validated_data["target_column"]
        column_type: str = data_instance.data_columns.get(target_column)
        acf_lags: int = validated_data.get("acf_lags", 36)
        tests_lags: int = validated_data.get("tests_lags", 5)

        handler = GarchResult.SUPPORTED_HANDLERS.get(validated_data.get("model_type"))
        if not handler:
            raise ValidationError("Selected model type is not supported.")

        if not column_type or column_type not in handler.SUPPORTED_COLUMN_TYPES:
            raise ValidationError("Selected target column type is not supported.")

        forecast_horizon = validated_data.get("forecast_horizon")
        train_percentage = validated_data["train_percent"]
        validation_percentage = validated_data["val_percent"]
        test_percentage = validated_data["test_percent"]

        p_mean_equation_lags: int = validated_data["p_mean_equation_lags"]
        q_variance_equation_lags: int = validated_data.get("q_variance_equation_lags", 0)
        if validated_data["model_type"] == GarchResult.ARCH_MODEL:
            q_variance_equation_lags = 0

        model_handler = handler(
            data=data_instance,
            column_name=target_column,
            forecast_horizon=forecast_horizon,
            train_percentage=train_percentage,
            validation_percentage=validation_percentage,
            test_percentage=test_percentage,
            p_mean_equation_lags=p_mean_equation_lags,
            q_variance_equation_lags=q_variance_equation_lags,
            tests_lags=tests_lags,
        )
        model_metadata, forecast = model_handler.handle()
        GarchResult.objects.create(
            data=data_instance,
            target_column=target_column,
            forecast=forecast,
            forecast_horizon=forecast_horizon,
            model_type=validated_data["model_type"],
            summary=model_metadata["summary"],
            train_percentage=train_percentage,
            validation_percentage=validation_percentage,
            test_percentage=test_percentage,
            raw_data_engle_arch_test_results=model_metadata["raw_data_engle_arch_test_results"],
            model_fit_engle_arch_test_results=model_metadata["model_fit_engle_arch_test_results"],
            raw_data_ljung_box_test_results=model_metadata["raw_data_ljung_box_test_results"],
            raw_data_ljung_box_test_results_squared=model_metadata["raw_data_ljung_box_test_results_squared"],
            model_fit_ljung_box_test_results=model_metadata["model_fit_ljung_box_test_results"],
            model_fit_ljung_box_test_results_squared=model_metadata["model_fit_ljung_box_test_results_squared"],
            p_mean_equation_lags=p_mean_equation_lags,
            q_variance_equation_lags=q_variance_equation_lags,
            model_result_resid=model_metadata["model_result_resid"],
            model_result_conditional_volatility=model_metadata["model_result_conditional_volatility"],
            acf_lags=acf_lags,
            tests_lags=tests_lags,
            **model_metadata.get("val_statistics", {}),
            **model_metadata.get("test_statistics", {}),
        )
        return redirect("garch:garch-details", pk=data_instance.id)
