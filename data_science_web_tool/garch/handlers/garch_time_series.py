import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch

from preprocessing.models import Data


class GarchHandlerBase:
    SUPPORTED_COLUMN_TYPES = (
        "int64",
        "float64",
    )

    def __init__(
        self,
        data: Data,
        column_name: str,
        forecast_horizon: int | None = None,
        train_percentage: int | None = None,
        validation_percentage: int | None = None,
        test_percentage: int | None = None,
        arch_order: int = 0,
        garch_order: int = 0,
        tests_lags: int = 5,
    ):
        self.data = data
        self.column_name = column_name
        self.column_name_log_diff = f"{column_name}_log_diff"
        self.forecast_horizon = forecast_horizon
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage
        self.arch_order = arch_order
        self.garch_order = garch_order
        self.tests_lags = tests_lags
        self.model_type = None

    def engle_arch_test(
        self,
        *,
        df: pd.DataFrame | None = None,
        model_result=None,
    ) -> dict:
        """
        H0: No ARCH effect.
        H1: Presence of ARCH effect.
        """
        if df is None and not model_result:
            raise ValueError("Must provide either `df` or `model_result` args.")

        if df is not None and model_result:
            raise ValueError("Must provide either `df` or `model_result` args.")

        if df is not None:
            returns_for_arch_test = df[self.column_name_log_diff].dropna()
            test_results = het_arch(returns_for_arch_test, nlags=self.tests_lags)
        else:
            resid_std = model_result.resid / model_result.conditional_volatility
            test_results = het_arch(resid_std, nlags=self.tests_lags)

        lm_statistics, p_value = test_results[0], test_results[1]
        return dict(lm_statistics=lm_statistics, p_value=p_value)

    def ljung_box_test(
        self,
        *,
        df: pd.DataFrame | None = None,
        model_result: pd.DataFrame | None = None,
        squared: bool = False,
    ) -> dict:
        """
        H0: The autocorrelations of the time series are all zero up to a specified lag (k).
        H1: At least one of the autocorrelations is non-zero.
        """
        if df is None and not model_result:
            raise ValueError("Must provide either `df` or `model_result` args.")

        if df is not None and model_result:
            raise ValueError("Must provide either `df` or `model_result` args.")

        lags: tuple[int, ...] = tuple(i for i in range(self.tests_lags, (self.tests_lags**2) + 1, self.tests_lags))
        if df is not None:
            returns_for_arch_test = df[self.column_name_log_diff].dropna()
            if squared:
                returns_for_arch_test = returns_for_arch_test**2

            test_result: pd.DataFrame = acorr_ljungbox(
                returns_for_arch_test,
                lags=lags,
                return_df=True,
            )
        else:
            resid_std = model_result.resid / model_result.conditional_volatility
            if squared:
                resid_std = resid_std**2

            test_result: pd.DataFrame = acorr_ljungbox(
                resid_std,
                lags=lags,
                return_df=True,
            )
        return dict(
            index=list(test_result.index),
            lb_stat=list(test_result.lb_stat),
            lb_pvalue=list(test_result.lb_pvalue),
        )

    def _prepare_df_data_sets(self):
        df: pd.DataFrame = self.data.get_df()
        df[self.column_name_log_diff] = np.log(df[self.column_name]).diff()
        df.dropna(inplace=True)
        df.reset_index(inplace=True)

        train_df, val_df, test_df = self._get_model_data_sets(df)
        return df, train_df, val_df, test_df

    def _get_model_data_sets(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
        """
        Returns training, validation and test data sets by getting data percentages selected by the user.
        """
        total_rows: int = len(df)

        train_size = round(total_rows * self.train_percentage / 100)
        val_size = round(total_rows * self.validation_percentage / 100)
        if train_size + val_size > total_rows:
            val_size = total_rows - train_size

        # Now define the index bounds
        train_end = train_size
        val_end = train_size + val_size

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        return train_df, val_df, test_df

    def _get_model_and_metadata(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple:
        model_train_set = arch_model(
            train_df[f"{self.column_name}_log_diff"],
            vol=self.model_type.upper(),
            p=self.arch_order,
            q=self.garch_order,
            rescale=False,
        )

        model_train_set_result = model_train_set.fit(disp="off")
        summary = model_train_set_result.summary()

        raw_data_engle_arch_test_results = self.engle_arch_test(df=train_df)
        raw_data_ljung_box_test_results = self.ljung_box_test(df=train_df)
        raw_data_ljung_box_test_results_squared = self.ljung_box_test(df=train_df, squared=True)

        model_fit_engle_arch_test_results = self.engle_arch_test(model_result=model_train_set_result)
        model_fit_ljung_box_test_results = self.ljung_box_test(model_result=model_train_set_result)
        model_fit_ljung_box_test_results_squared = self.ljung_box_test(
            model_result=model_train_set_result,
            squared=True,
        )

        val_statistics = self._get_model_statistics(
            main_df=train_df,
            compare_df=val_df,
            keys_prefix="val_",
        )
        test_statistics = self._get_model_statistics(
            main_df=pd.concat([train_df, val_df]),
            compare_df=test_df,
            keys_prefix="test_",
        )

        # Get statistics for validation and train sets
        model_metadata = {
            "train_values": train_df[self.column_name_log_diff],
            "val_statistics": val_statistics,
            "test_statistics": test_statistics,
            "raw_data_engle_arch_test_results": raw_data_engle_arch_test_results,
            "raw_data_ljung_box_test_results_squared": raw_data_ljung_box_test_results_squared,
            "model_fit_engle_arch_test_results": model_fit_engle_arch_test_results,
            "raw_data_ljung_box_test_results": raw_data_ljung_box_test_results,
            "model_fit_ljung_box_test_results": model_fit_ljung_box_test_results,
            "model_fit_ljung_box_test_results_squared": model_fit_ljung_box_test_results_squared,
            "model_result_resid": dict(model_train_set_result.resid),
            "model_result_conditional_volatility": dict(model_train_set_result.conditional_volatility),
            "summary": summary,
        }
        future_forecast = self._forecast_future_values(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )
        return model_metadata, future_forecast

    def _get_model_statistics(
        self,
        main_df: pd.DataFrame,
        compare_df: pd.DataFrame,
        keys_prefix: str = "",
    ) -> dict:
        """
        Takes `main_df` data, creates productions for it and the compares forecasts
        with `compare_df` (validation or test sets) data.
        """
        main_returns = main_df[self.column_name_log_diff].dropna().copy()
        compare_returns = compare_df[self.column_name_log_diff].dropna().copy()

        forecast_means = []
        forecast_vars = []
        forecast_index = compare_returns.index
        for i in range(len(forecast_index)):
            if i == 0:
                extended_train = main_returns
            else:
                extended_train = pd.concat([main_returns, pd.Series(forecast_means, index=forecast_index[:i])])

            model = arch_model(
                extended_train,
                vol=self.model_type,
                p=self.arch_order,
                q=self.garch_order,
                rescale=False,
            )
            fit = model.fit(disp="off")

            # forecast 1 time unit into the future, then append results and forecast once more
            forecast = fit.forecast(horizon=1)
            mu = forecast.mean.iloc[-1, 0]
            var = forecast.variance.iloc[-1, 0]
            forecast_means.append(mu)
            forecast_vars.append(var)

        forecast_df = pd.DataFrame(
            {
                "forecast_mean": forecast_means,
                "forecast_variance": forecast_vars,
                "forecast_std": np.sqrt(forecast_vars),
                "actual": compare_returns,
            },
            index=forecast_index,
        )

        # --- Volatility forecast evaluation ---
        actual_squared = forecast_df["actual"] ** 2
        predicted_variance = forecast_df["forecast_variance"]

        vol_mse = mean_squared_error(actual_squared, predicted_variance)
        vol_mae = mean_absolute_error(actual_squared, predicted_variance)
        vol_rmse = np.sqrt(vol_mse)
        vol_qlike = np.mean(np.log(predicted_variance) + actual_squared / predicted_variance)

        # --- Mean forecast evaluation ---
        predicted_mean = forecast_df["forecast_mean"]
        actual = forecast_df["actual"]

        mean_r2 = r2_score(actual, predicted_mean)
        mean_mse = mean_squared_error(actual, predicted_mean)
        mean_mae = mean_absolute_error(actual, predicted_mean)
        mean_rmse = np.sqrt(mean_mse)
        mean_mape = mean_absolute_percentage_error(actual, predicted_mean)
        mean_smape = self._smape(actual, predicted_mean)
        return {
            keys_prefix + "vol_mse": round(float(vol_mse), 6),
            keys_prefix + "vol_mae": round(float(vol_mae), 6),
            keys_prefix + "vol_rmse": round(float(vol_rmse), 6),
            keys_prefix + "vol_qlike": round(float(vol_qlike), 6),
            keys_prefix + "mean_r2": round(float(mean_r2), 6),
            keys_prefix + "mean_mse": round(float(mean_mse), 6),
            keys_prefix + "mean_mae": round(float(mean_mae), 6),
            keys_prefix + "mean_rmse": round(float(mean_rmse), 6),
            keys_prefix + "mean_mape": round(float(mean_mape), 6),
            keys_prefix + "mean_smape": round(float(mean_smape), 6),
        }

    def _forecast_future_values(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        """
        Tries to predict n future value by predicting n-1 future value and appending it to the prediction list.
        """
        if not self.forecast_horizon:
            return {}

        train_returns = train_df[self.column_name_log_diff].dropna().copy()
        val_returns = val_df[self.column_name_log_diff].dropna().copy()
        test_returns = test_df[self.column_name_log_diff].dropna().copy()
        data_for_model = pd.concat([train_returns.copy(), val_returns.copy(), test_returns.copy()])

        forecast_means = []
        forecast_vars = []
        last_index = data_for_model.index[-1]
        forecast_index = []
        for i in range(self.forecast_horizon):
            forecast_index_i = int(last_index) + i
            if forecast_means:
                # add last forecast mean to the overall data
                data_for_model = pd.concat([data_for_model, pd.Series(forecast_means[-1], index=[forecast_index_i])])

            model = arch_model(
                data_for_model,
                vol=self.model_type,
                p=self.arch_order,
                q=self.garch_order,
                rescale=False,
            )
            model_result = model.fit(disp="off")
            forecast = model_result.forecast(horizon=1)
            mu = forecast.mean.iloc[-1, 0]
            var = forecast.variance.iloc[-1, 0]
            forecast_means.append(mu)
            forecast_vars.append(var)
            forecast_index.append(forecast_index_i)

        return {
            "forecast_means": [round(float(value), 8) for value in forecast_means],
            "forecast_variance": [round(float(value), 8) for value in forecast_vars],
            "forecast_index": forecast_index,
        }

    def _smape(self, y_true, y_pred) -> float:
        """
        Symmetric Mean Absolute Percentage Error.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred)

        # Avoid division by zero
        nonzero_mask = denominator != 0
        smape_value = np.mean(diff[nonzero_mask] / denominator[nonzero_mask])

        return smape_value


class ArchTimeSeriesHandler(GarchHandlerBase):
    def handle(self):
        df, train_df, val_df, test_df = self._prepare_df_data_sets()
        model_metadata, forecast = self._arch(train_df, val_df, test_df)
        return model_metadata, forecast

    def _arch(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple:
        self.model_type = "ARCH"
        return self._get_model_and_metadata(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )


class GarchTimeSeriesHandler(GarchHandlerBase):
    def handle(self):
        df, train_df, val_df, test_df = self._prepare_df_data_sets()
        model_metadata, forecast = self._garch(train_df, val_df, test_df)
        return model_metadata, forecast

    def _garch(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple:
        self.model_type = "GARCH"
        return self._get_model_and_metadata(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )
