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
    SUPPORTED_COLUMN_TYPES = ("int64", "float64")

    def __init__(
        self,
        data: Data,
        column_name: str,
        forecast_horizon: int | None = None,
        train_percentage: int | None = None,
        validation_percentage: int | None = None,
        test_percentage: int | None = None,
        arch_order: int = 0,  # p w wariancji
        garch_order: int = 0,  # q w wariancji
        tests_lags: int = 5,
        mean_spec: str = "Constant",  # "Constant" lub "Zero"
        mean_lags: int = 0,  # gdybyś chciał ARX -> ustaw >0 i mean_spec="ARX"
        error_dist: str = "normal",  # rozważ: "StudentsT"
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
        self.model_type = None  # "ARCH"/"GARCH"
        self.mean_spec = mean_spec
        self.mean_lags = mean_lags
        self.error_dist = error_dist

    # ---------- helpers ----------

    def _build_model(self, series: pd.Series):
        vol_spec = self.model_type.upper()
        q_order = 0 if vol_spec == "ARCH" else self.garch_order
        return arch_model(
            series,
            mean=self.mean_spec,
            lags=self.mean_lags,
            vol=vol_spec,
            p=self.arch_order,
            q=q_order,
            dist=self.error_dist,
            rescale=False,
        )

    # ---------- tests ----------

    def engle_arch_test(self, *, df: pd.DataFrame | None = None, model_result=None) -> dict:
        """
        H0: No ARCH effect. H1: Presence of ARCH effect.
        """
        if (df is None) == (model_result is None):
            raise ValueError("Must provide either `df` or `model_result` args (exclusively).")

        if df is not None:
            returns_for_arch_test = df[self.column_name_log_diff].dropna()
            test_results = het_arch(returns_for_arch_test, nlags=self.tests_lags)
        else:
            resid_std = model_result.resid / model_result.conditional_volatility
            test_results = het_arch(resid_std, nlags=self.tests_lags)

        lm_statistics, p_value = test_results[0], test_results[1]
        return dict(lm_statistics=float(lm_statistics), p_value=float(p_value))

    def ljung_box_test(
        self,
        *,
        df: pd.DataFrame | None = None,
        model_result=None,
        squared: bool = False,
    ) -> dict:
        """
        H0: all autocorrelations up to k are zero.
        """
        if (df is None) == (model_result is None):
            raise ValueError("Must provide either `df` or `model_result` args (exclusively).")

        lags: tuple[int, ...] = tuple(i for i in range(self.tests_lags, (self.tests_lags**2) + 1, self.tests_lags))
        if df is not None:
            series = df[self.column_name_log_diff].dropna()
        else:
            series = (model_result.resid / model_result.conditional_volatility).dropna()
        if squared:
            series = series**2

        test_result: pd.DataFrame = acorr_ljungbox(series, lags=lags, return_df=True)
        return dict(
            index=list(map(int, test_result.index)),
            lb_stat=list(map(float, test_result.lb_stat)),
            lb_pvalue=list(map(float, test_result.lb_pvalue)),
        )

    # ---------- data prep ----------

    def _prepare_df_data_sets(self):
        df: pd.DataFrame = self.data.get_df()
        df[self.column_name_log_diff] = np.log(df[self.column_name]).diff()
        df.dropna(inplace=True)
        df.reset_index(inplace=True)

        train_df, val_df, test_df = self._get_model_data_sets(df)
        return df, train_df, val_df, test_df

    def _get_model_data_sets(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
        total_rows: int = len(df)
        train_size = round(total_rows * self.train_percentage / 100)
        val_size = round(total_rows * self.validation_percentage / 100)
        if train_size + val_size > total_rows:
            val_size = total_rows - train_size

        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size : train_size + val_size]
        test_df = df.iloc[train_size + val_size :]
        return train_df, val_df, test_df

    # ---------- fit + metadata ----------

    def _get_model_and_metadata(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple:
        series = train_df[self.column_name_log_diff]
        model_train_set = self._build_model(series)
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

        val_statistics = self._get_model_statistics(main_df=train_df, compare_df=val_df, keys_prefix="val_")
        test_statistics = self._get_model_statistics(
            main_df=pd.concat([train_df, val_df]), compare_df=test_df, keys_prefix="test_"
        )

        model_metadata = {
            "train_values": series,
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
        future_forecast = self._forecast_future_values(train_df=train_df, val_df=val_df, test_df=test_df)
        return model_metadata, future_forecast

    # ---------- rolling-origin metrics ----------

    def _get_model_statistics(
        self,
        main_df: pd.DataFrame,
        compare_df: pd.DataFrame,
        keys_prefix: str = "",
    ) -> dict:
        """
        Rolling 1-step-ahead on REALIZED returns:
        fit on [main + compare[:i]] -> forecast compare[i].
        """
        main_returns = main_df[self.column_name_log_diff].dropna().copy()
        compare_returns = compare_df[self.column_name_log_diff].dropna().copy()

        forecast_means, forecast_vars = [], []
        forecast_index = compare_returns.index

        for i, _ in enumerate(forecast_index):
            extended_train = main_returns if i == 0 else pd.concat([main_returns, compare_returns.iloc[:i]])

            fit = self._build_model(extended_train).fit(disp="off")
            fc = fit.forecast(horizon=1, reindex=False)

            mu = float(fc.mean.iloc[-1, 0])
            var = float(fc.variance.iloc[-1, 0])

            forecast_means.append(mu)
            forecast_vars.append(var)

        forecast_df = pd.DataFrame(
            {
                "forecast_mean": forecast_means,
                "forecast_variance": forecast_vars,
                "forecast_std": np.sqrt(forecast_vars),
                "actual": compare_returns.values,
            },
            index=forecast_index,
        )

        # --- Volatility forecast evaluation ---
        eps = 1e-12
        actual_squared = (forecast_df["actual"].values) ** 2
        predicted_variance = np.clip(forecast_df["forecast_variance"].values, eps, None)

        vol_mse = mean_squared_error(actual_squared, predicted_variance)
        vol_mae = mean_absolute_error(actual_squared, predicted_variance)
        vol_rmse = float(np.sqrt(vol_mse))
        vol_qlike = float(np.mean(np.log(predicted_variance) + (actual_squared / predicted_variance)))

        # --- Mean forecast evaluation (informacyjnie) ---
        predicted_mean = forecast_df["forecast_mean"].values
        actual = forecast_df["actual"].values

        mean_r2 = r2_score(actual, predicted_mean)
        mean_mse = mean_squared_error(actual, predicted_mean)
        mean_mae = mean_absolute_error(actual, predicted_mean)
        mean_rmse = float(np.sqrt(mean_mse))
        try:
            mean_mape = mean_absolute_percentage_error(actual, predicted_mean)
        except Exception:
            mean_mape = float("nan")
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
            keys_prefix + "mean_mape": round(float(mean_mape), 6) if np.isfinite(mean_mape) else "N/A",
            keys_prefix + "mean_smape": round(float(mean_smape), 6),
        }

    # ---------- multi-step out-of-sample forecast ----------

    def _forecast_future_values(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        """
        Wielokrokowa prognoza poza próbkę (fit raz i forecast na h kroków).
        """
        if not self.forecast_horizon:
            return {}

        train_returns = train_df[self.column_name_log_diff].dropna().copy()
        val_returns = val_df[self.column_name_log_diff].dropna().copy()
        test_returns = test_df[self.column_name_log_diff].dropna().copy()
        data_for_model = pd.concat([train_returns, val_returns, test_returns])

        fit = self._build_model(data_for_model).fit(disp="off")
        fc = fit.forecast(horizon=self.forecast_horizon, reindex=False)

        means = [round(float(x), 8) for x in fc.mean.iloc[-1].tolist()]
        vars_ = [round(float(x), 8) for x in fc.variance.iloc[-1].tolist()]

        last_index = int(data_for_model.index[-1])
        forecast_index = list(range(last_index + 1, last_index + 1 + self.forecast_horizon))

        return {
            "forecast_means": means,
            "forecast_variance": vars_,
            "forecast_index": forecast_index,
        }

    # ---------- misc ----------

    def _smape(self, y_true, y_pred) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred)
        mask = denom != 0
        return float(np.mean(diff[mask] / denom[mask]))


class ArchTimeSeriesHandler(GarchHandlerBase):
    def handle(self):
        df, train_df, val_df, test_df = self._prepare_df_data_sets()
        model_metadata, forecast = self._arch(train_df, val_df, test_df)
        return model_metadata, forecast

    def _arch(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        self.model_type = "ARCH"
        return self._get_model_and_metadata(train_df=train_df, val_df=val_df, test_df=test_df)


class GarchTimeSeriesHandler(GarchHandlerBase):
    def handle(self):
        df, train_df, val_df, test_df = self._prepare_df_data_sets()
        model_metadata, forecast = self._garch(train_df, val_df, test_df)
        return model_metadata, forecast

    def _garch(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        self.model_type = "GARCH"
        return self._get_model_and_metadata(train_df=train_df, val_df=val_df, test_df=test_df)
