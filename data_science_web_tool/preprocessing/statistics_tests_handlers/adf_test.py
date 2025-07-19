import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.diagnostic as smd


class ADFTestHandler:
    CRITICAL_VALUES_CONST = {
        "n": {
            "n_obs_gt_500": {
                "cv1": -2.567,
                "cv5": -1.941,
                "cv10": -1.616,
            },
            "n_obs_lte_500": {
                "cv1": np.nan,
                "cv5": np.nan,
                "cv10": np.nan,
            },
        },
        "c": {
            "n_obs_gt_500": {
                "cv1": -3.434,
                "cv5": -2.863,
                "cv10": -2.568,
            },
            "n_obs_lte_500": {
                "cv1": np.nan,
                "cv5": np.nan,
                "cv10": np.nan,
            },
        },
        "t": {
            "n_obs_gt_500": {
                "cv1": -3.963,
                "cv5": -3.413,
                "cv10": -3.128,
            },
            "n_obs_lte_500": {
                "cv1": np.nan,
                "cv5": np.nan,
                "cv10": np.nan,
            },
        },
    }
    SUPPORTED_TESTS_VERSIONS = tuple(CRITICAL_VALUES_CONST.keys())

    def __init__(
        self,
        series: pd.Series,
        *args,
        max_aug: int = 10,
        version: str = "n",
        differentiate_count: int = 0,
        **kwargs,
    ):
        self.series = series
        self.max_aug = max_aug
        self.version = version
        self.differentiate_count = differentiate_count if differentiate_count >= 0 else 0
        if self.version not in self.CRITICAL_VALUES_CONST:
            raise ValueError("Version not supported.")

        for i in range(self.differentiate_count):
            self.series = self.series.diff().dropna()

    def run(self):
        results = []

        y = self.series.diff()

        X = pd.DataFrame({"y_lag": self.series.shift()})
        if self.version == "c" or self.version == "t":  # dodanie stałej opcjonalnie
            X = sm.add_constant(X)

        if self.version == "t":  # dodanie komponentu trendu (deterministycznego) opcjonalnie
            X["trend"] = range(len(X))

        for i in range(0, self.max_aug):  # iteracja po różnych liczbach augmentacji
            for aug in range(1, i + 1):  # dodawanie augmentacji jedna po drugiej aż do obecnej liczby
                X["aug_" + str(aug)] = y.shift(aug)

            model = sm.OLS(self.series.diff(), X, missing="drop").fit()  # dopasowanie regresji liniowej metodą OLS
            ts = model.tvalues["y_lag"]  # statystyka testowa
            nobs: int = int(model.nobs)  # liczba obserwacji
            cv1, cv5, cv10 = self.__get_critical_values(n_observations=nobs)

            # test Breuscha-Godfreya dla autokorelacji reszt
            bg_test5 = smd.acorr_breusch_godfrey(model, nlags=5)
            bg_pvalue5 = round(bg_test5[1], 4)
            bg_test5 = smd.acorr_breusch_godfrey(model, nlags=10)
            bg_pvalue10 = round(bg_test5[1], 4)
            bg_test5 = smd.acorr_breusch_godfrey(model, nlags=15)
            bg_pvalue15 = round(bg_test5[1], 4)

            results.append([i, ts, cv1, cv5, cv10, bg_pvalue5, bg_pvalue10, bg_pvalue15])

        results_df = pd.DataFrame(results)
        results_df.columns = [
            "Augmentation Count",
            "ADF Test statistic",
            "Critical Value ADF (1%)",
            "Critical Value ADF (5%)",
            "Critical Value ADF (10%)",
            "Test BG (5 lags) (p-value)",
            "Test BG (10 lags) (p-value)",
            "Test BG (15 lags) (p-value)",
        ]
        return results_df

    def __get_critical_values(self, n_observations: int):
        critical_values = self.CRITICAL_VALUES_CONST[self.version]
        if n_observations > 500:
            return critical_values["n_obs_gt_500"].values()
        return critical_values["n_obs_lt_500"].values()
