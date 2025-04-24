import pandas as pd
import yfinance as yf

from preprocessing.data_sources_handlers.base import DataSourceHandlerBase


class YFinanceDataSourceHandler(DataSourceHandlerBase):
    SNIFFING_CHARS_COUNT = 2048

    def load_data(self) -> list[dict]:
        """Fetches data from Yahoo Finance"""
        ticker = self.kwargs.get("ticker")
        if not ticker:
            return []

        df: pd.DataFrame = yf.download(ticker)
        df_reset = df.reset_index()

        # Connect tuple key into one column name
        df_reset.columns = ["Date"] + [f"{col[0]}_{col[1]}" for col in df.columns]

        # Transform datetime values into date strings so they can be used in json
        df_reset["Date"] = df_reset["Date"].dt.strftime("%Y-%m-%d")

        # dump into dict
        json_data = df_reset.to_dict(orient="records")
        return json_data
