import pandas as pd
import yfinance as yf

from preprocessing.data_sources_handlers.base import DataSourceHandlerBase


class YFinanceDataSourceHandler(DataSourceHandlerBase):
    SNIFFING_CHARS_COUNT = 2048
    INTERVAL_CHOICES = (
        # "1m", "5m", "15m", "1h",  # Additional logic required, not supported for now.
        "1d",
        "1wk",
        "1mo",
    )
    PERIOD_CHOICES = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max")

    def load_data(self) -> list[dict]:
        """Fetches data from Yahoo Finance"""
        ticker = self.kwargs.get("ticker")
        if not ticker:
            return []

        period = self.kwargs.get("period")
        interval = self.kwargs.get("interval")
        df: pd.DataFrame = yf.download(ticker, period=period, interval=interval)
        df_reset = df.reset_index()

        # Connect tuple key into one column name
        df_reset.columns = ["Date"] + [f"{col[0]}_{col[1]}" for col in df.columns]

        # Transform datetime values into date strings so they can be used in json
        df_reset["Date"] = df_reset["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # dump into dict
        json_data = df_reset.to_dict(orient="records")
        return json_data
