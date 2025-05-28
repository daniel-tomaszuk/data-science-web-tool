from functools import lru_cache

import yfinance as yf
from django.db import transaction
from django.shortcuts import redirect
from django.utils import timezone
from django.views.generic import TemplateView
from rest_framework.generics import CreateAPIView

from integrations.serializers.serializers import YFinanceDataFormDownloadSerializer
from preprocessing.models import Data
from preprocessing.models import DataUpload


@lru_cache
def _get_yfinance_data(base_tickers: tuple) -> list[dict]:
    """
    Gets data from Yahoo Finance API and returns it as a list of dicts.
    """
    tickers_data = []
    tickers_batch = yf.Tickers(" ".join(base_tickers))
    for ticker_symbol, ticker in tickers_batch.tickers.items():
        info = ticker.info
        tickers_data.append(
            {
                "symbol": ticker_symbol,
                "shortName": info.get("shortName", ""),
                "sector": info.get("sector", ""),
                "marketCap": info.get("marketCap", ""),
            }
        )
    return tickers_data


class YahooFinanceListView(TemplateView):
    BASE_TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY")
    template_name = "integrations/yfinance.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["yfinance_error"] = None
        try:
            context["yfinance"] = _get_yfinance_data(self.BASE_TICKERS)
        except Exception:
            context["yfinance_error"] = (
                "Yahoo Finance API is not available. Visit https://query2.finance.yahoo.com for more details."
            )
        return context


class YahooFinanceUploadCreateView(CreateAPIView):
    model = DataUpload
    serializer_class = YFinanceDataFormDownloadSerializer
    created_data: Data = None

    def create(self, request, *args, **kwargs):
        super().create(request, *args, **kwargs)
        return redirect("preprocessing:data-detail", pk=self.created_data.id)

    def perform_create(self, serializer: YFinanceDataFormDownloadSerializer):
        validated_data: dict = serializer.validated_data
        ticker = validated_data.get("ticker")
        custom_ticker = validated_data.get("custom_ticker")
        ticker = ticker or custom_ticker
        data_source_handler = DataUpload.DATA_SOURCE_TYPE_PROCESSORS["yfinance"]
        data_source_handler = data_source_handler(
            ticker=ticker,
            period=validated_data.get("period"),
            interval=validated_data.get("interval"),
        )
        data: list[dict] = data_source_handler.load_data()
        timestamp = timezone.now().strftime("%Y-%m-%d %H:%M:%S")
        name = f"yFinance {ticker} {timestamp}.json"
        description = f"Automatically fetched from yFinance at {timestamp}"
        with transaction.atomic():
            self.created_data: Data = Data.objects.create(
                data=data,
                name=name,
                description=description,
            )
            DataUpload.objects.create(
                file_name=name,
                file_type=DataUpload.YFINANACE,
                description=description,
                file=None,
                data=self.created_data,
            )
