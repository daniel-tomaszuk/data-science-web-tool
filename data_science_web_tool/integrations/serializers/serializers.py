from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from preprocessing.data_sources_handlers.yfinance_source_handler import YFinanceDataSourceHandler


class YFinanceDataFormDownloadSerializer(serializers.Serializer):

    ticker = serializers.CharField(required=False, allow_blank=True)
    custom_ticker = serializers.CharField(required=False, allow_blank=True)
    period = serializers.CharField()
    interval = serializers.CharField()

    def validate(self, attrs: dict) -> dict:
        validated_data = super().validate(attrs)
        if not validated_data.get("ticker") and not validated_data.get("custom_ticker"):
            raise ValidationError("No ticker provided.")

        if validated_data.get("ticker") and validated_data.get("custom_ticker"):
            raise ValidationError("Only one ticker should be provided.")

        return validated_data

    def validate_period(self, period: str) -> str:
        if period not in YFinanceDataSourceHandler.PERIOD_CHOICES:
            raise ValidationError("Invalid period value.")

        return period

    def validate_interval(self, interval: str) -> str:
        if interval not in YFinanceDataSourceHandler.INTERVAL_CHOICES:
            raise ValidationError("Invalid interval value.")
        return interval
