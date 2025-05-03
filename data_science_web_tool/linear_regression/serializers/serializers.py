from rest_framework import serializers
from setuptools.config.pyprojecttoml import validate

from linear_regression.handlers.time_series import LinearRegressionTimeSeriesHandler


class LinearRegressionTimeSeriesCreateSerializer(serializers.Serializer):
    lag = serializers.IntegerField(min_value=1)
    normalize = serializers.BooleanField(default=False, required=False)
    target_column = serializers.CharField()
    object_id = serializers.IntegerField(required=True)
