from rest_framework import serializers

from linear_regression.models import LinearRegressionTimeSeriesResult


class LinearRegressionTimeSeriesCreateSerializer(serializers.Serializer):
    model_type = serializers.ChoiceField(
        choices=LinearRegressionTimeSeriesResult.MODEL_TYPES_CHOICES
    )
    lag = serializers.IntegerField(min_value=1)
    normalize = serializers.BooleanField(default=False, required=False)
    max_tree_depth = serializers.IntegerField(
        min_value=1, required=False, allow_null=True
    )
    target_column = serializers.CharField()
    object_id = serializers.IntegerField(required=True)
