from garch.models import GarchResult
from rest_framework import serializers


class GarchResultCreateSerializer(serializers.Serializer):
    model_type = serializers.ChoiceField(choices=GarchResult.MODEL_TYPES_CHOICES)
    forecast_horizon = serializers.IntegerField(min_value=0, required=False, allow_null=True)
    target_column = serializers.CharField()
    object_id = serializers.IntegerField(required=True)

    p_mean_equation_lags = serializers.IntegerField(required=True, min_value=0)
    q_variance_equation_lags = serializers.IntegerField(required=False, min_value=0)

    acf_lags = serializers.IntegerField(required=False, min_value=1, default=36)
    tests_lags = serializers.IntegerField(required=False, min_value=1, default=36)

    train_percent = serializers.IntegerField(min_value=1, max_value=100)
    val_percent = serializers.IntegerField(min_value=1, max_value=100)
    test_percent = serializers.IntegerField(min_value=1, max_value=100)

    def validate(self, data: dict) -> dict:
        data = super().validate(data)
        self.__validate_data_split_percentages(data)
        return data

    def __validate_data_split_percentages(self, data: dict):
        train_percent = data["train_percent"]
        validation_percent = data["val_percent"]
        test_percent = data["test_percent"]

        if not train_percent:
            raise serializers.ValidationError("Train data set can not be empty.")

        if not validation_percent:
            raise serializers.ValidationError("Validation data set can not be empty.")

        if not test_percent:
            raise serializers.ValidationError("Test data set can not be empty.")

        if train_percent + validation_percent + test_percent > 100:
            raise serializers.ValidationError("Sum of train, validation and test percentages cannot be greater than 100%")

        if train_percent + validation_percent + test_percent < 0:
            raise serializers.ValidationError("Sum of train, validation and test percentages cannot be less than 0%")

