from rest_framework import serializers

from preprocessing.models import DataTestResult


class ImageRetrieveSerializer(serializers.Serializer):
    axis_x_name = serializers.CharField(min_length=1)
    axis_y_name = serializers.CharField(min_length=1)
    group_by_name = serializers.CharField(min_length=1, required=False)
    plot_type = serializers.CharField(min_length=1)

    def validate(self, attrs: dict) -> dict:
        attrs = super().validate(attrs)
        if all(attrs.get(field_name, "-") == "-" for field_name in ("axis_x_name", "axis_y_name")):
            raise serializers.ValidationError("At least one axis name is required.")

        return attrs


class CreateStationaryTestRequestSerializer(serializers.Serializer):
    test_type = serializers.CharField(min_length=1)
    target_column = serializers.CharField(min_length=1)
    max_augmentation_count = serializers.IntegerField(min_value=1, max_value=100)
    differentiate_count = serializers.IntegerField(min_value=0, max_value=10)
    test_version = serializers.CharField(min_length=1, max_length=1)

    def validate(self, attrs: dict) -> dict:
        attrs: dict = super().validate(attrs)

        test_type = attrs.get("test_type")
        test_handler = DataTestResult.SUPPORTED_TEST_HANDLERS.get(test_type)
        if not test_handler:
            raise serializers.ValidationError(f"Unknown test type selected: {test_type}")

        if attrs.get("test_version") not in test_handler.SUPPORTED_TESTS_VERSIONS:
            raise serializers.ValidationError("Unsupported test version.")

        return attrs
