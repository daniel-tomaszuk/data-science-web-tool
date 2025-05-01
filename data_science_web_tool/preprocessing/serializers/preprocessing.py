from rest_framework import serializers


class ImageRetrieveSerializer(serializers.Serializer):
    axis_x_name = serializers.CharField(min_length=1)
    axis_y_name = serializers.CharField(min_length=1)
    group_by_name = serializers.CharField(min_length=1, required=False)
    plot_type = serializers.CharField(min_length=1)

    def validate(self, attrs: dict) -> dict:
        attrs = super().validate(attrs)
        if all(
            attrs.get(field_name, "-") == "-"
            for field_name in ("axis_x_name", "axis_y_name")
        ):
            raise serializers.ValidationError("At least one axis name is required.")

        return attrs
