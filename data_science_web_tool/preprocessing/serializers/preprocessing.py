from rest_framework import serializers


class ImageRetrieveSerializer(serializers.Serializer):
    axis_x_name = serializers.CharField(min_length=1)
    axis_y_name = serializers.CharField(min_length=1)
    plot_type = serializers.CharField(min_length=1)
