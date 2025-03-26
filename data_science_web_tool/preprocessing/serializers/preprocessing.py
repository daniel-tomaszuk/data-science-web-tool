from rest_framework import serializers


class ImageRetrieveSerializer(serializers.Serializer):
    column_name = serializers.CharField(min_length=1)
    plot_type = serializers.CharField(min_length=1)
