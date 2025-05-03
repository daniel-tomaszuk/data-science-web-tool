from django.contrib import admin

from linear_regression.models import LinearRegressionTimeSeriesResult


@admin.register(LinearRegressionTimeSeriesResult)
class LinearRegressionTimeSeriesResultAdmin(admin.ModelAdmin):
    """
    Admin for LinearRegressionTimeSeriesResult
    """
