from django.urls import path

from linear_regression import views

app_name = "linear_regression"


urlpatterns = [
    path(
        "<int:pk>/",
        views.LinearRegressionView.as_view(),
        name="linear-regression-details",
    ),
    path(
        "time-series/create/",
        views.LinearRegressionTimeSeriesCreateAPIView.as_view(),
        name="linear-regression-time-series-create",
    ),
]
