from django.urls import include
from django.urls import path
from preprocessing import views
from rest_framework.routers import DefaultRouter

app_name = "preprocessing"


urlpatterns = [
    path("", views.DataListView.as_view(), name="data-list"),
    path("<int:pk>/", views.DataDetailView.as_view(), name="data-detail"),
    path(
        "<int:pk>/change-column-type/",
        views.DetailChangeColumnType.as_view(),
        name="change-column-type",
    ),
]
