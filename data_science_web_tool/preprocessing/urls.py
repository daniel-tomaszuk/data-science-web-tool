from django.urls import path
from preprocessing import views

app_name = "preprocessing"


urlpatterns = [
    path("<int:pk>/", views.DataDetailView.as_view(), name="data-detail"),
    path(
        "<int:pk>/change-column-type/",
        views.DetailChangeColumnType.as_view(),
        name="change-column-type",
    ),
]
