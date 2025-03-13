from django.urls import path
from preprocessing import views

app_name = "preprocessing"

urlpatterns = [
    path("", views.DataListView.as_view(), name="data-list"),
    path("<int:pk>/", views.DataDetailView.as_view(), name="data-detail"),
]
