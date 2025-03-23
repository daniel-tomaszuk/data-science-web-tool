from django.urls import path
from general import views

app_name = "general"


urlpatterns = [
    path("", views.DataListView.as_view(), name="data-list"),
]
