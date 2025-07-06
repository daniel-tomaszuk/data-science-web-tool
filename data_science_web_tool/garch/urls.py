from django.urls import path
from garch import views

app_name = "garch"


urlpatterns = [
    path(
        "<int:pk>/",
        views.GarchResultView.as_view(),
        name="garch-details",
    ),
    path(
        "create/",
        views.GarchResultCreateAPIView.as_view(),
        name="garch-create",
    ),
]
