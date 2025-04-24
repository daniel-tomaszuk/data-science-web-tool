from django.urls import path
from integrations import views

app_name = "integrations"


urlpatterns = [
    path("yfinance/", views.YahooFinanceListView.as_view(), name="yahoo-finance-list"),
    path(
        "yfinance/download/<str:symbol>/",
        views.YahooFinanceUploadCreateView.as_view(),
        name="yahoo-finance-download",
    ),
]
