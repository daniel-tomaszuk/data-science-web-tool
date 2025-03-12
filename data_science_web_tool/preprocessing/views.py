from django.db.models import QuerySet
from django.shortcuts import render
from django.views.generic import DetailView
from django.views.generic import ListView
from preprocessing.models import Data


class DataListView(ListView):
    model = Data
    queryset = Data.objects.all()
    template_name = "preprocessing/data/list.html"
    paginate_by = 3


class DataDetailView(DetailView):
    model = Data
    template_name = "preprocessing/data/details.html"
