import base64
import io
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["statistics"] = self.object.get_statistics()
        # context["histogram"] = self.object.get_histogram("")
        return context
