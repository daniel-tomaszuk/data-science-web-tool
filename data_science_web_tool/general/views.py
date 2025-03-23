from django.views.generic import ListView
from preprocessing.models import Data


class DataListView(ListView):
    model = Data
    queryset = Data.objects.all()
    template_name = "general/data/list.html"
    paginate_by = 10
