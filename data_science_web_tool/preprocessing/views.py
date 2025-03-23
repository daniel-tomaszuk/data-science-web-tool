from django.contrib import messages
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect
from django.views.generic import DetailView
from preprocessing.models import Data
from rest_framework.generics import CreateAPIView


class DataDetailView(DetailView):
    model = Data
    template_name = "preprocessing/data/details.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["statistics"] = self.object.get_statistics()

        image, selected_column = self.object.get_histogram("Adj Close")
        context["histogram"] = image
        context["histogram_column"] = selected_column

        return context


class DetailChangeColumnType(CreateAPIView):
    queryset = Data.objects.all()
    serializer_class = None

    def create(self, request, *args, **kwargs):
        """
        Tries to change column type of selected data frame.
        Returns HTTP 400 Bad Request if operation fails.
        """
        instance: Data = get_object_or_404(Data, id=self.kwargs.get("pk", ""))
        column_types = {}
        current_columns = list(instance.data_columns.keys())
        for key, value in request.POST.items():
            if not key.startswith("data_type_"):
                continue

            new_type = request.POST.get(key, "")
            if not new_type or new_type not in Data.SUPPORTED_COLUMN_TYPES:
                messages.error(request, "Invalid column type.")
                return redirect("preprocessing:data-detail", pk=instance.id)

            column_name = key.replace("data_type_", "")
            if column_name not in current_columns:
                messages.error(request, "Invalid column name.")
                return redirect("preprocessing:data-detail", pk=instance.id)

            column_types[column_name] = new_type

        instance.data_columns = column_types
        is_success = True
        try:
            instance.save()
        except Exception as e:
            is_success = False
            messages.error(request, str(e))

        if is_success:
            messages.success(request, "Column types updated successfully!")

        return redirect("preprocessing:data-detail", pk=instance.id)
