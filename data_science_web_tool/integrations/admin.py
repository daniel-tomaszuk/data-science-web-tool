from django import forms
from django.contrib import admin

from integrations.models import Integration


class IntegrationAdminForm(forms.ModelForm):
    class Meta:
        model = Integration
        fields = (
            "name",
            "description",
            "url",
            "api_key",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.id and "api_key" in self.fields:
            self.fields["api_key"].widget.attrs[
                "placeholder"
            ] = self.instance.get_api_key_safe()
            self.fields["api_key"].required = False

    def clean_api_key(self):
        data = self.cleaned_data.get("api_key")
        if not data and self.instance:
            return self.instance.api_key
        return data


@admin.register(Integration)
class IntegrationAdmin(admin.ModelAdmin):
    """
    Admin panel for API integration.
    """

    form = IntegrationAdminForm
    list_display = ("name", "url")
    search_fields = ("name", "url")
    readonly_fields = ("integration_api_key",)

    def get_fields(self, request, obj=None):
        # Customize field layout so both editable and masked versions don't show at once
        fields = list(super().get_fields(request, obj))
        if obj and "api_key" in fields:
            fields.remove("api_key")
        return fields

    def integration_api_key(self, obj: Integration) -> str:
        return obj.get_api_key_safe()
