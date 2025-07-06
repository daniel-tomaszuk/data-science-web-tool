from django.contrib import admin
from garch.models import GarchResult


@admin.register(GarchResult)
class GarchResultAdmin(admin.ModelAdmin):
    """
    Admin for GarchResult
    """
