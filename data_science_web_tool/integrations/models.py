from django.db import models


class Integration(models.Model):
    name = models.CharField(max_length=256, help_text="Name of the integration")
    description = models.TextField(
        null=True,
        blank=True,
        help_text="Additional information about the integration API.",
    )
    api_key = models.CharField(
        null=True, blank=True, help_text="The API key for the integration."
    )
    url = models.URLField(
        null=True, blank=True, help_text="The URL for the integration."
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def get_api_key_safe(self):
        value = "****"
        if len(self.api_key) > 4:
            value = self.api_key[:2] + "*" * (len(self.api_key) - 4) + self.api_key[-2:]
        return value

    def __str__(self) -> str:
        return f"Integration with API `{self.name}`"
