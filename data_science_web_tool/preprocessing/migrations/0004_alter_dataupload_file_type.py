# Generated by Django 5.1.7 on 2025-05-03 07:33

from django.db import migrations
from django.db import models


class Migration(migrations.Migration):

    dependencies = [
        ("preprocessing", "0003_delete_datastatistics"),
    ]

    operations = [
        migrations.AlterField(
            model_name="dataupload",
            name="file_type",
            field=models.CharField(
                choices=[("csv", "csv"), ("yfinance", "yfinance")], max_length=32
            ),
        ),
    ]
