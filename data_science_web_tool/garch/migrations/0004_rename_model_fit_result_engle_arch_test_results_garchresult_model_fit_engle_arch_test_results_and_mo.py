# Generated by Django 5.1.7 on 2025-06-28 11:36

from django.db import migrations
from django.db import models


class Migration(migrations.Migration):

    dependencies = [
        ("garch", "0003_garchresult_model_fit_result_engle_arch_test_results_and_more"),
    ]

    operations = [
        migrations.RenameField(
            model_name="garchresult",
            old_name="model_fit_result_engle_arch_test_results",
            new_name="model_fit_engle_arch_test_results",
        ),
        migrations.RenameField(
            model_name="garchresult",
            old_name="model_fit_result_result_ljung_box_test_results",
            new_name="model_fit_ljung_box_test_results_squared",
        ),
        migrations.AddField(
            model_name="garchresult",
            name="model_fit_result_ljung_box_test_results",
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="garchresult",
            name="raw_data_ljung_box_test_results_squared",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
