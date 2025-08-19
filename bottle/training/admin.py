# from django.contrib import admin

# Register your models here.

from django.contrib import admin
from .models import TrainingSession, TrainingMetric, ClassMetric


@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = [
        "model_name",
        "version",
        "status",
        # "current_epoch",
        # "total_epochs",
        "epochs",
        "progress_percentage",
        "created_at",
    ]
    list_filter = ["status", "model_name", "created_at"]
    search_fields = ["model_name", "version", "dataset_name"]
    readonly_fields = ["progress_percentage", "training_duration"]


@admin.register(TrainingMetric)
class TrainingMetricAdmin(admin.ModelAdmin):
    list_display = [
        "session",
        "epoch",
        "train_loss",
        "val_loss",
        "map50",
        "map95",
        "timestamp",
    ]
    list_filter = ["session", "timestamp"]
    search_fields = ["session__model_name"]


@admin.register(ClassMetric)
class ClassMetricAdmin(admin.ModelAdmin):
    list_display = [
        "session",
        "class_name",
        "precision",
        "recall",
        "f1_score",
        "instances",
    ]
    list_filter = ["session", "class_name"]
    search_fields = ["class_name", "session__model_name"]
