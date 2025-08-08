from django.urls import path
from . import views

app_name = "training"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("api/training/<int:session_id>/", views.training_data_api, name="training_data_api",),
]
