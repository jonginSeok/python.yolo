from django.urls import path
from . import views

app_name = "training"

urlpatterns = [
    path("upload/", views.upload_dataset, name="upload_dataset"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("sessions/", views.training_sessions_list, name="sessions_list"),
    path("delete/", views.delete_session, name="delete_session"),
    path("session/<int:session_id>/", views.training_data_api, name="training_data_api"),
]
