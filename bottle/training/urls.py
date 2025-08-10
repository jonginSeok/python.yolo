from django.urls import path
from . import views

app_name = "training"

urlpatterns = [
    path("", views.training_list, name="training_list"),
    path("input/", views.training_input, name="training_input"),
    path("get/<int:session_id>/", views.training_output, name="training_output"),
]
