from django.urls import path
from . import views

app_name = "training"

urlpatterns = [
    path("", views.train_result, name="train_result"),
    # path("api/training/<int:session_id>/", views.training_data_api, name="training_data_api",),
    path("data_api/<int:session_id>/", views.training_data_api, name="training_data_api",),
]
