from django.urls import path
from . import views

app_name = "training"

urlpatterns = [
    # path("", views.training_list, name="training_list"),
    # path("input/", views.training_input, name="training_input"), # 삭제예정 ngins7512 2025.08.12
    # path("training_output/", views.training_output, name="training_output"),
    
    path("dashboard/", views.dashboard, name="dashboard"),
    
    # path('', views.training_list, name='training_list'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('sessions/', views.training_sessions_list, name='sessions_list'),
    path('api/training/<int:session_id>/', views.training_data_api, name='training_data_api'),
]
