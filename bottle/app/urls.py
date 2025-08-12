from django.urls import path
from . import views


# /static
urlpatterns = [
    path("", views.main_view, name="main"),
]
