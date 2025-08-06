from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView


urlpatterns = [
    path("", views.main_view, name="main"),
    path("login/", views.login_view, name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
]
