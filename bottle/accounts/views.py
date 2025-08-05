#  from django.shortcuts import render

# Create your views here.
from django.contrib.auth.views import LoginView


class CustomLoginView(LoginView):
    template_name = "registration/login.html"


login_view = CustomLoginView.as_view()
