from django.contrib.auth import views as auth_views
from django.urls import path
from . import views

urlpatterns = [
    # path("login/", views.login_view, name="accounts_login"),
    
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    # path('logout/', auth_views.LogoutView.as_view(next_page='logout'), name='logout'), # 기존 방식
    path('logout/', auth_views.LogoutView.as_view(template_name='registration/logout.html'), name='logout'),
]
