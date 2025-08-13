# from django.contrib.auth import views as auth_views
from django.urls import path
from . import views
# from django.contrib.auth.views import LogoutView

# /static
urlpatterns = [
    path("", views.main_view, name="main"),
    
    # path("login/", views.login_view, name="login"),
    # path("logout/", LogoutView.as_view(), name="logout"),
    
    # path("logout/", LogoutView.as_view(template_name="logout.html"), name="logout"),
    
    # path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    # # path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    # path('logout/', auth_views.LogoutView.as_view(template_name='logout.html'), name='logout'),

]
