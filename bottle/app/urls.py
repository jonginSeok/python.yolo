# from django.contrib.auth import views as auth_views
from django.urls import path
from . import views
# from django.contrib.auth.views import LogoutView

# /static
urlpatterns = [
    path("", views.dashboard_view, name="dashboard"),
    path("api/dashboard-data/", views.dashboard_data, name="dashboard_data"),
    path("api/realtime-dashboard-data/", views.realtime_dashboard_data, name="realtime_dashboard_data"),
    
    # 새로운 API 엔드포인트 추가
    path("api/realtime-quality-data/", views.realtime_quality_data, name="realtime_quality_data"),
    path("api/quality-data/", views.quality_data, name="quality_data"),
    
    # path("login/", views.login_view, name="login"),
    # path("logout/", LogoutView.as_view(), name="logout"),
    
    # path("logout/", LogoutView.as_view(template_name="logout.html"), name="logout"),
    
    # path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    # # path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    # path('logout/', auth_views.LogoutView.as_view(template_name='logout.html'), name='logout'),
]