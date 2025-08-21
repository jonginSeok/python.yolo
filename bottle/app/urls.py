# 1. 앱 레벨 urls.py (현재 수정하고 있는 파일)
from django.urls import path
from . import views

urlpatterns = [
    # 메인 대시보드 페이지
    path('', views.dashboard_view, name='dashboard'),
    
    # API 엔드포인트  
    path('api/dashboard-data/', views.dashboard_data_api, name='dashboard_data_api'),
    
    # 다른 앱별 URL들이 있다면 여기에 추가
    # path('training/', views.training_view, name='training'),
]

# =====================================

# 2. 프로젝트 메인 urls.py (프로젝트명/urls.py)
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    path('training/', include('training.urls')),  # training 앱이 별도로 있다면
    path('', include('your_app_name.urls')),  # 현재 앱의 URLs 포함
]
"""