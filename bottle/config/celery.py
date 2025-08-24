# config/celery.py

# 1. Celery 설정 확인 (celery.py)
# from __future__ import absolute_import

import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# app = Celery("config")
app = Celery("python.yolo")

# Django 설정에서 CELERY 관련 설정을 가져옴
app.config_from_object("django.conf:settings", namespace="CELERY")

# 모든 앱의 tasks.py 자동 등록
app.autodiscover_tasks()
