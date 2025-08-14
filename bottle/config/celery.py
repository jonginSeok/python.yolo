# config/celery.py
from __future__ import absolute_import

import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")

# Django 설정에서 CELERY 관련 설정을 가져옴
app.config_from_object("django.conf:settings", namespace="CELERY")

# 모든 앱의 tasks.py 자동 등록
app.autodiscover_tasks()