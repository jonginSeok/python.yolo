# training/tasks.py
import os
from celery import shared_task
# from django.shortcuts import get_object_or_404
# from .models import TrainingSession, TrainingMetric, ClassMetric
# from ultralytics import YOLO

import time


@shared_task(bind=True)
def start_training_async(self, x, y):
    # 여기에 실제 훈련 로직 작성    # 예: 모델 학습, 파일 처리, 로그 저장 등
    time.sleep(5)
    return {'result': x + y, 'task_id': self.request.id}

# 7. Celery 워커 실행
# celery -A config worker -l info
# celery -A config worker --loglevel=info

# Flower는 웹 UI를 통해 작업 목록, 상태, 실행 시간, 실패 로그를 확인할 수 있습니다.
# celery -A config flower --port=5555



# 1. Celery 워커 실행 확인
# 터미널에서 Celery 워커를 실행했을 때, 로그에 이런 메시지가 떠야 정상 작동 중이야:
# Bash
# celery -A config worker --loglevel=info

# 정상 로그 예시:
# [INFO/MainProcess] Connected to redis://localhost:6379/0
# [INFO/MainProcess] Task training.tasks.start_training[abc123] received
# [INFO/ForkPoolWorker-1] Session 42 training complete

# 2. 태스크 호출 후 Redis 큐 확인
# Python
# from training.tasks import start_training
# start_training.delay(session.id)

# 3. 태스크 결과 확인 (선택 사항)
# Celery는 기본적으로 결과를 저장하지 않지만, 설정하면 결과도 확인할 수 있어.
# 설정 예시 (settings.py):
# Python
# CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# 결과 확인:
# Python
# result = start_training.delay(session.id)
# print(result.get(timeout=10))  # 결과를 기다렸다가 출력
# 단, .get()은 동기적으로 기다리는 거라서 테스트용으로만 쓰는 게 좋아.

# 실행
# (pytorch_env) PS C:\Users\ngins\Git\python.yolo\bottle> celery -A config worker --loglevel=info


Flower 웹 UI 인증 기본값은 비활성화이고, 개발·테스트 환경에서만 이 변수를 켜서 사용합니다.
설정 방법
1) 터미널에서 직접 설정
export FLOWER_UNAUTHENTICATED_API=true   # macOS / Linux
set FLOWER_UNAUTHENTICATED_API=true      # Windows CMD
$env:FLOWER_UNAUTHENTICATED_API="true"   # PowerShell
그리고 Flower 실행:
    celery -A myproject flower --port=5555

2) .env 파일에 추가 (Docker나 Compose 환경)
FLOWER_UNAUTHENTICATED_API=true