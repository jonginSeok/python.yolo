# training/tasks.py
import os
from celery import shared_task
from django.shortcuts import get_object_or_404
from .models import TrainingSession
from ultralytics import YOLO


@shared_task
def start_training_async(session_id):
    # 여기에 실제 훈련 로직 작성    # 예: 모델 학습, 파일 처리, 로그 저장 등
    print(f"[start_training_async] 세션 {session_id}에 대해 훈련 시작!")
    session = get_object_or_404(TrainingSession, id=session_id)
    session.status = "진행 중"
    session.save()


    # if __name__ == "__main__":
    #     from ultralytics import YOLO

    # 기존 모델 불러오기 (COCO 학습됨)
    model = YOLO("yolo11n.pt")
    
    model.train(
        # bottle/media/datasets/Bottle
        data=os.path.join("bottle/media/datasets/", session.dataset_name, "/data.yaml"),
        epochs=session.current_epoch,           # 10 -> 50 -> 100
        imgsz=session.image_size,               # GPU 메모리에 따라 640, 512, 416 등으로 조절 가능
        batch=session.batch_size,               # 16 -> 16 -> 8  # 메모리 문제로 배치 사이즈 줄임
        project="JonginSeok/dataset/result",
        name=session.dataset_name,
        verbose=True,                           # 학습 과정 출력
        hsv_h=0.03,
        hsv_s=0.6,
        hsv_v=0.5,
        mosaic=1.0,
        fliplr=0.5,

        
        
        # close_mosaic=10,
        # pretrained=True,
        # patience=10,                          # 정확도(es_metric)가 10번을 넘기면 그만
        # es_metric='metrics/mAP50-95(B)'       # mAP50' # old version
        # hyp='hyp.yaml'
    )

    print(f"세션 {session.id}에 대한 훈련 완료!")
    session.status = "완료"
    session.save()


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

# (pytorch_env) PS C:\Users\ngins\Git\python.yolo\bottle\config> celery -A config worker --loglevel=info
# Usage: celery [OPTIONS] COMMAND [ARGS]...
# Try 'celery --help' for help.

# Error: 
# Unable to load celery application.
# The module config was not found.