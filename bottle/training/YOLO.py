import os
import torch

from ultralytics import YOLO

from django.shortcuts import render, redirect, get_object_or_404
from .models import TrainingSession, TrainingMetric, ClassMetric


def RunYOLO(upload_dir, session_id):

    session = get_object_or_404(TrainingSession, id=session_id)
    metrics = session.metrics.all()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("True여야 GPU 사용 가능 :", torch.cuda.is_available())  # True여야 GPU 사용 가능
    print(f"사용 가능한 GPU({device}) 수:", torch.cuda.device_count())  # 사용 가능한 GPU 수

    # Celery + Redis: 가장 강력하고 안정적인 방식
    # result = start_training_async.delay(3, 7)  # 비동기 실행
    # print(f"# 백그라운드 작업으로 훈련 시작 start_training_async({session.id}) ")
    # return JsonResponse({'task_id': result.id}) # Celery + Redis

    data_yaml_path = os.path.join(upload_dir, "data.yaml")
    print(f" data_yaml_path : [{data_yaml_path}]")

    # 기존 모델 불러오기 (COCO 학습됨)
    model = YOLO("yolo11n.pt")  # 각자의 경로 .to('cuda')  yoloModel

    model.train(
        data=data_yaml_path,  # bottle\media\datasets\bottle\data.yaml
        epochs=session.epochs,
        imgsz=session.image_size,  # GPU 메모리에 따라 640, 512, 416 등으로 조절 가능
        batch=session.batch_size,  # 메모리 문제로 배치 사이즈 줄임
        project=os.path.join(upload_dir, "result", str(session.id)),
        name=session.model_name,
        verbose=True,  # 학습 과정 출력
        patience=session.patience,  # 정확도(es_metric)가 10번을 넘기면 그만
        # close_mosaic=10,
        # pretrained=True,
        # es_metric='metrics/mAP50-95(B)'   # mAP50' # old version
    )
    
