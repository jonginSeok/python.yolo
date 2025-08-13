# tasks.py

from celery import shared_task
from django.shortcuts import get_object_or_404
from session import TrainingSession



@shared_task
def start_training_async(session_id):
    # 여기에 실제 훈련 로직 작성
    print(f"세션 {session_id}에 대해 훈련 시작!")
    # 예: 모델 학습, 파일 처리, 로그 저장 등
    
    session = get_object_or_404(TrainingSession, id=session_id)

    if __name__ == "__main__":
        from ultralytics import YOLO

        # 기존 모델 불러오기 (COCO 학습됨)
        model = YOLO("yolo11n.pt")  # 각자의 경로 .to('cuda')

        # 2025.07.24 add
        model.train(
            data="bottle/media/datasets/" + session.dataset_name + "/extracted/data.yaml",
            epochs=session.current_epoch,     # 10->50->100
            imgsz=session.image_size,        # GPU 메모리에 따라 640, 512, 416 등으로 조절 가능해요.
            batch=session.batch_size,       # 16->16->8  # 메모리 문제로 배치 사이즈 줄임
            project="JonginSeok/dataset/result",
            name= session.dataset_name,
            verbose=False,   # 학습 과정 출력
            
            hsv_h=0.03,
            hsv_s=0.6,
            hsv_v=0.5,
            mosaic=1.0,
            fliplr=0.5
            
            # close_mosaic=10,
            # pretrained=True,
            # patience=10, # 정확도(es_metric)가 10번을 넘기면 그만
            # es_metric='metrics/mAP50-95(B)'   # mAP50' # old version
            # hyp='hyp.yaml'
        )
        
        print(f"세션 {session.id}에 대한 훈련 완료!")
        