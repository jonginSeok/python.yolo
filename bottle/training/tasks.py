# training/tasks.py
import os
import django
import time
import pandas as pd

from django.db import connection
from django.core.mail import send_mail
from celery import shared_task


# Django 환경 설정
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from training.models import TrainingSession  # 실제 모델 경로에 맞게 수정
from ultralytics import YOLO

@shared_task(bind=True)
def start_training_async(self, session_id, upload_dir, data_yaml_path):
    try:
        # 1. 세션 정보 조회
        session = TrainingSession.objects.get(id=session_id)
        # 예시: 세션에서 필요한 정보 추출
        model_name = session.model_name  # 예: 'YOLOv8n'
        data_path = session.data_path    # 데이터셋 경로
        epochs = session.current_epoch
        batch = session.batch_size
        imgsz = session.image_size
        optmz = session.optimizer
        lr = session.learning_rate
        early_stp = session.early_stopping
        patnc = session.patience

        # 2. YOLO 모델 준비
        model = YOLO(model_name+".pt")  # 예: 'yolov8n.pt' 등

        # 3. 학습 실행
        results = model.train(
            data=data_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            optimizer=optmz,
            lr0=lr,
            project=os.path.join(upload_dir, "result", ),  # 저장 경로
            name=model_name,
            exist_ok=True,
            verbose=True,
            # 정확도(es_metric) 10번 넘기면 stop
            patience=(patnc if early_stp else 0),
            device=0  # GPU 사용, CPU만 쓸 경우 'cpu'
        )
        
        # 결과파일 읽어들이기
        df = pd.read_csv(os.path.join(upload_dir, "result", session.model_name, "results.csv",))
        
        # 모든 데이터를 저장 (예: 변수로)
        all_data = df.to_dict(orient="list")  # 열 기준으로 리스트로 저장
        first_key = next(iter(all_data))  # 첫 번째 key 가져오기
        size = len(all_data[first_key])  # 해당 key의 리스트 길이
        print(f"모델 훈련 결과: all_data size:{size}")  # 디버깅용 출력
        
        conn = connection
        cursor = conn.cursor()

        for i in range(size):
            epoch = int(df.loc[i, "epoch"])
            train_loss = float(df.loc[i, "train/box_loss"])
            precision = float(df.loc[i, "metrics/precision(B)"])
            recall = float(df.loc[i, "metrics/recall(B)"])
            val_loss = float(df.loc[i, "val/box_loss"])
            map50 = float(df.loc[i, "metrics/mAP50(B)"])
            map95 = float(df.loc[i, "metrics/mAP50-95(B)"])

            cursor.execute(
                """
            INSERT INTO training_trainingmetric (session_id, epoch, train_loss, val_loss, map50, map95, precision, recall, timestamp ,created_at, created_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    session.id,
                    epoch,
                    train_loss,
                    val_loss,
                    map50,
                    map95,
                    precision,
                    recall,
                    datetime.now(ZoneInfo("Asia/Seoul")),
                    datetime.now(ZoneInfo("Asia/Seoul")),
                    str(request.user),
                ),
            )

        conn.commit()
        conn.close()

        # 이메일 알림 (성공)
        send_training_finished_email(session.id, success=True)

        # 4. 학습 결과를 세션에 저장 (예시)
        session.status = "done"
        session.save()

        return {"status": "success", "session_id": session_id}
    except Exception as e:
        # 에러 발생 시 상태 저장
        if 'session' in locals():
            session.status = "failed"
            session.save()
        return {"status": "failed", "error": str(e), "session_id": session_id}
    
    

# 훈련 종료 이메일 알림 함수 (재사용 가능)
def send_training_finished_email(
    session_id: int, success: bool = True, extra_msg: str = ""
):
    """
    훈련 종료 시 사용자에게 이메일 알림을 보낸다.
    session.notify_email 이 없으면 아무 것도 하지 않음.
    """
    try:
        session = TrainingSession.objects.get(id=session_id)
    except TrainingSession.DoesNotExist:
        print(f"[notify] session not found: {session_id}")
        return
    # 이메일 주소가 없으면 스킵
    if not getattr(session, "notify_email", None):
        print(f"[notify] no notify_email for session {session_id}, skip")
        return
    subject = "[YOLO] 훈련 완료" if success else "[YOLO] 훈련 실패"
    lines = [
        f"모델: {session.model_name} (v{session.version})",
        f"상태: {'성공' if success else '실패'}",
    ]
    if session.dataset_name:
        lines.append(f"데이터셋: {session.dataset_name}")
    if extra_msg:
        lines.append(f"메시지: {extra_msg}")
    # 시간 정보(있을 경우)
    if getattr(session, "start_time", None):
        lines.append(f"시작: {session.start_time}")
    if getattr(session, "end_time", None):
        lines.append(f"종료: {session.end_time}")
    message = "\n".join(lines)
    try:
        send_mail(
            subject=subject,
            message=message,
            from_email=getattr(settings, "DEFAULT_FROM_EMAIL", None),
            recipient_list=[session.notify_email],
            fail_silently=False,
        )
        print(f"[notify] sent email to {session.notify_email} for session {session_id}")
    except Exception as e:
        print(f"[notify] email send error for session {session_id}: {e}")
