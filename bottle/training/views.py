import os
import zipfile
import json
import plotly.utils
import plotly.graph_objs as go
import torch

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.http import HttpResponse
from django.contrib import messages
from django.db.models import Q, Avg, Max, Min
from django.utils import timezone
from django.conf import settings

from ultralytics import YOLO
from datetime import timedelta

from .models import TrainingSession, TrainingMetric, ClassMetric
from .forms import DataUploadForm
from .tasks import start_training_async

# from training.tasks import start_training_async  # .delay()로 비동기 실행


def training_data_api(request, session_id):
    """훈련 데이터 API"""
    session = get_object_or_404(TrainingSession, id=session_id)
    metrics = session.metrics.all()

    data = {
        "session": {
            "id": session.id,
            "model_name": session.model_name,
            "version": session.version,
            "status": session.status,
            "dataset_name": session.dataset_name,
            "gpu_info": session.gpu_info,
            "memory_info": session.memory_info,
            # "total_epochs": session.total_epochs,
            # "current_epoch": session.current_epoch,
            "epochs": session.epochs,
            "learning_rate": session.learning_rate,
            "image_size": session.image_size,
            "optimizer": session.optimizer,
            "augmentation": session.augmentation,
            "early_stopping": session.early_stopping,
            "patience": session.patience,
            "description": session.description,
            "dataset_path": session.dataset_path,
            "config": session.config,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "progress": session.progress_percentage,
            "training_time": session.training_duration,
        },
        "metrics": [
            {
                "epoch": metric.epoch,
                "train_loss": metric.train_loss,
                "val_loss": metric.val_loss,
                "map50": metric.map50,
                "map95": metric.map95,
            }
            for metric in metrics
        ],
        "class_metrics": [
            {
                "class_name": cm.class_name,
                "precision": cm.precision,
                "recall": cm.recall,
                "f1_score": cm.f1_score,
                "instances": cm.instances,
            }
            for cm in session.class_metrics.all()
        ],
    }

    return JsonResponse(data)


def create_loss_chart(session):
    """손실 차트 생성"""
    metrics = session.metrics.all()

    epochs = [m.epoch for m in metrics]
    train_losses = [m.train_loss for m in metrics]
    val_losses = [m.val_loss for m in metrics]

    trace1 = go.Scatter(
        x=epochs,
        y=train_losses,
        mode="lines",
        name="Training Loss",
        line=dict(color="#8b5cf6", width=2),
    )

    trace2 = go.Scatter(
        x=epochs,
        y=val_losses,
        mode="lines",
        name="Validation Loss",
        line=dict(color="#06b6d4", width=2),
    )

    layout = go.Layout(
        title="Training & Validation Loss",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Loss"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=True,
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_map_chart(session):
    """mAP 차트 생성"""
    metrics = session.metrics.all()

    epochs = [m.epoch for m in metrics]
    map50_values = [m.map50 for m in metrics]
    map95_values = [m.map95 for m in metrics]

    trace1 = go.Scatter(
        x=epochs,
        y=map50_values,
        mode="lines",
        name="mAP@0.5",
        line=dict(color="#f59e0b", width=2),
    )

    trace2 = go.Scatter(
        x=epochs,
        y=map95_values,
        mode="lines",
        name="mAP@0.5:0.95",
        line=dict(color="#ef4444", width=2),
    )

    layout = go.Layout(
        title="Mean Average Precision (mAP)",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="mAP"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=True,
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_demo_data():
    """데모 데이터 생성"""

    class DemoSession:
        model_name = "YOLOv8n"
        version = "1.0.2"
        status = "training"
        dataset_name = "COCO 2017"
        gpu_info = "RTX 4090"
        memory_info = "24GB"
        # current_epoch = 50
        epoch = 50
        total_epochs = 100

        @property
        def progress_percentage(self):
            return 50

        @property
        def training_duration(self):
            return "2h 34m"

    class DemoMetrics:
        train_loss = 0.22
        val_loss = 0.22
        map50 = 0.88
        map95 = 0.63

    class DemoClassMetric:
        def __init__(self, class_name, precision, recall, f1_score, instances):
            self.class_name = class_name
            self.precision = precision
            self.recall = recall
            self.f1_score = f1_score
            self.instances = instances

    demo_class_metrics = [
        DemoClassMetric("person", 0.89, 0.91, 0.90, 1247),
        DemoClassMetric("car", 0.85, 0.87, 0.86, 892),
        DemoClassMetric("bicycle", 0.78, 0.82, 0.80, 456),
        DemoClassMetric("motorbike", 0.82, 0.79, 0.80, 234),
        DemoClassMetric("bus", 0.91, 0.88, 0.89, 156),
        DemoClassMetric("truck", 0.87, 0.85, 0.86, 203),
    ]

    return DemoSession(), DemoMetrics(), demo_class_metrics


def create_demo_loss_chart():
    """데모 손실 차트"""
    epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    train_losses = [0.8, 0.62, 0.48, 0.39, 0.34, 0.31, 0.28, 0.26, 0.24, 0.23, 0.22]
    val_losses = [0.75, 0.58, 0.45, 0.37, 0.32, 0.29, 0.27, 0.25, 0.24, 0.23, 0.22]

    trace1 = go.Scatter(
        x=epochs,
        y=train_losses,
        mode="lines",
        name="Training Loss",
        line=dict(color="#8b5cf6", width=2),
    )
    trace2 = go.Scatter(
        x=epochs,
        y=val_losses,
        mode="lines",
        name="Validation Loss",
        line=dict(color="#06b6d4", width=2),
    )

    layout = go.Layout(
        title="Training & Validation Loss",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="Loss"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=True,
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_demo_map_chart():
    """데모 mAP 차트"""
    epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    map50_values = [0.42, 0.56, 0.68, 0.74, 0.78, 0.81, 0.83, 0.85, 0.86, 0.87, 0.88]
    map95_values = [0.24, 0.32, 0.41, 0.47, 0.52, 0.55, 0.57, 0.59, 0.61, 0.62, 0.63]

    trace1 = go.Scatter(
        x=epochs,
        y=map50_values,
        mode="lines",
        name="mAP@0.5",
        line=dict(color="#f59e0b", width=2),
    )
    trace2 = go.Scatter(
        x=epochs,
        y=map95_values,
        mode="lines",
        name="mAP@0.5:0.95",
        line=dict(color="#ef4444", width=2),
    )

    layout = go.Layout(
        title="Mean Average Precision (mAP)",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="mAP"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=True,
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def dashboard(request):
    """메인 대시보드 뷰"""
    # 최신 훈련 세션 가져오기
    try:
        latest_session = TrainingSession.objects.latest("created_at")
        latest_metrics = latest_session.metrics.last()
        class_metrics = latest_session.class_metrics.all()

        # 차트 데이터 생성
        loss_chart = create_loss_chart(latest_session)
        map_chart = create_map_chart(latest_session)

        # 성능 개선 계산 (이전 10개 에포크와 비교)
        metrics_count = latest_session.metrics.count()
        if metrics_count > 10:
            recent_avg = latest_session.metrics.order_by("-epoch")[:5].aggregate(Avg("map50"))["map50__avg"]
            old_avg = latest_session.metrics.order_by("-epoch")[5:10].aggregate(Avg("map50"))["map50__avg"]
            map_change = ((recent_avg - old_avg) / old_avg * 100) if old_avg else 0
        else:
            map_change = 0

        # 손실 변화 계산
        if metrics_count > 5:
            recent_loss = latest_session.metrics.order_by("-epoch")[:3].aggregate(Avg("train_loss"))["train_loss__avg"]
            old_loss = latest_session.metrics.order_by("-epoch")[3:6].aggregate(Avg("train_loss"))["train_loss__avg"]
            loss_change = ((old_loss - recent_loss) / old_loss * 100) if old_loss else 0
        else:
            loss_change = 0

    except TrainingSession.DoesNotExist:
        # 데모 데이터 생성
        latest_session, latest_metrics, class_metrics = create_demo_data()
        loss_chart = create_demo_loss_chart()
        map_chart = create_demo_map_chart()
        map_change = 2.3
        loss_change = 12.5

    context = {
        "session": latest_session,
        "latest_metrics": latest_metrics,
        "class_metrics": class_metrics,
        "loss_chart": loss_chart,
        "map_chart": map_chart,
        "map_change": round(map_change, 1),
        "loss_change": round(loss_change, 1),
    }

    return render(request, "training/dashboard.html", context)


# 학습 세션 입력 데이터 전송
def upload_dataset(request):
    """데이터셋 업로드 페이지"""
    # submit 하여 POST 방식으로 호출
    if request.method == "POST":
        form = DataUploadForm(request.POST, request.FILES)
        print(f"# form.is_valid:{form.is_valid()}")

        if form.is_valid():
            # 파일 저장 및 처리
            zip_file = form.cleaned_data["zip_file"]

            # 업로드 디렉토리 생성
            upload_dir = os.path.join("media", "datasets", form.cleaned_data["dataset_name"])
            os.makedirs(upload_dir, exist_ok=True)

            # ZIP 파일 저장
            zip_path = os.path.join(upload_dir, zip_file.name)
            with open(zip_path, "wb+") as destination:
                for chunk in zip_file.chunks():
                    destination.write(chunk)

            # ZIP 파일 검증
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    file_list = zip_ref.namelist()

                    # 이미지와 라벨 파일 확인
                    image_files = [f for f in file_list if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
                    label_files = [f for f in file_list if f.lower().endswith(".txt")]

                    if not image_files:
                        messages.error(request, "이미지 파일이 ZIP에 포함되어 있지 않습니다.")
                        os.remove(zip_path)
                        return render(request, "training/upload.html", {"form": form})

                    if not label_files:
                        messages.error(request, "라벨 파일(.txt)이 ZIP에 포함되어 있지 않습니다.")
                        os.remove(zip_path)
                        return render(request, "training/upload.html", {"form": form})

                    # ZIP 파일 압축 해제
                    # extract_dir = os.path.join(upload_dir, "extracted")
                    zip_ref.extractall(upload_dir)

            except zipfile.BadZipFile:
                messages.error(request, "올바르지 않은 ZIP 파일입니다.")
                os.remove(zip_path)
                return render(request, "training/upload.html", {"form": form})

            # 훈련 세션 생성(training_trainingsession)
            session = TrainingSession.objects.create(
                model_name=form.cleaned_data["model_name"],
                dataset_name=form.cleaned_data["dataset_name"],
                status=form.cleaned_data["status"],  # "pending",
                # total_epochs=form.cleaned_data["total_epochs"],
                # current_epoch=form.cleaned_data["current_epoch"],
                epochs=form.cleaned_data["epochs"],
                batch_size=form.cleaned_data["batch_size"],
                learning_rate=form.cleaned_data["learning_rate"],
                image_size=form.cleaned_data["image_size"],
                optimizer=form.cleaned_data["optimizer"],
                augmentation=form.cleaned_data["augmentation"],
                early_stopping=form.cleaned_data["early_stopping"],
                patience=form.cleaned_data["patience"],
                description=form.cleaned_data["description"],
                dataset_path=upload_dir,
                config={
                    "epochs": form.cleaned_data["epochs"],
                    "batch_size": form.cleaned_data["batch_size"],
                    "learning_rate": form.cleaned_data["learning_rate"],
                    "image_size": form.cleaned_data["image_size"],
                    "optimizer": form.cleaned_data["optimizer"],
                    "augmentation": form.cleaned_data["augmentation"],
                    "early_stopping": form.cleaned_data["early_stopping"],
                    "patience": form.cleaned_data["patience"],
                    "image_count": len(image_files),
                    "label_count": len(label_files),
                },
                created_id=request.user,
            )

            # 모델 인스턴스 생성(training_classmetric)
            class_names = []

            print(f"# model_name: {form.cleaned_data["model_name"]}")

            if "CNN" == form.cleaned_data["model_name"]:
                # ClassMetric 모델에 클래스 이름 저장 /CNN
                class_names = request.POST.getlist("class_name")
                print("Class Names:", class_names)
                print(
                    f"# CNN class_names len: {len(class_names)}"
                )

            elif "YOLOv11n" == form.cleaned_data["model_name"]:
                # data.yaml 파일 읽어 저장 /YOLO
                data_yaml_path = os.path.join(upload_dir, "data.yaml")
                print(f"# YOLOv11n data_yaml_path: {data_yaml_path}")

                import yaml

                # YOLO data.yaml 파일 읽기
                with open(data_yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                # 각 항목을 변수에 담기
                train_path = data.get("train")
                val_path = data.get("val")
                test_path = data.get("test", None)  # test가 없을 수도 있으니 기본값 설정
                num_classes = data.get("nc")
                class_names = data.get("names")

                # 확인 출력
                print("Train Path:", train_path)
                print("Validation Path:", val_path)
                print("Test Path:", test_path)
                print("Number of Classes:", num_classes)
                print("Class Names:", class_names)

            # 모델 인스턴스 리스트 생성
            class_objects = [
                ClassMetric(
                    session_id=session.id,
                    index=i,  # 0부터 시작하는 인덱스
                    class_name=name,
                    created_id=request.user,
                )
                for i, name in enumerate(class_names)
            ]

            # 한 번에 저장
            ClassMetric.objects.bulk_create(class_objects)

            messages.success(request, f"데이터셋이 성공적으로 업로드되었습니다. 훈련 세션 ID: {session.id}",)

            # 여기서 실제 YOLO 훈련을 시작할 수 있습니다
            print("여기서 실제 YOLO 훈련을 시작할 수 있습니다")
            # 우선 코드 붙여넣기 / 파일을 분리해서 parameters를 넘겨서 하면 좋을 듯

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
            model = YOLO("yolo11n.pt")  # 각자의 경로 .to('cuda')
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

            # 결과파일 읽어들이기 bottle\media\datasets\bottle\result\26\YOLOv11n\results.csv
            import pandas as pd
            import psycopg2
            from datetime import datetime

            df = pd.read_csv(os.path.join(upload_dir, "result", str(session.id), session.model_name, "results.csv",))

            # 모든 데이터를 저장 (예: 변수로)
            all_data = df.to_dict(orient="list")  # 열 기준으로 리스트로 저장

            conn = psycopg2.connect(
                dbname="postgres",
                user="postgres",
                password="yolo11ai",
                host="postgres.cxg2cwseemwh.ap-northeast-2.rds.amazonaws.com",
                port="5432",
            )
            cursor = conn.cursor()

            for i in range(session.epochs):
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
                        datetime.now(),
                        datetime.now(),
                        str(request.user),
                    ),
                )

            conn.commit()
            conn.close()

            # 상태 변경 completed:완료
            session = TrainingSession.objects.update(
                status="completed",
                # updated_at=datetime.now(),
                # updated_id="system",
            )

            # return HttpResponse("훈련이 백그라운드에서 시작되었습니다.")
            return redirect("training:dashboard")

    else:
        # 초기값세팅 2. 뷰에서 동적으로  설정. 상황에 따라 기본값을 바꿀 수 있어요 (예: 로그인한 사용자 이름 등).
        form = DataUploadForm(initial={"dataset_name": "bottle"})

    return render(request, "training/upload.html", {"form": form})


def training_sessions_list(request):
    """훈련 세션 목록"""
    sessions = TrainingSession.objects.all().order_by("-created_at")
    context = {"sessions": sessions}
    return render(request, "training/sessions.html", context)


# train_loss = to_native(last_row["train/box_loss"])
def to_native(value):
    if hasattr(value, "item"):
        return value.item()  # numpy 타입 → Python 타입
    return value


# resolved_user = resolve_lazy(request.user)
from django.utils.functional import SimpleLazyObject


def resolve_lazy(obj):
    if isinstance(obj, SimpleLazyObject):
        return str(obj)  # 또는 obj.id, obj.username 등
    return obj
