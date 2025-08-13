import os
import zipfile
import json
import plotly.utils
import plotly.graph_objs as go

# 우선 코드 붙여넣기 / 파일을 분리해서 parameters를 넘겨서 하면 좋을 듯
import torch

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.http import HttpResponse
from django.contrib import messages
from django.db.models import Q, Avg, Max, Min
from django.utils import timezone

from datetime import timedelta

from .models import TrainingSession, TrainingMetric, ClassMetric
from .forms import DataUploadForm
from app.tasks import start_training_async  # .delay()로 비동기 실행



def training_data_api(request, session_id):
    """훈련 데이터 API"""
    print(f'[training/views.py] training_data_api session_id:{session_id}')
    
    session = get_object_or_404(TrainingSession, id=session_id)
    metrics = session.metrics.all()

    data = {
        "session": {
            "id": session.id,
            "model_name": session.model_name,
            "version": session.version,
            "status": session.status,
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
        current_epoch = 50
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
    print(f'[trining/views.py] upload_dataset -----request.method:{request.method}')
    
    if request.method == "POST":
        form = DataUploadForm(request.POST, request.FILES)
        print(f'[trining/views.py] upload_dataset -----form.is_valid:{form.is_valid()}')
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

            # 훈련 세션 생성
            session = TrainingSession.objects.create(
                model_name=form.cleaned_data["model_name"],
                dataset_name=form.cleaned_data["dataset_name"],
                status=form.cleaned_data["status"], #"pending",
                total_epochs=form.cleaned_data["total_epochs"],
                current_epoch=form.cleaned_data["current_epoch"],
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
                    "epochs": form.cleaned_data["current_epoch"],
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
            )
            
            # ClassMetric 모델에 클래스 이름 저장
            class_names = request.POST.getlist("class_name")            
            print(f"[trining/views.py] class_nameslen: {len(class_names)}")
            
            # 모델 인스턴스 리스트 생성
            class_objects = [ClassMetric(session_id=session.id, class_name=name) for name in class_names]

            # 한 번에 저장
            ClassMetric.objects.bulk_create(class_objects)


            messages.success(request, f"데이터셋이 성공적으로 업로드되었습니다. 훈련 세션 ID: {session.id}",)

            # 여기서 실제 YOLO 훈련을 시작할 수 있습니다
            print("[trining/views.py] 여기서 실제 YOLO 훈련을 시작할 수 있습니다")
            # 우선 코드 붙여넣기 / 파일을 분리해서 parameters를 넘겨서 하면 좋을 듯
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print('True여야 GPU 사용 가능 :', torch.cuda.is_available())  # True여야 GPU 사용 가능
            print(f'사용 가능한 GPU({device}) 수:', torch.cuda.device_count())  # 사용 가능한 GPU 수
            
            
            
            # 1. Celery + Redis: 가장 강력하고 안정적인 방식
            # pip install celery redis           
            
            start_training_async.delay(session.id)  # .delay()로 비동기 실행
            # start_training_async(session.id)  # 백그라운드 작업으로 훈련 시작
            
            print(f"start_training_async({session.id})  # 백그라운드 작업으로 훈련 시작")
            return HttpResponse("훈련이 백그라운드에서 시작되었습니다.")

            # return redirect("training:dashboard")
    else:
        # print('Before:',form.errors)
        form = DataUploadForm()
        print('After:',form.errors)

    return render(request, "training/upload.html", {"form": form})


def training_sessions_list(request):
    """훈련 세션 목록"""
    sessions = TrainingSession.objects.all().order_by("-created_at")
    context = {"sessions": sessions}
    return render(request, "training/sessions.html", context)

