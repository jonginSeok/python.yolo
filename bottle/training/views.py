import os
import zipfile
import json
import plotly.utils
import plotly.graph_objs as go
import torch
# CNN
import torch.nn as nn
import torch.optim as optim
import yaml
import random
import shutil
import pandas as pd
import psycopg2

from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST
from django.utils.functional import SimpleLazyObject
from django.http import JsonResponse
from django.contrib import messages
from django.db.models import Avg
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image

from .models import TrainingSession, ClassMetric, TrainingMetric
from .forms import DataUploadForm

from datetime import datetime
from zoneinfo import ZoneInfo


# from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
# from torchvision import datasets, transforms
# from torchvision import transforms
# from .tasks import start_training_async
# from training.tasks import start_training_async  # .delay()로 비동기 실행
# resolved_user = resolve_lazy(request.user)
# from .models import TrainingSession, TrainingMetric, ClassMetric
# from datetime import timedelta
# from django.utils import timezone 
# from django.conf import settings
# from django.http import HttpResponse
# from django.db.models import Q, Avg, Max, Min




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
        if form.is_valid():
            # 훈련 세션 생성(training_trainingsession) / 위치고정
            session = TrainingSession.objects.create(
                model_name=form.cleaned_data["model_name"],
                version=form.cleaned_data["version"],
                status=form.cleaned_data["status"],  # "pending",
                dataset_name=form.cleaned_data["dataset_name"],
                gpu_info=form.cleaned_data["gpu_info"],
                memory_info=form.cleaned_data["memory_info"],
                epochs=form.cleaned_data["epochs"],
                batch_size=form.cleaned_data["batch_size"],
                learning_rate=form.cleaned_data["learning_rate"],
                image_size=form.cleaned_data["image_size"],
                optimizer=form.cleaned_data["optimizer"],
                augmentation=form.cleaned_data["augmentation"],
                early_stopping=form.cleaned_data["early_stopping"],
                patience=form.cleaned_data["patience"],
                description=form.cleaned_data["description"],
                created_id=request.user,
            )

            # 파일 저장 및 처리
            zip_file = form.cleaned_data["zip_file"]

            # 업로드 디렉토리 생성 / 위치고정
            dataset_path = os.path.join("media", "datasets", str(session.id)) # media/datasets/34
            upload_dir = os.path.join(dataset_path, form.cleaned_data["dataset_name"]) # media/datasets/34/bottle
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
                    extract_dir = os.path.join(upload_dir, "extracted")  # media/datasets/34/bottle/extracted
                    zip_ref.extractall(extract_dir)
                    
            except zipfile.BadZipFile:
                messages.error(request, "올바르지 않은 ZIP 파일입니다.")
                os.remove(zip_path)
                return render(request, "training/upload.html", {"form": form})

            # TrainingSession update
            # 방법 1: 객체를 가져와서 수정 후 .save() 사용
            # session = TrainingSession.objects.get(id=1)  # 원하는 객체 가져오기
            # session.name = "Updated Name"               # 필드 수정
            # session.duration = 90
            # session.save()                              # 변경사항 저장
            # 장점: 모델의 save() 메서드가 호출되므로 커스텀 로직이 실행됨
            # 단점: 객체를 메모리에 로드해야 하므로 성능이 떨어질 수 있음

            # 방법 2: QuerySet.update() 사용
            TrainingSession.objects.filter(id=session.id).update(
                dataset_path=dataset_path,
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
            )
            # 장점: SQL UPDATE를 직접 실행하므로 빠름
            # 단점: save() 메서드나 pre_save/post_save 시그널이 호출되지 않음

            # ClassMetric 모델 인스턴스 생성(training_classmetric)
            class_names = []
            print(f"# model_name: {session.model_name}")

            # model_name 선택값에 따른 ClassMetric 모델에 클래스 데이터 생성
            if "CNN" == session.model_name:
                class_names = request.POST.getlist("class_name") # ClassMetric 모델에 클래스 이름 저장 /CNN
                print(f"# CNN Class_names len: {len(class_names)} Class Name:{class_names}")

            elif "YOLOv11n" == session.model_name:
                data_yaml_path = os.path.join(extract_dir, "data.yaml") # data.yaml 파일 읽어 저장 /YOLO
                print(f"# YOLOv11n data_yaml_path: {data_yaml_path}")

                # YOLO data.yaml 파일 읽기
                with open(data_yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                # 각 항목을 변수에 담기
                num_classes = data.get("nc")
                class_names = data.get("names")
                # 확인 출력
                print("Number of Classes:", num_classes)
                print("Class Names:", class_names)

            # ClassMetric 모델 인스턴스 리스트 생성
            class_objects = [
                ClassMetric(
                    session_id=session.id,
                    index=i,  # 0부터 시작하는 인덱스
                    class_name=name,
                    created_id=request.user,
                )
                for i, name in enumerate(class_names)
            ]
            # ClassMetric 한 번에 저장  
            ClassMetric.objects.bulk_create(class_objects)
            # messages.success(request, f"데이터셋이 성공적으로 업로드되었습니다. 훈련 세션 ID: {session.id}",)
            
            ######################################################################################
            # 데이터 증강 시작
            ######################################################################################
            print(f"데이터 증강 시작: {session.augmentation}")
            if "True" == session.augmentation:
                print("데이터 증강 시작")
                
                
                
                # CNN 과 YOLO 분기
                if "CNN" == session.model_name:
                    print("CNN 데이터 증강 시작")
                
                
                
                
                elif "YOLOv11n" == session.model_name:
                    print("YOLO 데이터 증강 시작")
                    
                    
                
            elif "False" == session.augmentation:
                print("데이터 증강 없음")

            print(f"데이터 증강 종료: {session.augmentation}")
            ######################################################################################
            # 데이터 증강 종료
            ######################################################################################

            print("여기서 실제 YOLO 훈련을 시작할 수 있습니다")

            # CNN 과 YOLO 분기
            if "CNN" == session.model_name:
                
                # CNN 모델 훈련 시작
                print("CNN 모델 훈련 시작")
                class DatasetWithSize(Dataset):
                    def __init__(self, root_dir, label_map, transform=None):
                        self.samples = []  # 파일 경로와 라벨을 짝지어 리스트에 저장
                        self.transform = transform
                        self.label_map = label_map

                        # samples 리스트에 이미지파일 경로와 라벨을 모두 저장
                        for class_name, label in label_map.items():  # ex: {'cat': 0, 'dog': 1}
                            class_path = os.path.join(root_dir, class_name)
                            for fname in os.listdir(class_path):
                                if fname.lower().endswith((".jpg", ".png")):
                                    self.samples.append((os.path.join(class_path, fname), label))  # (이미지 파일 경로, 0/1)

                    def __len__(self):
                        return len(self.samples)

                    def __getitem__(self, idx):
                        img_path, label = self.samples[idx]
                        image = Image.open(img_path).convert("RGB")

                        # 원본 이미지 크기 정보
                        orig_w, orig_h = image.size
                        area = orig_w * orig_h
                        aspect_ratio = orig_w / orig_h
                        size_features = torch.tensor([orig_w, orig_h, area, aspect_ratio], dtype=torch.float32)

                        if self.transform:
                            image = self.transform(image)

                        return image, size_features, label  # 이미지, 크기, 라벨 리턴


                # CNN
                # 신경망을 정의할 때는 Batch 크기를 고려하지 않지만 Tensor 연산은 배치단위 병렬처리가 기본임
                # 신경망에 데이터를 전달할 때는 Batch 단위로 전달해야 하며 신경망 출력측에서도 배치단위로 출력된다
                # 개발자는 하나의 샘플을 기준으로 신경망 구조만 정의하면 되며,
                # 여러 샘플을 병렬로 처리하는 역할은 프레임워크(Pytorch, TensorFlow)가 자동으로 수행한다.
                class CNNWithSize(nn.Module):
                    def __init__(self):
                        super().__init__()  # 신경망을 선언할 때는 Batch 크기를 고려하지 않으며 Tensor연산은 배치단위 병렬처리를 지원함
                        
                        self.conv = nn.Sequential(           # 3 channel, 16 filters, filter - size 3
                            nn.Conv2d(3, 16, 3, padding=1),  # 224x224x3 -> 224x224x16, padding=1은 1픽셀 추가하여 출력크기 유지
                            nn.ReLU(),
                            nn.MaxPool2d(2),                 # -> 112x112x16, 이미지 크기를 1/2로 축소(국소적 특징 요약)
                            nn.Conv2d(16, 32, 3, padding=1), # 112x112x16 -> 112x112x32
                            nn.ReLU(),
                            nn.MaxPool2d(2),                 # -> 56x56x32
                        )
                        
                        self.flat_size = 56 * 56 * 32
                        
                        self.fc = nn.Sequential(
                            nn.Linear(self.flat_size + 4, 128),  # +4 for size info
                            nn.ReLU(),
                            nn.Linear(128, 2),
                        )

                    def forward(self, x, size_feats):       # 순전파(forward) 정의
                        x = self.conv(x)                    # x: (B, 56, 56, 32)
                        x = x.view(x.size(0), -1)  # Flatten # x: (B, 56*56*32) 1차원 데이터가 Batch 만큼 리턴됨. -1은 자동으로 설정
                        x = torch.cat([x, size_feats], dim=1)  # 각 배치에 이미지 사이즈 정보 추가. 2번째 차원에 추가
                        return self.fc(x)


                # Letterbox 클래스 선언
                class Letterbox:
                    def __init__(self, size, color=(128, 128, 128)):  # LetterBox색 지정
                        self.size = size  # 정사각형 대상 사이즈 (ex: 224)
                        self.color = color  # 패딩 색 (회색)

                    def __call__(self, img):
                        # 원본 크기
                        iw, ih = img.size
                        scale = min(self.size / iw, self.size / ih)  # 이미지 폭,높이 중 큰 것과 CNN입력 크기의 비 # scale :분모
                        nw, nh = int(iw * scale), int(ih * scale)  # 이미지 폭,높이 중 큰 것을 CNN입력 크기에 맞춘다
                        # 리사이즈
                        img = img.resize((nw, nh), Image.BILINEAR)  # 이미지 크기를 CNN입력 크기로 변경 # BILINEAR:양선보강법
                        # Image.BILINEAR:속도와 품질의 균형이 좋은 보강법. 특히 딥러닝용 이미지 전처리 등에서 자주 사용
                        # 패딩
                        new_img = Image.new("RGB", (self.size, self.size), self.color)  # CNN입력 크기의 빈 이미지 생성
                        pad_left = (self.size - nw) // 2  # 좌우 여백의 크기 # //: 정수 반환
                        pad_top = (self.size - nh) // 2  # 상하 여백의 크기 # 중간에 맞추기 위해
                        new_img.paste(img, (pad_left, pad_top))  # 빈 이미지에 실제 이미지를 붙여넣기
                        return new_img


                # Dataset 에 전달할 transform 생성(수동으로도 호출 가능함)
                transform = transforms.Compose(
                    [
                        Letterbox(224),  # 이미지 비율 유지
                        # transforms.Resize((224, 224)),  # 이미지를 고정 크기로 설정
                        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
                        # 빠르고 안정적인 학습을 위한 정규화(0~1 -> -1~1), (x-0.5)/0.5
                        transforms.Normalize([0.5] * 3, [0.5] * 3),
                    ]
                )

                # 학습 루프
                BATCH_SIZE = session.batch_size  # BATCH_SIZE = 32  # 배치 크기
                EPOCHS = session.epochs  # EPOCHS = 10  # 에포크 수
                LR = session.learning_rate  # LR = 0.001  # 학습률
                # GPU 사용 여부 확인                
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                data_path = upload_dir if upload_dir else "media/datasets/0" # 에러 방지 "cuda" if torch.cuda.is_available() else "cpu"
                label_map = {name: idx for idx, name in enumerate(class_names)} # {"BAD": 0, "GOOD": 1}  # class_names = ['BAD', 'GOOD']
                train_dataset = DatasetWithSize(data_path + "/train", label_map, transform=transform)
                valid_dataset = DatasetWithSize(data_path + "/val", label_map, transform=transform)
                # DataLoader 생성
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
                model = CNNWithSize().to(DEVICE)  # 모델 생성 및 GPU에 이동
                criterion = nn.CrossEntropyLoss()  # Softmax 포함

                print("CNN 모델 훈련 시작")
                print(f"CNN 모델 훈련 optimizer: {session.optimizer}")
                if "SGD" == session.optimizer:
                    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

                elif "Adam" == session.optimizer:
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    
                elif "AdamW" == session.optimizer:
                    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

                elif "RMSprop" == session.optimizer:
                    optimizer = optim.RMSprop(model.parameters(), lr=LR, alpha=0.99)

                train_acc_list, val_acc_list = [], []
                
                conn = psycopg2.connect(
                    dbname="postgres",
                    user="postgres",
                    password="yolo11ai",
                    host="postgres.cxg2cwseemwh.ap-northeast-2.rds.amazonaws.com",
                    port="5432",
                )
                cursor = conn.cursor()

                for epoch in range(EPOCHS):
                    model.train()
                    correct, total, loss_total = 0, 0, 0
                    for x, size_feats, y in train_loader:  # Batch 단위 로드
                        x, y, size_feats = (x.to(DEVICE), y.to(DEVICE), size_feats.to(DEVICE),)  # image, label, size_feats GPU에 이동
                        optimizer.zero_grad()
                        outputs = model(x, size_feats)  # __call__() -> forward()
                        loss = criterion(outputs, y)
                        loss.backward()
                        optimizer.step()
                        loss_total += loss.item()
                        correct += (outputs.argmax(1) == y).sum().item()
                        total += y.size(0)  # 라벨의 Batch 샘플 수. 총 샘플 수 합산
                        
                    train_acc = correct / total
                    train_acc_list.append(train_acc)  # epoch 정확도 리스트에 저장

                    # 검증
                    model.eval()
                    correct, total = 0, 0
                    with torch.no_grad():
                        for x, size_feats, y in valid_loader:
                            x, y, size_feats = x.to(DEVICE), y.to(DEVICE), size_feats.to(DEVICE)  #
                            outputs = model(x, size_feats)  # GPU에서 작동하는 모델이 리턴하는 값과 연산하는 대상 데이터도 GPU에 존재해야 한다
                            correct += ((outputs.argmax(1) == y).sum().item())  # GPU에서 리턴된 값과 연산하는 대상 데이터 y
                            total += y.size(0)
                            
                    val_acc = correct / total
                    val_acc_list.append(val_acc)

                    print(f"Epoch {epoch+1} | Loss: {loss_total:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

                    cursor.execute(
                        """
                    INSERT INTO training_trainingmetric (session_id, epoch, loss_total, train_acc, val_acc, timestamp ,created_at, created_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
                            session.id,
                            epoch,
                            loss_total,
                            train_acc,
                            val_acc,
                            datetime.now(ZoneInfo("Asia/Seoul")),
                            datetime.now(ZoneInfo("Asia/Seoul")),
                            str(request.user),
                        ),
                    )

                conn.commit()  # 데이터베이스에 저장
                conn.close()  # 데이터베이스 연결 종료

                # 모델 저장 (학습 루프 끝난 후)
                save_path = os.path.join("media", "datasets", upload_dir, str(session.id), "result", "cnn_with_size_letterbox.pth") 
                torch.save(model.state_dict(), save_path)
                print(f"CNN Model saved to {save_path}")

            elif session.model_name == "YOLOv11n":
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                print("True여야 GPU 사용 가능 :", torch.cuda.is_available())  # True여야 GPU 사용 가능
                print(f"사용 가능한 GPU({device}) 수:", torch.cuda.device_count())  # 사용 가능한 GPU 수

                # Celery + Redis: 가장 강력하고 안정적인 방식
                # result = start_training_async.delay(3, 7)  # 비동기 실행
                # return JsonResponse({'task_id': result.id}) # Celery + Redis

                # 기존 모델 불러오기 (COCO 학습됨)
                model = YOLO("yolo11n.pt")  # 각자의 경로 .to('cuda')
                model.train(
                    data=data_yaml_path,  # bottle\media\datasets\bottle\data.yaml
                    epochs=session.epochs,
                    imgsz=session.image_size,  # GPU 메모리에 따라 640, 512, 416 등으로 조절 가능
                    batch=session.batch_size,  # 메모리 문제로 배치 사이즈 줄임
                    optimizer=session.optimizer,  # 옵티마이져
                    lr0=float(session.learning_rate),
                    weight_decay=0.01,
                    project=os.path.join(upload_dir, "result", str(session.id)), # 저장 경로
                    name=session.model_name,     # yolo이름
                    verbose=True,                # 학습 과정 출력
                    patience=session.patience if session.early_stopping else 0 , # 정확도(es_metric)가 10번을 넘기면 그만
                )

                # 결과파일 읽어들이기 ex) bottle\media\datasets\bottle\result\26\YOLOv11n\results.csv


                df = pd.read_csv(os.path.join(upload_dir,"result",str(session.id),session.model_name,"results.csv",))

                # 모든 데이터를 저장 (예: 변수로)
                all_data = df.to_dict(orient="list")  # 열 기준으로 리스트로 저장
                first_key = next(iter(all_data))  # 첫 번째 key 가져오기
                size = len(all_data[first_key])  # 해당 key의 리스트 길이


                
                print(f"모델 훈련 결과: {all_data}  size:{size}")  # 디버깅용 출력

                conn = psycopg2.connect(
                    dbname="postgres",
                    user="postgres",
                    password="yolo11ai",
                    host="postgres.cxg2cwseemwh.ap-northeast-2.rds.amazonaws.com",
                    port="5432",
                )
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

                # 상태 변경 completed:완료
                session = TrainingSession.objects.update(
                    status="completed",
                    # updated_at=datetime.now(ZoneInfo("Asia/Seoul")),
                    # updated_id="system",
                )

            # return HttpResponse("훈련이 백그라운드에서 시작되었습니다.")
            return redirect("training:dashboard")

    else:
        # 초기값세팅 2. 뷰에서 동적으로  설정. 상황에 따라 기본값을 바꿀 수 있어요 (예: 로그인한 사용자 이름 등).
        form = DataUploadForm()

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


def resolve_lazy(obj):
    if isinstance(obj, SimpleLazyObject):
        return str(obj)  # 또는 obj.id, obj.username 등
    return obj



@require_POST
def delete_session(request):
    session_id = request.POST.get('session_id')
    try:
        session = TrainingSession.objects.get(id=session_id)
        session.delete()
        # 관련된 메트릭과 클래스 메트릭도 삭제
        TrainingMetric.objects.filter(session_id=session_id).delete()
        ClassMetric.objects.filter(session_id=session_id).delete()
        # 추가로 세션에 관련된 파일이나 디렉토리도 삭제할 수 수 있습니다.
        dataset_path = session.dataset_path
        if dataset_path and os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)  # 디렉토리와 그 안의 모든 파일 삭제
        
        # 성공적으로 삭제되었음을 알리는 JSON 응답        
        return JsonResponse({'success': True})
    except TrainingSession.DoesNotExist:
        return JsonResponse({'success': False, 'error': '세션을 찾을 수 없습니다.'})
