import os
import sys
import glob
import json
import yaml
import django
import random
import shutil
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.utils
import plotly.graph_objs as go
import pandas as pd
import psycopg2

from datetime import date, datetime
from django.contrib import messages
from django.conf import settings
from django.core.mail import send_mail
from django.core.mail import EmailMultiAlternatives
from django.db import connection
from django.db.models import Avg
from django.db.utils import OperationalError
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.template.loader import render_to_string
from django.template import TemplateDoesNotExist
from django.utils.html import strip_tags
from django.utils import timezone
from django.utils.functional import SimpleLazyObject
from django.views.decorators.http import require_POST
from django.urls import reverse
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from zoneinfo import ZoneInfo

from .models import TrainingSession, ClassMetric, TrainingMetric
from .forms import DataUploadForm, DataSearchForm
from .tasks import start_training_async  # .delay()로 비동기 실행


# Django 환경 설정
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()


# 방법 A: 코드 상단에 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))



def create_loss_chart(session):
    """손실 차트 생성"""
    metrics = session.metrics.all()

    epochs = [m.epoch for m in metrics]
    
    if session.model_name == 'cnn':
        train_losses = [m.loss_total for m in metrics]
        val_losses = [m.loss_total for m in metrics]
    
    elif session.model_name == 'yolo11n':
        train_losses = [m.train_acc for m in metrics]
        val_losses = [m.val_acc for m in metrics]

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
    
    if session.model_name == 'cnn':
        map50_values = [m.map50 for m in metrics]
        map95_values = [m.map95 for m in metrics]
    elif session.model_name == 'yolo11n':
        map50_values = [m.train_acc for m in metrics]
        map95_values = [m.val_acc for m in metrics]

    trace1 = go.Scatter(
        x=epochs,
        y=map50_values,
        mode="lines",
        name="mAP@0.5" if session.model_name == 'yolo11n' else "train accuracy",
        line=dict(color="#f59e0b", width=2),
    )
    trace2 = go.Scatter(
        x=epochs,
        y=map95_values,
        mode="lines",
        name="mAP@0.5:0.95" if session.model_name == 'yolo11n' else "valid accuracy",
        line=dict(color="#ef4444", width=2),
    )

    layout = go.Layout(
        title="Mean Average Precision (mAP)" if session.model_name == 'yolo11n' else "Train/Valid Accuracy",
        xaxis=dict(title="Epoch"),
        yaxis=dict(title="mAP" if session.model_name == 'yolo11n' else "Accuracy"),
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



# 훈련 종료 이메일 알림 함수 (HTML+텍스트, 템플릿 우선, 폴백 제공)
def send_training_finished_email(request, session_id: int, success: bool = True, extra_msg: str = ""):
    """
    훈련 종료 시 사용자에게 이메일 알림(HTML + 텍스트)을 보낸다.
    - templates/emails/training_finished.html / .txt 가 있으면 템플릿 사용
    - 없으면 inline HTML/text 로 폴백
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

    # 시간 포맷 (KST)
    def _fmt(dt):
        if not dt:
            return "-"
        return timezone.localtime(dt, timezone=ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")

    status_kor = "성공" if success else "실패"

    # 세션별 상세 URL (절대경로 보장)
    try:
        relative = reverse("training:training_data_api", args=[session.id])
        base = (getattr(settings, "SITE_BASE_URL", "") or "").strip()
        if base:
            result_url = (base.rstrip("/") + relative)
        else:
            # 요청 객체에서 호스트를 사용해 절대 URL 생성
            try:
                result_url = request.build_absolute_uri(relative)
            except Exception:
                # 최후의 수단: 상대경로
                result_url = relative
    except Exception as e:
        print(f"[notify] reverse url error: {e}")
        result_url = ""

    ctx = {
        "success": success,
        "status_kor": status_kor,
        "session_id": session.id,
        "model_name": session.model_name,
        "version": session.version,
        "dataset_name": getattr(session, "dataset_name", None),
        "start_kst": _fmt(getattr(session, "start_time", None)),
        "end_kst": _fmt(getattr(session, "end_time", None)),
        "extra_msg": extra_msg,
        "result_url": result_url,
    }

    subject = f"[YOLO] 훈련 {status_kor} - {session.model_name} (#{session.id})"
    from_email = getattr(settings, "DEFAULT_FROM_EMAIL", None)
    to = [session.notify_email]

    # 템플릿 기반 렌더링 (HTML + 텍스트)
    # HTML 템플릿은 필수, TXT는 없으면 strip_tags로 대체
    try:
        html_body = render_to_string("training/emails/training_finished.html", ctx)
    except TemplateDoesNotExist as e:
        print(f"[notify] HTML template missing, no send: {e}")
        return
    except Exception as e:
        print(f"[notify] HTML template render error, no send: {e}")
        return

    try:
        text_body = render_to_string("training/emails/training_finished.txt", ctx)
    except TemplateDoesNotExist:
        # TXT 템플릿이 없으면 HTML에서 텍스트 추출
        text_body = strip_tags(html_body)
        print("[notify] TXT template missing, using strip_tags(html) as fallback.")
    except Exception as e:
        # 기타 렌더 에러도 텍스트는 폴백 생성
        text_body = strip_tags(html_body)
        print(f"[notify] TXT template render error, using fallback: {e}")

    msg = EmailMultiAlternatives(subject, text_body, from_email, to)
    msg.attach_alternative(html_body, "text/html")
    try:
        msg.send(fail_silently=False)
        print(f"[notify] sent HTML email to {session.notify_email} for session {session_id}")
    except Exception as e:
        print(f"[notify] email send error for session {session_id}: {e}")


# 학습 세션 입력 데이터 전송
def upload_dataset(request):
    """데이터셋 업로드 페이지"""

    # submit 하여 POST 방식으로 호출
    if request.method == "POST":
        form = DataUploadForm(request.POST, request.FILES)

        if form.is_valid():
            # 훈련 세션 생성(training_trainingsession)
            session = TrainingSession.objects.create(
                model_name=form.cleaned_data["model_name"],
                version=form.cleaned_data["version"],
                status=form.cleaned_data["status"],
                dataset_name=form.cleaned_data["dataset_name"],
                gpu_info=form.cleaned_data["gpu_info"],
                memory_info=form.cleaned_data["memory_info"],
                total_epochs=form.cleaned_data["total_epochs"],
                current_epoch=form.cleaned_data["current_epoch"],
                batch_size=form.cleaned_data["batch_size"],
                learning_rate=form.cleaned_data["learning_rate"],
                image_size=form.cleaned_data["image_size"],
                optimizer=form.cleaned_data["optimizer"],
                augmentation=form.cleaned_data["augmentation"],
                patience=form.cleaned_data["patience"],
                early_stopping=form.cleaned_data["early_stopping"],
                rotation_angle=form.cleaned_data["rotation_angle"],
                train_percent=form.cleaned_data["train_percent"],
                valid_percent=form.cleaned_data["valid_percent"],
                test_percent=form.cleaned_data["test_percent"],
                description=form.cleaned_data["description"],
                notify_method="email",
                notify_email=request.POST.get("notify_email_addr") or None,
                created_id=request.user,
                updated_id=request.user,
            )
            
            # 상태 변경 training:훈련
            # TrainingSession update
            # 방법 1: 객체를 가져와서 수정 후 .save() 사용
            # session = TrainingSession.objects.get(id=session.id) # 원하는 객체 가져오기
            # session.status="training"                    # 필드 수정
            # session.save()                               # 변경사항 저장
            # 장점: 모델의 save() 메서드가 호출되므로 커스텀 로직이 실행됨
            # 단점: 객체를 메모리에 로드해야 하므로 성능이 떨어질 수 있음

            # ******************************************************************
            # 파일 저장 및 처리
            # ******************************************************************
            zip_file = form.cleaned_data["zip_file"]

            # 업로드 디렉토리 생성
            dataset_path = os.path.join(settings.MEDIA_ROOT, "datasets", str(session.id))

            upload_dir = os.path.join(dataset_path, session.dataset_name)
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
                    image_files = [
                        f
                        for f in file_list
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                    ]
                    label_files = [f for f in file_list if f.lower().endswith(".txt")]

                    if not image_files:
                        messages.error(request, "이미지 파일이 ZIP에 포함되어 있지 않습니다.")
                        os.remove(zip_path)
                        return render(request, "training/upload.html", {"form": form})

                    if session.model_name == "yolo11n":
                        if not label_files:
                            messages.error(request, "라벨 파일(.txt)이 ZIP에 포함되어 있지 않습니다.")
                            os.remove(zip_path)
                            return render(request, "training/upload.html", {"form": form})

                    # ZIP 파일 압축 해제
                    extract_dir = os.path.join(upload_dir, "extracted")
                    zip_ref.extractall(extract_dir)
                    print(f" extract_dir:{extract_dir} upload_dir:{upload_dir}")

            except zipfile.BadZipFile:
                messages.error(request, "올바르지 않은 ZIP 파일입니다.")
                os.remove(zip_path)
                return render(request, "training/upload.html", {"form": form})
            # ******************************************************************
            # 파일 저장 및 처리
            # ******************************************************************
            
            # 파일 개수 세기
            target_dir = extract_dir  # 실제 경로로 수정하세요
            try:
                file_count = sum(
                    len(files) for _, _, files in os.walk(target_dir)
                )
                # return JsonResponse({'file_count': file_count})
                print(f"✅ 2.파일 개수:{file_count}")
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)

            # ******************************************************************
            # ClassMetric 담을 리스트 생성(training_classmetric) 시작
            # ******************************************************************
            class_names = []
            print(f"✅ 모델 명: {session.model_name}")

            # model_name 에 따른 ClassMetric 모델의 class_names 데이터 생성
            if "cnn" == session.model_name:
                class_names = request.POST.getlist("class_name")

            elif "yolo11n" == session.model_name:

                data_yaml_path = os.path.join(extract_dir, "data.yaml")
                print(f"✅ 욜로v11n data.yaml파일 path: {data_yaml_path}")

                # YOLO data.yaml 파일 읽기
                with open(data_yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                # 각 항목을 변수에 담기
                num_classes = data.get("nc")
                class_names = data.get("names")

                print("✅ 클래스들의 숫자:", num_classes)
                print("✅ 클래스명들:", class_names)

            # ClassMetric 모델 인스턴스 리스트 생성
            class_objects = [
                ClassMetric(
                    session_id=session.id,
                    index=i,  # 0부터 시작하는 인덱스
                    class_name=name,
                    created_id=request.user,
                    updated_at=None,
                    # updated_id=None,
                )
                for i, name in enumerate(class_names)
            ]
            # ClassMetric 한 번에 저장
            ClassMetric.objects.bulk_create(class_objects)
            
            # precision,recall,f1_score,instances
            
            # ******************************************************************
            # ClassMetric 담을 리스트 생성(training_classmetric) 종료
            # ******************************************************************

            # ******************************************************************
            # *************************** 증강 시작 ****************************
            # ******************************************************************

            def rotate_and_split_cnn_dataset(
                source_dir,
                output_dir,
                train_percent=70,
                valid_percent=20,
                test_percent=10,
                label_map=None,
            ):
                print(f"✅ 소스경로 source_dir:{source_dir} output_dir:{output_dir}")
                total_percent = train_percent + valid_percent + test_percent
                print(f"✅ 퍼센트 total:{total_percent} train:{train_percent} valid:{valid_percent} test:{test_percent}")
                assert (total_percent == 100), f"비율 합이 100이 되어야 합니다. 현재: {total_percent}"

                train_ratio = train_percent / 100
                valid_ratio = valid_percent / 100
                # test_ratio = test_percent / 100

                # Step 1: 클래스별 이미지 수집
                class_images = {}
                for class_name in label_map.keys():
                    print(f"✅ 소스 디렉토리:{source_dir} 클래스명:{class_name}")
                    class_path = os.path.join(source_dir, "train", class_name)
                    images = glob.glob(os.path.join(class_path, "*.*"))  # 모든 확장자 포함
                    class_images[class_name] = images
                    class_path = os.path.join(source_dir, "valid", class_name)
                    images = glob.glob(os.path.join(class_path, "*.*"))  # 모든 확장자 포함
                    class_images[class_name] = images

                # Step 2: train/valid/test 디렉토리 생성
                for split in ["train", "valid", "test"]:
                    for class_name in label_map.keys():
                        split_class_dir = os.path.join(output_dir, split, class_name)
                        os.makedirs(split_class_dir, exist_ok=True)

                # Step 3: 이미지 분할 및 복사
                for class_name, images in class_images.items():
                    random.shuffle(images)
                    total = len(images)
                    train_end = int(total * train_ratio)
                    valid_end = train_end + int(total * valid_ratio)

                    split_map = {
                        "train": images[:train_end],
                        "valid": images[train_end:valid_end],
                        "test": images[valid_end:],
                    }

                    for split, split_images in split_map.items():
                        for img_path in split_images:
                            filename = os.path.basename(img_path)
                            dest_path = os.path.join(output_dir, split, class_name, filename)
                            shutil.copy2(img_path, dest_path)

                print("✅ 데이터셋 분할 및 정리가 완료되었습니다.")

            print(f"✅ 회전 및 분할 CNN 데이터 증강 시작 : {session.augmentation}")
            if session.augmentation:
                print(f"✅ 회전 및 분할 CNN 데이터 증강 모델명: {session.model_name}")

                if "cnn" == session.model_name:
                    print("🔢 회전 및 분할 CNN 데이터셋 Start")

                    filtered_objects = ClassMetric.objects.filter(session_id=session.id)
                    label_map = {obj.class_name: obj.index for obj in filtered_objects}
                    print(f"✅ 회전 및 분할 CNN 데이터 레벨맵 :  {label_map} ")

                    rotate_and_split_cnn_dataset(
                        source_dir=extract_dir,
                        output_dir=upload_dir,
                        train_percent=session.train_percent,
                        valid_percent=session.valid_percent,
                        test_percent=session.test_percent,
                        label_map=label_map,
                    )
                    print("🔢 회전 및 분할 CNN 데이터셋 End")

                elif "yolo11n" == session.model_name:
                    print("🔢 회전 및 분할 YOLO 데이터셋 Start ")

                    rotate_and_split_yolo_dataset(
                        root_dir=extract_dir,
                        output_dir=upload_dir,
                        rotation_angle=session.rotation_angle,
                        rate_img=[
                            session.train_percent,
                            session.valid_percent,
                            session.test_percent,
                        ],
                    )

                    # 실행 yaml 파일 지정
                    data_yaml_path = os.path.join(upload_dir, "data.yaml")
                    print(f"✅ 회전 및 분할 YOLO 데이터 data.yaml path:  {data_yaml_path} ")
                    print("🔢 회전 및 분할 YOLO 데이터셋 End ")

                    try:
                        abs_base = Path(upload_dir).resolve()
                        train_images = (abs_base / "train" / "images").resolve()
                        val_images = (abs_base / "valid" / "images").resolve()

                        # Ensure directories exist (defensive)
                        if not train_images.exists():
                            print(f"✅ [data.yaml] missing train images dir: {train_images}")
                        if not val_images.exists():
                            print(f"✅ [data.yaml] missing val images dir: {val_images}")

                        # class_names may be list or dict; normalize to list
                        if isinstance(class_names, dict):
                            try:
                                # sort by numeric key if possible
                                names_list = [v for k, v in sorted(class_names.items(), key=lambda kv: int(kv[0]))]
                            except Exception:
                                # fallback to insertion order
                                names_list = list(class_names.values())
                        elif isinstance(class_names, list):
                            names_list = class_names
                        else:
                            names_list = []

                        abs_yaml_path = Path(data_yaml_path).resolve()
                        abs_yaml_path.parent.mkdir(parents=True, exist_ok=True)
                        data_yaml_payload = {
                            "path": abs_base.as_posix(),
                            "train": train_images.as_posix(),
                            "val": val_images.as_posix(),
                            "names": {i: n for i, n in enumerate(names_list)},
                        }

                        with abs_yaml_path.open("w", encoding="utf-8") as yf:
                            yaml.safe_dump(data_yaml_payload, yf, sort_keys=False, allow_unicode=True,)

                        # overwrite variable for downstream training call
                        data_yaml_path = abs_yaml_path.as_posix()
                        print(f"🗂️ [data.yaml] rewritten with absolute paths:")
                        print(f"✅ train={data_yaml_payload['train']}\n   val={data_yaml_payload['val']}")
                    except Exception as e:
                        print(f"✅ [data.yaml] rewrite error: {e}")

            else:
                print(f"✅ 증강 없음 extract_dir={extract_dir}\n   upload_dir={upload_dir}")
                copy_files_from_paths(extract_dir, upload_dir)
                # upload_dir = extract_dir
                print("⚠️ 회전 및 분할 CNN 데이터 증강 없음")

            print("✅ 데이터 증강 종료")

            # 압축해제 폴더 삭제
            shutil.rmtree(os.path.join(extract_dir), ignore_errors=True)

            # 파일 개수 세기
            target_dir = upload_dir  # 실제 경로로 수정하세요
            try:
                file_count = sum(
                    len(files) for _, _, files in os.walk(target_dir)
                )
                # return JsonResponse({'file_count': file_count})
                print(f"✅ 1.파일 개수:{file_count}")
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)
            
            # ******************************************************************
            # *************************** 증강 종료 ****************************
            # ******************************************************************

            # ******************************************************************
            # *************************** 실행 시작 ****************************
            # ******************************************************************
            print("✅ 여기서부터 실제 YOLO/CNN 훈련을 시작합니다")

            # ******************************************************************
            # ***************************  CNN 실행 ****************************
            # ******************************************************************
            # CNN 과 YOLO 분기
            if "cnn" == session.model_name:
                print("✅ CNN 모델 훈련 시작")

                class DatasetWithSize(Dataset):
                    def __init__(self, root_dir, label_map, transform=None):
                        self.samples = []  # 파일 경로와 라벨을 짝지어 리스트에 저장
                        self.transform = transform
                        self.label_map = label_map

                        # samples 리스트에 이미지파일 경로와 라벨을 모두 저장
                        for (class_name,label,) in label_map.items():
                            class_path = os.path.join(root_dir, class_name)
                            for fname in os.listdir(class_path):
                                if fname.lower().endswith((".jpg", ".png")):
                                    self.samples.append((os.path.join(class_path, fname), label))

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

                        return (image,size_features,torch.tensor(label, dtype=torch.long),)

                # CNN
                # 신경망을 정의할 때는 Batch 크기를 고려하지 않지만 Tensor 연산은 배치단위 병렬처리가 기본임
                # 신경망에 데이터를 전달할 때는 Batch 단위로 전달해야 하며 신경망 출력측에서도 배치단위로 출력된다
                # 개발자는 하나의 샘플을 기준으로 신경망 구조만 정의하면 되며,
                # 여러 샘플을 병렬로 처리하는 역할은 프레임워크(Pytorch, TensorFlow)가 자동으로 수행한다.
                class CNNWithSize(nn.Module):
                    def __init__(self, image_size):

                        # 신경망을 선언할 때는 Batch 크기를 고려하지 않으며, Tensor연산은 배치단위 병렬처리를 지원
                        super().__init__()

                        self.conv = nn.Sequential(
                            nn.Conv2d(3, 16, 3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Conv2d(32, 16, 3, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                        )

                        with torch.no_grad():
                            dummy_input = torch.zeros(1, 3, image_size, image_size)
                            dummy_output = self.conv(dummy_input)
                            self.flat_size = dummy_output.view(1, -1).size(1)

                        self.fc = nn.Sequential(
                            # +4는 for size info
                            nn.Linear(self.flat_size + 4, 128),
                            nn.ReLU(),
                            # 최종 출력이 1이면 Sigmoid연결, 2이면 Softmax연결
                            nn.Linear(128, len(class_names)),
                        )

                    # 순전파(forward) 정의
                    def forward(self, x, size_feats):
                        x = self.conv(x)
                        # Flatten x: (B, 56*56*32) 1차원 데이터가 Batch 만큼 리턴됨. -1은 자동으로 설정
                        x = x.view(x.size(0),-1,)
                        # 각 배치에 이미지 사이즈 정보 추가. 2번째 차원에 추가
                        x = torch.cat([x, size_feats], dim=1)
                        return self.fc(x)

                # Letterbox 클래스 선언
                class Letterbox:
                    def __init__(self, size, color=(128, 128, 128)):  # LetterBox색 지정(중간 회색)
                        self.size = size  # 정사각형 대상 사이즈 (ex: 224)
                        self.color = color  # 패딩 색 (회색)

                    def __call__(self, img):
                        # 원본 크기
                        iw, ih = img.size
                        scale = min(self.size / iw, self.size / ih)     # 이미지 폭, 높이 중 큰 것과 CNN입력 크기의 비 # scale :분모
                        nw, nh = int(iw * scale), int(ih * scale)       # 이미지 폭, 높이 중 큰 것을 CNN입력 크기에 맞춘다
                        # 리사이즈
                        img = img.resize((nw, nh), Image.BILINEAR)      # 이미지 크기를 CNN입력 크기로 변경 # BILINEAR:양선보강법
                                    # Image.BILINEAR:속도와 품질의 균형이 좋은 보강법. 특히 딥러닝용 이미지 전처리 등에서 자주 사용
                        # 패딩
                        new_img = Image.new("RGB", (self.size, self.size), self.color) # CNN입력 크기의 빈 이미지 생성
                        pad_left = (self.size - nw) // 2                # 좌우 여백의 크기 # //: 정수 반환
                        pad_top = (self.size - nh) // 2                 # 상하 여백의 크기 # 중간에 맞추기 위해
                        new_img.paste(img, (pad_left, pad_top))         # 빈 이미지에 실제 이미지를 붙여넣기
                        return new_img

                # Dataset 에 전달할 transform 생성(수동으로도 호출 가능함)
                transform = transforms.Compose(
                    [
                        Letterbox(int(session.image_size)),         # 이미지 비율 유지
                        transforms.ToTensor(),                      # 이미지를 PyTorch 텐서로 변환
                        transforms.Normalize([0.5] * 3, [0.5] * 3), # 빠르고 안정적인 학습을 위한 정규화(0~1 -> -1~1), (x-0.5)/0.5
                    ]
                )

                # 학습 루프
                BATCH_SIZE = int(session.batch_size)
                EPOCHS = int(session.current_epoch)
                LR = float(session.learning_rate)  # LR = 0.001  # 학습률

                # GPU 사용 여부 확인
                DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"✅ CNN 모델 훈련 DEVICE: {DEVICE}")

                data_path = (upload_dir if upload_dir else "media/datasets/0")  # 에러 방지
                label_map = {name: idx for idx, name in enumerate(class_names)}
                
                train_dataset = DatasetWithSize(data_path + "/train", label_map, transform=transform)
                valid_dataset = DatasetWithSize(data_path + "/valid", label_map, transform=transform)

                # DataLoader 생성
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

                model = CNNWithSize(int(session.image_size)).to(DEVICE)  # 모델 생성 및 GPU에 이동
                criterion = nn.CrossEntropyLoss()  # Softmax 포함

                print(f"✅ CNN 모델 훈련 session.optimizer: {session.optimizer}")

                optimizer_map = {
                    "SGD": lambda: optim.SGD(model.parameters(), lr=LR, momentum=0.9),
                    "Adam": lambda: optim.Adam(model.parameters(), lr=LR),
                    "AdamW": lambda: optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01),
                    "RMSprop": lambda: optim.RMSprop(model.parameters(), lr=LR, alpha=0.99),
                }

                optimizer = optimizer_map.get(session.optimizer, lambda: None)()
                print(f"✅ CNN 모델 훈련 optimizer: {optimizer}")

                train_acc_list, val_acc_list = [], []

                conn = connection
                cursor = conn.cursor()
                for epoch in range(EPOCHS):
                    model.train()
                    correct, total, loss_total = 0, 0, 0

                    # Batch 단위 로드
                    for x, size_feats, y in train_loader:
                        # image, label, size_feats GPU에 이동
                        x, y, size_feats = (x.to(DEVICE), y.to(DEVICE), size_feats.to(DEVICE),)
                        optimizer.zero_grad()

                        # __call__() -> forward()
                        outputs = model(x, size_feats)
                        loss = criterion(outputs, y)
                        loss.backward()  # 오류발생 2025.08.20
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
                            x, y, size_feats = (x.to(DEVICE), y.to(DEVICE), size_feats.to(DEVICE),)
                            outputs = model(x, size_feats)
                            # GPU에서 리턴된 값과 연산하는 대상 데이터 y
                            correct += (outputs.argmax(1) == y).sum().item()
                            total += y.size(0)

                    val_acc = correct / total
                    val_acc_list.append(val_acc)

                    print(f"✅ Epoch {epoch+1} | Loss: {loss_total:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                    cursor.execute(
                        """
                    INSERT INTO training_trainingmetric (session_id, epoch, loss_total, train_acc, val_acc, timestamp ,created_at, created_id, updated_at, updated_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
                            session.id,
                            epoch + 1,
                            loss_total,
                            train_acc,
                            val_acc,
                            datetime.now(ZoneInfo("Asia/Seoul")),
                            datetime.now(ZoneInfo("Asia/Seoul")),
                            str(request.user),
                            None,
                            None,
                        ),
                    )
                conn.commit()  # 데이터베이스에 저장
                conn.close()  # 데이터베이스 연결 종료

                # 모델 저장 (학습 루프 끝난 후)
                save_path = os.path.join(data_path, "result",)
                os.makedirs(save_path, exist_ok=True) # 폴더가 없으면 생성
                
                save_path = os.path.join(save_path, str(session.id) + "_cnn_with_size_letterbox_.pth")
                torch.save(model.state_dict(), save_path,)
                print(f"✅ CNN Model saved to {save_path}")

                # 이메일 전 end_time을 먼저 저장/반영
                end_now = timezone.now()
                TrainingSession.objects.filter(id=session.id).update(end_time=end_now)
                session.end_time = end_now
                # 이메일 알림 (성공)
                send_training_finished_email(request, session.id, success=True)

            # ******************************************************************
            # *************************** YOLO 실행 ***************************
            # ******************************************************************
            elif "yolo11n" == session.model_name:

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print("📊 True여야 GPU 사용 가능 :", torch.cuda.is_available())
                print(f"📊 사용 가능한 GPU({device}) 수:", torch.cuda.device_count())
                # 비동기 실행
                # Celery + Redis: 가장 강력하고 안정적인 방식
                # start_training_async.delay(session.id, upload_dir, data_yaml_path)

                # 기존 모델 불러오기 (COCO 학습됨)
                model = YOLO(session.model_name + ".pt")
                model.train(
                    data=str(Path(data_yaml_path).resolve().as_posix()),
                    epochs=session.current_epoch,
                    batch=session.batch_size,
                    imgsz=session.image_size,
                    optimizer=session.optimizer,
                    lr0=float(session.learning_rate),
                    weight_decay=0.01,
                    project=os.path.join(upload_dir, "result"),  # 저장 경로
                    name=session.model_name,
                    verbose=True,  # 학습 과정 출력
                    # 정확도(es_metric)가 10번을 넘기면 그만
                    patience=(session.patience if session.early_stopping else 0),
                )

                # 결과파일 읽어들이기
                df = pd.read_csv(os.path.join(upload_dir, "result", session.model_name, "results.csv",))

                # 모든 데이터를 저장 (예: 변수로)
                all_data = df.to_dict(orient="list")  # 열 기준으로 리스트로 저장
                first_key = next(iter(all_data))  # 첫 번째 key 가져오기
                size = len(all_data[first_key])  # 해당 key의 리스트 길이
                print(f"📝 모델 훈련 결과: all_data size:{size}")  # 디버깅용 출력
                
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
                    INSERT INTO training_trainingmetric (session_id, epoch, train_loss, val_loss, map50, map95, precision, recall, timestamp ,created_at, created_id, updated_at, updated_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            None,
                            None,
                        ),
                    )
                conn.commit()
                conn.close()

                # 이메일 전 end_time을 먼저 저장/반영
                end_now = timezone.now()
                TrainingSession.objects.filter(id=session.id).update(end_time=end_now)
                session.end_time = end_now
                # 이메일 알림 (성공)
                send_training_finished_email(request, session.id, success=True)
            # ******************************************************************
            # *************************** 실행 종료 ***************************
            # ******************************************************************

            # 방법 2: QuerySet.update() 사용
            TrainingSession.objects.filter(id=session.id).update(
                dataset_path=dataset_path,
                status="completed",
                notify_method="email",
                notify_email=request.POST.get("notify_email_addr"),
                config={
                    "total_epochs": session.total_epochs,
                    "current_epoch": session.current_epoch,
                    "batch_size": session.batch_size,
                    "learning_rate": session.learning_rate,
                    "image_size": session.image_size,
                    "optimizer": session.optimizer,
                    "augmentation": session.augmentation,
                    "early_stopping": session.early_stopping,
                    "patience": session.patience,
                    "image_count": len(image_files),
                    "label_count": len(label_files),
                    "end_time": datetime.now(ZoneInfo("Asia/Seoul")),
                },
                end_time=datetime.now(ZoneInfo("Asia/Seoul")),
                updated_at=datetime.now(ZoneInfo("Asia/Seoul")),
                updated_id=str(request.user),
            )
            # 장점: SQL UPDATE를 직접 실행하므로 빠름
            # 단점: save() 메서드나 pre_save/post_save 시그널이 호출되지 않음

            # messages.success(request, f"데이터셋이 성공적으로 업로드되었습니다. 훈련 세션 ID: {session.id}",)
            # return HttpResponse("훈련이 백그라운드에서 시작되었습니다.")
            return redirect("training:dashboard")
    else:
        form = DataUploadForm()

    return render(request, "training/upload.html", {"form": form})


def dashboard(request):
    """메인 대시보드 뷰"""

    try:
        # 최신 훈련 세션 가져오기
        latest_session = TrainingSession.objects.latest("created_at")
        latest_metrics = latest_session.metrics.last()

        class_metrics = latest_session.class_metrics.all()

        # 차트 데이터 생성
        loss_chart = create_loss_chart(latest_session)
        map_chart = create_map_chart(latest_session)

        # 성능 개선 계산 (이전 10개 에포크와 비교)
        metrics_count = latest_session.metrics.count()

        if metrics_count > 10:
            recent_avg = latest_session.metrics.order_by("-epoch")[:5].aggregate(
                Avg("map50")
            )["map50__avg"]
            old_avg = latest_session.metrics.order_by("-epoch")[5:10].aggregate(
                Avg("map50")
            )["map50__avg"]
            map_change = ((recent_avg - old_avg) / old_avg * 100) if old_avg else 0
        else:
            map_change = 0

        # 손실 변화 계산
        if metrics_count > 5:
            recent_loss = latest_session.metrics.order_by("-epoch")[:3].aggregate(
                Avg("train_loss")
            )["train_loss__avg"]
            old_loss = latest_session.metrics.order_by("-epoch")[3:6].aggregate(
                Avg("train_loss")
            )["train_loss__avg"]
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
        "metrics": latest_metrics,
        "class_metrics": class_metrics,
        "loss_chart": loss_chart,
        "map_chart": map_chart,
        "map_change": round(map_change, 1),
        "loss_change": round(loss_change, 1),
    }

    return render(request, "training/dashboard.html", context)


def training_data_api(request, session_id):
    """훈련 데이터 API"""
    session = get_object_or_404(TrainingSession, id=session_id)
    metrics = session.metrics.all()

    # 차트 데이터 생성
    loss_chart = create_loss_chart(session)
    map_chart = create_map_chart(session)
    
    data = {}
    
    if "cnn" == session.model_name:
        print("----------------- CNN ---------------")
        
        # 성능 개선 계산 (이전 10개 에포크와 비교)
        metrics_count = session.metrics.count()

        if metrics_count > 10:
            recent_avg = session.metrics.order_by("-epoch")[:5].aggregate(Avg("map50"))["map50__avg"]
            old_avg = session.metrics.order_by("-epoch")[5:10].aggregate(Avg("map50"))["map50__avg"]
            map_change = ((recent_avg - old_avg) / old_avg * 100) if old_avg else 0
        else:
            map_change = 0

        # 손실 변화 계산
        if metrics_count > 5:
            recent_loss = session.metrics.order_by("-epoch")[:3].aggregate(Avg("train_loss"))["train_loss__avg"]
            old_loss = session.metrics.order_by("-epoch")[3:6].aggregate(Avg("train_loss"))["train_loss__avg"]
            loss_change = ((old_loss - recent_loss) / old_loss * 100) if old_loss else 0
        else:
            loss_change = 0

        data = {
            "session": {
                "id": session.id,
                "model_name": session.model_name,
                "version": session.version,
                "status": session.status,
                "dataset_name": session.dataset_name,
                "gpu_info": session.gpu_info,
                "memory_info": session.memory_info,
                "total_epochs": session.total_epochs,
                "current_epoch": session.current_epoch,
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
            "loss_chart": loss_chart,
            "map_chart": map_chart,
            "map_change": round(map_change, 1),
            "loss_change": round(loss_change, 1),
        }
    elif "yolo11n" == session.model_name:
        print("----------------- YOLO ---------------")

        # 성능 개선 계산 (이전 10개 에포크와 비교)
        metrics_count = session.metrics.count()

        if metrics_count > 10:
            recent_avg = session.metrics.order_by("-epoch")[:5].aggregate(Avg("map50"))["map50__avg"]
            old_avg = session.metrics.order_by("-epoch")[5:10].aggregate(Avg("map50"))["map50__avg"]
            map_change = ((recent_avg - old_avg) / old_avg * 100) if old_avg else 0
        else:
            map_change = 0

        # 손실 변화 계산
        if metrics_count > 5:
            recent_loss = session.metrics.order_by("-epoch")[:3].aggregate(Avg("train_loss"))["train_loss__avg"]
            old_loss = session.metrics.order_by("-epoch")[3:6].aggregate(Avg("train_loss"))["train_loss__avg"]
            loss_change = ((old_loss - recent_loss) / old_loss * 100) if old_loss else 0
        else:
            loss_change = 0

        data = {
            "session": {
                "id": session.id,
                "model_name": session.model_name,
                "version": session.version,
                "status": session.status,
                "dataset_name": session.dataset_name,
                "gpu_info": session.gpu_info,
                "memory_info": session.memory_info,
                "total_epochs": session.total_epochs,
                "current_epoch": session.current_epoch,
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
            "loss_chart": loss_chart,
            "map_chart": map_chart,
            "map_change": round(map_change, 1),
            "loss_change": round(loss_change, 1),
        }
    print(f"훈련 데이터 API data:{data}")

    return render(request, "training/dashboard.html", data)


def training_sessions_list(request):
    """훈련 세션 목록"""
    # submit 하여 POST 방식으로 호출
    if request.method == "POST":

        form = DataSearchForm(request.POST or None)
        sessions = TrainingSession.objects.all()

        try:
            if form.is_valid():
                model = form.cleaned_data.get("model_name")
                id = form.cleaned_data.get("session_id")
                start = form.cleaned_data.get("start_date")
                end = form.cleaned_data.get("end_date")

                # 날짜 객체라면 datetime으로 변환
                if isinstance(start, date) and not isinstance(start, datetime):
                    start = datetime.combine(start, datetime.min.time())

                if isinstance(end, date) and not isinstance(end, datetime):
                    end = datetime.combine(end, datetime.max.time())

                # 이후 타임존 보정
                if start and timezone.is_naive(start):
                    start = timezone.make_aware(start)
                if end and timezone.is_naive(end):
                    end = timezone.make_aware(end)

                print(f"[훈련 세션 목록] 조회조건 model_name:{model} id:{id} start:{start} end:{end}")

                if start and end:
                    sessions = sessions.filter(start_time__range=(start, end))
                elif start:
                    sessions = sessions.filter(start_time__gte=start)
                elif end:
                    sessions = sessions.filter(end_time__lte=end)

                if model:
                    sessions = sessions.filter(model_name=model)
                if id:
                    sessions = sessions.filter(id=id)

            # return JsonResponse({"success": True})
        except TrainingSession.DoesNotExist:
            return JsonResponse({"success": False, "error": "세션을 찾을 수 없습니다."})

        # return redirect("training:sessions")
        return render(
            request,
            "training/sessions.html",
            {
                "form": form,
                "sessions": sessions.order_by("-created_at"),
            },
        )

    else:
        form = DataSearchForm()
        sessions = TrainingSession.objects.all().order_by("-created_at")
        context = {"sessions": sessions, "form": form}
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
    session_id = request.POST.get("session_id")
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
        return JsonResponse({"success": True})
    except TrainingSession.DoesNotExist:
        return JsonResponse({"success": False, "error": "세션을 찾을 수 없습니다."})


# User/dataset/
# ├── train/
# │   ├── images/
# │   └── labels/
# ├── valid/
# │   ├── images/
# │   └── labels/
# ├── test/
# │   ├── images/
# │   └── labels/
def rotate_and_split_yolo_dataset(root_dir, output_dir, rotation_angle, rate_img):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    all_images = []

    # 이미지 수집
    for dirpath, _, _ in os.walk(root_dir):
        for ext in image_extensions:
            all_images.extend(glob.glob(os.path.join(dirpath, ext)))

    print(f"🔍 총 이미지 수: {len(all_images)}")

    # 이미지 회전 및 저장
    rotated_images = []
    for img_path in all_images:
        try:
            img = Image.open(img_path).convert("RGBA")
            basename = os.path.splitext(os.path.basename(img_path))[0]

            for i in range(0, int(360 // rotation_angle)):
                rotation = i * rotation_angle
                ratio = f"rot{rotation}"

                rotated = img.rotate(rotation, resample=Image.BICUBIC, expand=True)
                white_bg = Image.new("RGBA", rotated.size, (255, 255, 255, 255))
                merged = Image.alpha_composite(white_bg, rotated)

                center_x, center_y = merged.size[0] // 2, merged.size[1] // 2
                original_size = img.size
                left = center_x - original_size[0] // 2
                top = center_y - original_size[1] // 2
                right = left + original_size[0]
                bottom = top + original_size[1]
                cropped = merged.crop((left, top, right, bottom)).convert("RGB")

                save_name = (
                    f"{basename}.jpg" if rotation == 0 else f"{basename}_{ratio}.jpg"
                )
                save_path = os.path.join(output_dir, "temp", "images", save_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cropped.save(save_path)

                # 라벨 복사
                label_path = os.path.join(
                    os.path.dirname(img_path), "..", "labels", f"{basename}.txt"
                )
                if os.path.exists(label_path):
                    label_name = (
                        f"{basename}.txt"
                        if rotation == 0
                        else f"{basename}_{ratio}.txt"
                    )
                    label_save_path = os.path.join(
                        output_dir, "temp", "labels", label_name
                    )
                    os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
                    shutil.copy(label_path, label_save_path)

                rotated_images.append(save_name)

        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")

    print(f"✅ 회전된 이미지 수: {len(rotated_images)}")

    # 🔀 데이터 셋 분할
    random.shuffle(rotated_images)
    total = len(rotated_images)
    train_count = int(total * rate_img[0] / 100)
    valid_count = int(total * rate_img[1] / 100)
    # test_count = total - train_count - valid_count # test_count 쓰임없음

    splits = {
        "train": rotated_images[:train_count],
        "valid": rotated_images[train_count : train_count + valid_count],
        "test": rotated_images[train_count + valid_count :],
    }

    for split_name, files in splits.items():
        for fname in files:
            # 이미지 이동
            src_img = os.path.join(output_dir, "temp", "images", fname)
            dst_img = os.path.join(output_dir, split_name, "images", fname)
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.move(src_img, dst_img)

            # 라벨 이동
            label_fname = os.path.splitext(fname)[0] + ".txt"
            src_label = os.path.join(output_dir, "temp", "labels", label_fname)
            dst_label = os.path.join(output_dir, split_name, "labels", label_fname)
            if os.path.exists(src_label):
                os.makedirs(os.path.dirname(dst_label), exist_ok=True)
                shutil.move(src_label, dst_label)

    # ✅ YAML 파일 복사
    yaml_files = glob.glob(os.path.join(root_dir, "*.yaml"))
    for yaml_path in yaml_files:
        try:
            shutil.copy(yaml_path, output_dir)
            print(f"📄 YAML 복사됨: {os.path.basename(yaml_path)}")
        except Exception as e:
            print(f"⚠️ YAML 복사 오류: {yaml_path} → {e}")

    print("🎉 데이터셋 분할 및 YAML 복사 완료")

    # 임시 폴더 삭제
    shutil.rmtree(os.path.join(output_dir, "temp"), ignore_errors=True)
    print("🎉 데이터셋 분할 완료: train / valid / test")


def copy_files_from_paths(source_path: str, target_path: str) -> dict:
    """
    source_path 하위의 모든 파일과 폴더를 target_path로 복사합니다.
    """
    if not os.path.exists(source_path):
        return {'status': 'error', 'message': f'원본 경로가 존재하지 않습니다: {source_path}'}

    try:
        for root, dirs, files in os.walk(source_path):
            relative_path = os.path.relpath(root, source_path)
            target_dir = os.path.join(target_path, relative_path)
            os.makedirs(target_dir, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_dir, file)
                shutil.copy2(src_file, dst_file)

        return {'status': 'success', 'message': '파일 복사가 완료되었습니다.'}

    except Exception as e:
        return {'status': 'error', 'message': str(e)}

