# training/forms.py
from django import forms
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError
from .models import TrainingSession

from datetime import datetime



class DataUploadForm(forms.Form):
    """YOLO 훈련 데이터 업로드 폼"""

    # 기본 정보
    model_name = forms.CharField(
        required=True,
        max_length=100,
        label="모델명",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "YOLOv8n"}
        ),
    )
    
    version = forms.CharField(
        required=True,
        max_length=20,
        label="버전",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "0.0.1"}
        ),
    )
    
    status = forms.ChoiceField(    
        required=True,
        label="상태",
        initial="training",
        choices=[
            ('training', '훈련 중'),
            ('completed', '완료'),
            ('failed', '실패'),
            ('paused', '일시정지'),
        ],
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    
    dataset_name = forms.CharField(
        required=True,
        max_length=100,
        label="데이터셋명",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "Custom Dataset"}
        ),
    )
    
    gpu_info = forms.CharField(
        required=True,
        max_length=100,
        label="GPU 정보",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "NVIDIA RTX 2070"}
        ),
    )
    memory_info = forms.CharField(
        required=True,
        max_length=50,
        label="메모리 정보",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "64GB"}
        ),
    )

    # 파일 업로드
    zip_file = forms.FileField(
        required=True,
        label="데이터셋 ZIP 파일",
        help_text="이미지와 라벨 파일이 포함된 ZIP 파일을 업로드하세요",
        widget=forms.FileInput(attrs={"class": "form-control", "accept": ".zip"}),
    )
    
    
    class_name = forms.CharField(
        required=True,
        max_length=50,
        label="클래스 정보",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "good or bad"}
        ),
    )
    
    total_epochs = forms.IntegerField(
        required=True,
        label="총 에포크",
        initial=100,
        min_value=1,
        max_value=1000,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    # 훈련 설정
    # current_epoch
    current_epoch = forms.IntegerField(
        required=True,
        label="현재 에포크",
        initial=20,
        min_value=1,
        max_value=1000,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    batch_size = forms.IntegerField(
        required=True,
        label="배치 크기",
        initial=4,
        min_value=1,
        max_value=64,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    learning_rate = forms.FloatField(
        required=True,
        label="학습률",
        initial=0.01,
        min_value=0.0001,
        max_value=1.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.0001"}),
    )

    # image_size = forms.IntegerField(
    image_size = forms.ChoiceField(    
        required=True,
        label="이미지 크기",
        initial=128,
        choices=[
            (128, "128"),
            (256, "256"),
            (512, "512"),
            (640, "640"),
            (800, "800"),
            (1024, "1024"),
        ],
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    # 고급 설정
    optimizer = forms.ChoiceField(
        required=True,
        label="옵티마이저",
        choices=[
            ("SGD", "SGD"),
            ("Adam", "Adam"),
            ("AdamW", "AdamW"),
            ("RMSprop", "RMSprop"),
        ],
        initial="Adam",
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    patience = forms.IntegerField(
        required=True,
        label="조기 종료 patience",
        initial=10,
        min_value=1,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    augmentation = forms.BooleanField(
        label="데이터 증강 사용",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )

    early_stopping = forms.BooleanField(
        label="조기 종료 사용",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )
    
    start_time = forms.DateTimeField(
        required=True,
        label="시작 시간",
        initial=datetime.now,
        widget=forms.DateTimeInput(
            attrs={"class": "form-control", "type": "datetime-local"}
        ),
    )
        
    end_time = forms.DateTimeField(
        required=False,
        label="종료 시간",
        widget=forms.DateTimeInput(
            attrs={"class": "form-control", "type": "datetime-local"}
        ),
    )

    # 설명 (선택사항)
    description = forms.CharField(
        required=False,
        label="설명",
        widget=forms.Textarea(
            attrs={
                "class": "form-control",
                "rows": 3,
                "placeholder": "훈련에 대한 설명을 입력하세요",
            }
        ),
    )
    
    def clean_zip_file(self):
        """ZIP 파일 검증"""
        
        zip_file = self.cleaned_data.get("zip_file")
        
        if zip_file:
            if not zip_file.name.endswith(".zip"):
                raise forms.ValidationError("ZIP 파일만 업로드 가능합니다.")
            if zip_file.size > 500 * 1024 * 1024:  # 500MB 제한
                raise forms.ValidationError("파일 크기는 500MB를 초과할 수 없습니다.")
            
        return zip_file
    