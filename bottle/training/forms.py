# training/forms.py
from django import forms
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError
from .models import TrainingSession
from datetime import datetime, timedelta


class DataUploadForm(forms.Form):
    """YOLO 훈련 데이터 업로드 폼"""

    # 기본 정보
    model_name = forms.ChoiceField(
        required=True,
        label="모델",
        choices=[
            ("CNN", "CNN"),
            ("YOLOv11n", "YOLO v11n"),
        ],
        initial="YOLOv11n",  # 초기값세팅
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    version = forms.CharField(
        required=True,
        label="버전",
        max_length=20,
        initial="0.0.1",  # 초기값세팅
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "0.0.1"}),
    )

    status = forms.ChoiceField(
        required=True,
        label="상태",
        choices=[
            # ('begin', '훈련 시작'),
            ("training", "훈련 중"),
            ("completed", "완료"),
            ("failed", "실패"),
            ("paused", "일시정지"),
        ],
        initial="training",  # 초기값세팅
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    dataset_name = forms.CharField(
        required=True,
        max_length=100,
        label="데이터셋명",
        initial="bottle",  # 초기값세팅
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "Custom Dataset"}
        ),
    )

    gpu_info = forms.CharField(
        required=True,
        max_length=100,
        label="GPU 정보",
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "NVIDIA RTX 2070",
                "value": "NVIDIA RTX 2070",  # 초기값세팅 3. 의 로 직접  지정
            }
        ),
    )

    memory_info = forms.CharField(
        required=True,
        max_length=50,
        label="메모리 정보",
        widget=forms.TextInput(
            attrs={
                "class": "form-control",
                "placeholder": "64GB",
                "value": "64GB",  # 초기값세팅 3. 의 로 직접  지정
            }
        ),
    )

    # 파일 업로드
    zip_file = forms.FileField(
        required=True,
        label="데이터셋 ZIP 파일",
        help_text="이미지와 라벨 파일이 포함된 ZIP 파일을 업로드하세요",
        widget=forms.FileInput(
            attrs={"class": "form-control", "accept": ".zip"}),
    )

    class_name = forms.CharField(
        required=False,
        max_length=50,
        label="클래스 정보",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "good or bad"}
        ),
    )

    # 훈련 설정
    total_epochs = forms.IntegerField(
        required=True,
        label="총 에포크",
        min_value=1,
        max_value=1000,
        initial=100,  # 초기값세팅
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    current_epoch = forms.IntegerField(
        required=True,
        label="현재 에포크",
        min_value=1,
        max_value=1000,
        initial=50,  # 초기값세팅
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    batch_size = forms.IntegerField(
        required=True,
        label="배치 크기",
        min_value=1,
        max_value=64,
        initial=4,  # 초기값세팅
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    learning_rate = forms.FloatField(
        required=True,
        label="학습률",
        min_value=0.01,
        max_value=1.0,
        initial=0.01,  # 초기값세팅
        widget=forms.NumberInput(
            attrs={"class": "form-control", "step": "0.01"}),
    )

    # image_size = forms.IntegerField(
    image_size = forms.ChoiceField(
        required=True,
        label="이미지 크기",
        choices=[
            (128, "128"),
            (224, "224"),
            (256, "256"),
            (512, "512"),
            (640, "640"),
            (800, "800"),
            (1024, "1024"),
        ],
        initial=128,  # 초기값세팅
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
        initial="Adam",  # 초기값세팅
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    early_stopping = forms.BooleanField(
        label="조기 종료 사용",
        required=False,
        initial=False,  # 초기값세팅
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )

    patience = forms.IntegerField(
        required=True,
        label="조기 종료 patience",
        min_value=1,
        max_value=100,
        initial=10,  # 초기값세팅
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    augmentation = forms.BooleanField(
        label="데이터 증강 사용",
        required=False,
        initial=False,  # 초기값세팅
        widget=forms.CheckboxInput(attrs={"class": "form-check-input"}),
    )

    # 회전각도
    rotation_angle = forms.IntegerField(
        required=False,
        label="이미지 회전(각)",
        min_value=5,
        max_value=159,
        initial=100,  # 초기값세팅
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    train_percent = forms.IntegerField(
        required=False,
        label="train %",
        min_value=5,
        max_value=100,
        initial=70,  # 초기값세팅
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    valid_percent = forms.IntegerField(
        required=False,
        label="valid %",
        min_value=5,
        max_value=100,
        initial=20,  # 초기값세팅
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    test_percent = forms.IntegerField(
        required=False,
        label="test %",
        min_value=5,
        max_value=100,
        initial=10,  # 초기값세팅
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    start_time = forms.DateTimeField(
        required=True,
        label="시작 시간",
        initial=datetime.now,  # 초기값세팅
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
        initial="훈련에 대한 설명",  # 초기값세팅 1.  속성 사용 (가장 일반적인 방법)
        widget=forms.Textarea(
            attrs={
                "class": "form-control",
                "rows": 3,
                "placeholder": "훈련에 대한 설명을 입력하세요",
            }
        ),
    )
    
    
    
    
    
    
    
    
    
    

    # 초기값세팅 4.  메서드에서 동적 설정
    # • 	뷰에서 으로 생성하면 가 기본값으로 들어갑니다.
    # • 	더 복잡한 로직이 필요한 경우 유용해요.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.fields["memory_info"].initial = "64GB"  # 초기값세팅

    def clean_zip_file(self):
        """ZIP 파일 검증"""

        zip_file = self.cleaned_data.get("zip_file")

        if zip_file:
            if not zip_file.name.endswith(".zip"):
                raise forms.ValidationError("ZIP 파일만 업로드 가능합니다.")
            if zip_file.size > 500 * 1024 * 1024:  # 500MB 제한
                raise forms.ValidationError("파일 크기는 500MB를 초과할 수 없습니다.")

        return zip_file


class DataSearchForm(forms.Form):
    """YOLO 훈련 데이터 업로드 폼"""
    
    model_name = forms.ChoiceField(
        required=True,
        label="모델",
        choices=[
            ("CNN", "CNN"),
            ("YOLOv11n", "YOLO v11n"),
        ],
        initial="YOLOv11n",  # 초기값세팅
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    session_id = forms.IntegerField(
        required=False,
        label="세션ID",
        min_value=1,
        # max_value=1000,
        # initial=100,  # 초기값세팅
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "placeholder": "훈련에 대한 세션ID를 입력하세요",
        })
    )

    start_date = forms.DateField(
        required=False,
        label="시작 일자",
        # initial=datetime.now() - timedelta(days=7),  # 7일 전으로 초기값 설정
        # initial=date.today,
        widget=forms.DateInput(
            attrs={"class": "form-control", "type": "date"}
        ),
    )

    end_date = forms.DateField(
        required=False,
        label="종료 일자",
        # initial=datetime.now,
        # initial=date.today,
        widget=forms.DateInput(
            attrs={"class": "form-control", "type": "date"}
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
