# trainings/forms.py
from django import forms
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError
from .models import TrainingSession


class TrainingSessionForm(forms.ModelForm):
    attachment = forms.FileField(
        label="세션 자료 첨부 (선택)",
        required=False,
        help_text="ZIP 혹은 7Z만 업로드 가능 (최대 100MB)",
        error_messages={
            "invalid": "유효하지 않은 파일입니다.",
            "required": "파일을 첨부해주세요.",
        },
        validators=[
            FileExtensionValidator(
                ["zip", "7z"], message="ZIP 혹은 7Z 파일만 첨부할 수 있습니다."
            )
        ],
        widget=forms.ClearableFileInput(attrs={"class": "form-control-file"}),
    )

    class Meta:
        model = TrainingSession
        # fields = ['title', 'description', 'attachment']
        fields = ["attachment"]




# training/forms.py 2025.08.10 ngins7512
class DataUploadForm(forms.Form):
    """YOLO 훈련 데이터 업로드 폼"""

    # 기본 정보
    model_name = forms.CharField(
        max_length=100,
        label="모델명",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "YOLOv8n"}
        ),
    )

    dataset_name = forms.CharField(
        max_length=100,
        label="데이터셋명",
        widget=forms.TextInput(
            attrs={"class": "form-control", "placeholder": "Custom Dataset"}
        ),
    )

    # 파일 업로드
    zip_file = forms.FileField(
        label="데이터셋 ZIP 파일",
        help_text="이미지와 라벨 파일이 포함된 ZIP 파일을 업로드하세요",
        widget=forms.FileInput(attrs={"class": "form-control", "accept": ".zip"}),
    )

    # 훈련 설정
    epochs = forms.IntegerField(
        label="에포크 수",
        initial=100,
        min_value=1,
        max_value=1000,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    batch_size = forms.IntegerField(
        label="배치 크기",
        initial=16,
        min_value=1,
        max_value=64,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    learning_rate = forms.FloatField(
        label="학습률",
        initial=0.01,
        min_value=0.0001,
        max_value=1.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.0001"}),
    )

    # image_size = forms.IntegerField(
    image_size = forms.ChoiceField(    
        label="이미지 크기",
        initial=640,
        choices=[
            (416, "416"),
            (512, "512"),
            (640, "640"),
            (800, "800"),
            (1024, "1024"),
        ],
        widget=forms.Select(attrs={"class": "form-control"}),
    )

    # 고급 설정
    optimizer = forms.ChoiceField(
        label="옵티마이저",
        choices=[
            ("SGD", "SGD"),
            ("Adam", "Adam"),
            ("AdamW", "AdamW"),
            ("RMSprop", "RMSprop"),
        ],
        initial="SGD",
        widget=forms.Select(attrs={"class": "form-control"}),
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

    patience = forms.IntegerField(
        label="조기 종료 patience",
        initial=10,
        min_value=1,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )

    description = forms.CharField(
        label="설명",
        required=False,
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