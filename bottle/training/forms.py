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
