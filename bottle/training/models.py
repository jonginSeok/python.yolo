from django.db import models
from django.utils import timezone

class TrainingSession(models.Model):
    STATUS_CHOICES = [
        ('training', '훈련 중'),
        ('completed', '완료'),
        ('failed', '실패'),
        ('paused', '일시정지'),
    ]
    
    model_name = models.CharField(max_length=100, verbose_name="모델명")
    version = models.CharField(max_length=20, verbose_name="버전")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='training', verbose_name="상태")
    dataset_name = models.CharField(max_length=100, verbose_name="데이터셋")
    gpu_info = models.CharField(max_length=100, verbose_name="GPU 정보")
    memory_info = models.CharField(max_length=50, verbose_name="메모리 정보")
    
    # zip_file = models.FileField(upload_to='trainings/%Y/%m/%d/',verbose_name="데이터셋 ZIP 파일")
    
    
    
    epochs = models.IntegerField(default=100, verbose_name="총 에포크")
    current_epoch = models.IntegerField(default=0, verbose_name="현재 에포크")    
    batch_size = models.IntegerField(default=1, verbose_name="배치 크기")
    learning_rate = models.FloatField(default=0.01, verbose_name="학습률")    
    image_size = models.IntegerField(default=640, verbose_name="이미지 크기")    
    optimizer = models.CharField(default="Adam", verbose_name="옵티마이저")
    augmentation = models.BooleanField(default=True, verbose_name="데이터 증강")
    early_stopping = models.BooleanField(default=True, verbose_name="조기 종료 사용")
    patience = models.IntegerField(default=10, verbose_name="조기 종료 patience")
    description = models.CharField(max_length=16000, verbose_name="설명")
    
    dataset_path = models.CharField(max_length=6000, verbose_name="데이터셋 경로")
    config = models.CharField(max_length=16000, verbose_name="설정정보 ")
    
    start_time = models.DateTimeField(default=timezone.now, verbose_name="시작 시간")
    end_time = models.DateTimeField(null=True, blank=True, verbose_name="종료 시간")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
        
    def __str__(self):
        return f"{self.model_name} v{self.version}"
    
    @property
    def progress_percentage(self):
        if self.epochs == 0:
            return 0
        return min((self.current_epoch / self.epochs) * 100, 100)
    
    @property
    def training_duration(self):
        if self.end_time:
            duration = self.end_time - self.start_time
        else:
            duration = timezone.now() - self.start_time
        
        hours = duration.total_seconds() // 3600
        minutes = (duration.total_seconds() % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

class TrainingMetric(models.Model):
    session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='metrics')
    epoch = models.IntegerField(verbose_name="에포크")
    train_loss = models.FloatField(verbose_name="훈련 손실")
    val_loss = models.FloatField(verbose_name="검증 손실")
    map50 = models.FloatField(verbose_name="mAP@0.5")
    map95 = models.FloatField(verbose_name="mAP@0.5:0.95")
    precision = models.FloatField(default=0.0, verbose_name="정밀도")
    recall = models.FloatField(default=0.0, verbose_name="재현율")
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['epoch']
        unique_together = ['session', 'epoch']
    
    def __str__(self):
        return f"Epoch {self.epoch} - {self.session.model_name}"

class ClassMetric(models.Model):
    session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='class_metrics')
    class_name = models.CharField(max_length=50, verbose_name="클래스명")
    precision = models.FloatField(verbose_name="정밀도")
    recall = models.FloatField(verbose_name="재현율")
    f1_score = models.FloatField(verbose_name="F1 점수")
    instances = models.IntegerField(verbose_name="인스턴스 수")
    
    class Meta:
        unique_together = ['session', 'class_name']
    
    def __str__(self):
        return f"{self.class_name} - {self.session.model_name}"