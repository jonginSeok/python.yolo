from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.db.models import Q, Avg, Max, Min
from django.utils import timezone
from datetime import timedelta
import json
import plotly.graph_objs as go
import plotly.utils

from .models import TrainingSession, TrainingMetric, ClassMetric
from .forms import TrainingSessionForm



# 학습 세션 목록(다건 조회)
def training_list(request):
    return render(request, 'training/training_list.html')

# 학습 세션 입력
def training_form(request):
    return render(request, 'training/training_form.html')





# training/views.py
def training_create(request):
    if request.method == 'POST':
        form = TrainingSessionForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('trainings:list')
    else:
        form = TrainingSessionForm()
    return render(request, 'training/training_form.html', {'form': form})

def training_update(request, pk):
    session = get_object_or_404(TrainingSession, pk=pk)
    if request.method == 'POST':
        form = TrainingSessionForm(request.POST, request.FILES, instance=session)
        if form.is_valid():
            form.save()
            return redirect('trainings:detail', pk=pk)
    else:
        form = TrainingSessionForm(instance=session)
    return render(request, 'training/training_form.html', {'form': form})





# 학습 세션 출력(단건 조회)
def training_output(request):
    """메인 대시보드 뷰"""
    # 최신 훈련 세션 가져오기
    try:
        latest_session = TrainingSession.objects.latest('created_at')
        latest_metrics = latest_session.metrics.last()
        class_metrics = latest_session.class_metrics.all()
        
        # 차트 데이터 생성
        loss_chart = create_loss_chart(latest_session)
        map_chart = create_map_chart(latest_session)
        
        # 성능 개선 계산 (이전 10개 에포크와 비교)
        metrics_count = latest_session.metrics.count()
        if metrics_count > 10:
            recent_avg = latest_session.metrics.order_by('-epoch')[:5].aggregate(Avg('map50'))['map50__avg']
            old_avg = latest_session.metrics.order_by('-epoch')[5:10].aggregate(Avg('map50'))['map50__avg']
            map_change = ((recent_avg - old_avg) / old_avg * 100) if old_avg else 0
        else:
            map_change = 0
            
        # 손실 변화 계산
        if metrics_count > 5:
            recent_loss = latest_session.metrics.order_by('-epoch')[:3].aggregate(Avg('train_loss'))['train_loss__avg']
            old_loss = latest_session.metrics.order_by('-epoch')[3:6].aggregate(Avg('train_loss'))['train_loss__avg']
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
        'session': latest_session,
        'latest_metrics': latest_metrics,
        'class_metrics': class_metrics,
        'loss_chart': loss_chart,
        'map_chart': map_chart,
        'map_change': round(map_change, 1),
        'loss_change': round(loss_change, 1),
    }
    
    return render(request, 'training/training_output.html', context)

def training_data_api(request, session_id):
    """훈련 데이터 API"""
    session = get_object_or_404(TrainingSession, id=session_id)
    metrics = session.metrics.all()
    
    data = {
        'session': {
            'id': session.id,
            'model_name': session.model_name,
            'version': session.version,
            'status': session.status,
            'progress': session.progress_percentage,
            'training_time': session.training_duration,
        },
        'metrics': [
            {
                'epoch': metric.epoch,
                'train_loss': metric.train_loss,
                'val_loss': metric.val_loss,
                'map50': metric.map50,
                'map95': metric.map95,
            }
            for metric in metrics
        ],
        'class_metrics': [
            {
                'class_name': cm.class_name,
                'precision': cm.precision,
                'recall': cm.recall,
                'f1_score': cm.f1_score,
                'instances': cm.instances,
            }
            for cm in session.class_metrics.all()
        ]
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
        mode='lines',
        name='Training Loss',
        line=dict(color='#8b5cf6', width=2)
    )
    
    trace2 = go.Scatter(
        x=epochs,
        y=val_losses,
        mode='lines',
        name='Validation Loss',
        line=dict(color='#06b6d4', width=2)
    )
    
    layout = go.Layout(
        title='Training & Validation Loss',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
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
        mode='lines',
        name='mAP@0.5',
        line=dict(color='#f59e0b', width=2)
    )
    
    trace2 = go.Scatter(
        x=epochs,
        y=map95_values,
        mode='lines',
        name='mAP@0.5:0.95',
        line=dict(color='#ef4444', width=2)
    )
    
    layout = go.Layout(
        title='Mean Average Precision (mAP)',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='mAP'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
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
    
    trace1 = go.Scatter(x=epochs, y=train_losses, mode='lines', name='Training Loss', line=dict(color='#8b5cf6', width=2))
    trace2 = go.Scatter(x=epochs, y=val_losses, mode='lines', name='Validation Loss', line=dict(color='#06b6d4', width=2))
    
    layout = go.Layout(
        title='Training & Validation Loss',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='Loss'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_demo_map_chart():
    """데모 mAP 차트"""
    epochs = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    map50_values = [0.42, 0.56, 0.68, 0.74, 0.78, 0.81, 0.83, 0.85, 0.86, 0.87, 0.88]
    map95_values = [0.24, 0.32, 0.41, 0.47, 0.52, 0.55, 0.57, 0.59, 0.61, 0.62, 0.63]
    
    trace1 = go.Scatter(x=epochs, y=map50_values, mode='lines', name='mAP@0.5', line=dict(color='#f59e0b', width=2))
    trace2 = go.Scatter(x=epochs, y=map95_values, mode='lines', name='mAP@0.5:0.95', line=dict(color='#ef4444', width=2))
    
    layout = go.Layout(
        title='Mean Average Precision (mAP)',
        xaxis=dict(title='Epoch'),
        yaxis=dict(title='mAP'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)