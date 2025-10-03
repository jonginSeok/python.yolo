# YOLO Training Dashboard - Django 템플릿

 YOLO/CNN Training 을 위한 Django 웹 애플리케이션입니다.

## 🚀 설치 및 실행

### 1. 가상환경 생성 및 활성화(python 자체 가상머신. anaconda3 활용)
```bash
C:\Users\사용자\Git\python.yolo> C:/ProgramData/anaconda3/Scripts/activate
(base) C:\Users\ngins\Git\python.yolo> conda activate torch_venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows

(torch_venv) C:\Users\ngins\Git\python.yolo>cd bottle
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 데이터베이스 설정
PostgreSQL 데이터베이스가 필요합니다. AWS RDS 또는 로컬 PostgreSQL을 사용하세요.

#### 데이터베이스 마이그레이션
```bash
python manage.py check                    # 변경확인
python manage.py makemigrations           # 마이그레이션 파일 생성 확인
python manage.py migrate                  # 마이그레이션 DB적용
python manage.py showmigrations training  # 마이그레이션 상태 확인

```

#### 데이터베이스 스키마 문제 해결
만약 `column training_trainingsession.total_epochs does not exist` 에러가 발생한다면:
```bash
python manage.py migrate training
```

### 4. 관리자 계정 생성 (선택사항)
```bash
python manage.py createsuperuser
```

### 5. 데모 데이터 로드
```bash
python manage.py load_demo_data
```

### 6. 개발 서버 실행
```bash
python manage.py runserver
```
### 7. 개발 환경
```bash
git url : https://github.com/jonginSeok/python.yolo.git
database : PostgreSQL 17.5 (AWS)
tool : VS Code, SQLGate for PostgreSQL , EditPlus, UltraEdit, OneNote(sticky notes) 등등
site : copilot, lovable.dev, supabase.com 

브라우저에서 `http://127.0.0.1:8000`으로 접속하면 대시보드를 확인할 수 있습니다.
```
## 📊 주요 기능

### 🎯 대시보드 메트릭
- **현재 mAP@0.5**: 모델의 평균 정밀도
- **Training/Validation Loss**: 훈련 및 검증 손실
- **Current Epoch**: 현재 훈련 진행 상황

### 📈 시각화 차트
- **Loss 차트**: 훈련/검증 손실 그래프 (Plotly.js)
- **mAP 차트**: 평균 정밀도 진행 그래프
- **반응형 디자인**: 모든 화면 크기 지원

### 🔧 모델 정보
- 모델명, 버전, 상태
- GPU 정보, 메모리 사용량
- 데이터셋 정보
- 훈련 진행률 바

### 📋 클래스별 성능
- 각 객체 클래스별 정밀도, 재현율, F1-score
- 시각적 진행률 바
- 인스턴스 수 표시

## 🗂️ 프로젝트 구조
```bash
bottle/
├── config/                 # Django 프로젝트 설정
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── training/               # 메인 앱
│   ├── models.py           # 데이터 모델
│   ├── views.py            # 뷰 로직
│   ├── urls.py             # URL 설정
│   ├── admin.py            # 관리자 페이지
│   └── management/
│       └── commands/
│           └── load_demo_data.py

...

├── requirements.txt        # 의존성 목록
└── manage.py               # Django 관리 스크립트
```

## 🎨 디자인 시스템

- **다크 테마**: 전문적인 ML 대시보드 디자인
// - **Tailwind CSS**: 유틸리티 우선 CSS 프레임워크
- **반응형**: 모바일, 태블릿, 데스크톱 지원
- **색상 팔레트**: 보라색 계열 프라이머리 컬러

## 📦 데이터 모델

### TrainingSession
- 훈련 세션 정보 (모델명, 버전, 상태 등)
- GPU, 메모리, 데이터셋 정보
- 진행률 및 시간 추적

### TrainingMetric
- 에포크별 훈련 메트릭
- Loss, mAP, 정밀도, 재현율 데이터

### ClassMetric
- 클래스별 성능 메트릭
- 정밀도, 재현율, F1-score, 인스턴스 수

## 🔌 API 엔드포인트

- `GET /`: 메인 대시보드
- `GET /training/api/<int:session_id>/'`: 훈련 데이터 JSON API

## 🛠️ 커스터마이징

### 새로운 메트릭 추가
1. `models.py`에 필드 추가
2. `views.py`에서 차트 로직 수정
3. `dashboard.html`에서 UI 업데이트

### 실제 YOLO 데이터 연동
```python
# views.py에서 실제 훈련 데이터를 연동
def update_training_metrics(session_id, epoch_data):
    session = TrainingSession.objects.get(id=session_id)
    TrainingMetric.objects.create(
        session=session,
        epoch=epoch_data['epoch'],
        train_loss=epoch_data['train_loss'],
        val_loss=epoch_data['val_loss'],
        map50=epoch_data['map50'],
        map95=epoch_data['map95']
    )
```

## 📱 브라우저 지원

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🔐 보안 설정

프로덕션 환경에서는 다음 설정을 변경하세요:

1. `SECRET_KEY` 변경
2. `DEBUG = False` 설정
3. `ALLOWED_HOSTS` 설정
4. HTTPS 적용
5. 데이터베이스를 PostgreSQL로 변경

## 🤝 기여하기

1. Fork 프로젝트
2. Feature 브랜치 생성
3. 변경사항 커밋
4. 브랜치에 Push
5. Pull Request 생성

--- 
🎉
