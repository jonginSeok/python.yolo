import torch

print('True여야 GPU 사용 가능 :', torch.cuda.is_available())  # True여야 GPU 사용 가능
print('사용 가능한 GPU 수:', torch.cuda.device_count())  # 사용 가능한 GPU 수

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" \n device ::: {device} \n", )

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if __name__ == '__main__':
    from ultralytics import YOLO

    # 기존 모델 불러오기 (COCO 학습됨)
    model = YOLO('yolo11n.pt')        # 각자의 경로 .to('cuda') 

    # model.train(
    #     data='JonginSeok/dataset/data.yaml',    # 데이터셋 구성 파일
    #     epochs=100,                             # 학습 에폭 수
    #     imgsz=640,                              # 이미지 크기
    #     batch=16,                               # 배치 사이즈
    #     project = 'JonginSeok/dataset/result',  # 결과물 생성 경로
    #     name='yolo11n_bottle_4class',           # 결과물 폴더이름 
    #     lr0=0.01,                               # 초기 학습률
    #     cache=True,                             # 캐싱 활성화
    #     # cls_weights=[1.0, 1.0, 1.0, 2.0],     # 클래스별 가중치 (bottle-good 강조)
    #     patience=20,                            # early stopping
    #     cos_lr=True                             # cosine learning rate scheduler
    # )
    
    # 2025.07.24 add
    model.train(
        data='JonginSeok/dataset/data.yaml',
        epochs=100, # 10->50->100
        imgsz=640,
        batch=16, #16->16->8  # 메모리 문제로 배치 사이즈 줄임
        project = 'JonginSeok/dataset/result',
        name='yolo11n_bottle_4class',
        pretrained=True,
        # patience=10, # 정확도(es_metric)가 10번을 넘기면 그만
        # es_metric='metrics/mAP50-95(B)'   # mAP50' # old version
        # verbose=True,  # 학습 과정 출력
    )
