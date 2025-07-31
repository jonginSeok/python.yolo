import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    from ultralytics import YOLO

    # 기존 모델 불러오기 (COCO 학습됨)
    model = YOLO('yolo11n.pt') # 각자의 경로
    
    # 2025.07.24 add
    model.train(
        data='JonginSeok/ngins7512/dataset/data.yaml',
        epochs=100, # 10->50->100
        imgsz=640,
        batch=8, #16->16->8  # 메모리 문제로 배치 사이즈 줄임
        project = 'JonginSeok/ngins7512/dataset',
        name='yolo11n_bottle_4class',
        pretrained=True,
        # patience=10, # 정확도(es_metric)가 10번을 넘기면 그만
        # es_metric='metrics/mAP50-95(B)'   # mAP50' # old version
        # verbose=True,  # 학습 과정 출력
    )
