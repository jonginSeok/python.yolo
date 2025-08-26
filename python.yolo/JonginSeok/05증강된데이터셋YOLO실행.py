import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('True여야 GPU 사용 가능 :', torch.cuda.is_available())  # True여야 GPU 사용 가능
# print(f'사용 가능한 GPU({device}) 수:', torch.cuda.device_count())  # 사용 가능한 GPU 수

if __name__ == "__main__":
    from ultralytics import YOLO

    # 기존 모델 불러오기 (COCO 학습됨)
    model = YOLO("yolo11n.pt")  # 각자의 경로 .to('cuda')

    # 2025.07.24 add
    model.train(
        data="JonginSeok/dataset/data.yaml",
        epochs=100,  # 10->50->100
        imgsz=640,  # GPU 메모리에 따라 640, 512, 416 등으로 조절 가능해요.
        batch=16,  # 16->16->8  # 메모리 문제로 배치 사이즈 줄임
        project="JonginSeok/dataset/result",
        name="yolo11n_bottle_4class",
        verbose=True,  # 학습 과정 출력
        # close_mosaic=10,
        # pretrained=True,
        # patience=10, # 정확도(es_metric)가 10번을 넘기면 그만
        # es_metric='metrics/mAP50-95(B)'   # mAP50' # old version
        # hyp='hyp.yaml'
    )
