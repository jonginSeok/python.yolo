from ultralytics import YOLO
from pathlib import Path
import numpy as np

root = Path(__file__).parent.resolve()

# 모델 경로 및 이미지 폴더 경로
model_path = root / 'runs/bottle/cls2/weights/best.pt'
images_folder = root / 'data/test/images'

# 모델 로드
model = YOLO(model_path)

# 평균 신뢰도 계산을 위한 변수
confidences = []
image_files = list(images_folder.glob("*"))

for image_path in image_files:
    results = model(image_path)

    # 바운딩 박스 정보
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            conf = box.conf[0].item()
            confidences.append(conf)

# 평균 계산
if confidences:
    avg_conf = np.mean(confidences)
    print(f"전체 이미지에 대한 평균 Confidence: {avg_conf:.2f}")
else:
    print("어떤 이미지에서도 바운딩 박스가 검출되지 않았습니다.")