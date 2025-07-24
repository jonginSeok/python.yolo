from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

root = Path(__file__).parent.resolve()

# 1. 경로 설정
model_path = root / 'runs/bottle/cls2/weights/best.pt'
image_path = root / 'data/test/images/010_png.rf.7d2c4b35c86ae8f2ac61eb7d7f20423b_v_aug.jpg'

# 2. 모델 로드
model = YOLO(model_path)

# 3. 이미지 추론
results = model(image_path)

# 4. 바운딩 박스가 그려진 결과 이미지 얻기
# 결과는 list 형태이므로 첫 번째 결과만 사용
res_plotted = results[0].plot()  # numpy.ndarray (BGR)

# 5. 시각화 (OpenCV는 BGR, matplotlib은 RGB)
res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(res_rgb)
plt.axis('off')
plt.title('Detection Result')
plt.show()

# 6. 예측 결과 정보 출력
boxes = results[0].boxes  # 바운딩 박스 정보

if boxes is not None:
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())      # 클래스 ID
        conf = box.conf[0].item()            # Confidence score
        xyxy = box.xyxy[0].tolist()          # Bounding box 좌표

        class_name = model.names[cls_id]     # 클래스 이름
        print(f"[{i}] Class: {class_name}, Confidence: {conf:.2f}, BBox: {xyxy}")
else:
    print("No detections found.")
