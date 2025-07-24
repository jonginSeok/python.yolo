from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

root = Path(__file__).parent.resolve()

# 1. 경로 설정
model_path = root / 'runs/bottle/cls/weights/best.pt'
image_path = root / 'data/test/images/012_png.rf.37d2cda9e7abf369cb71e83ea0974e0f.jpg'

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