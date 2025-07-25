# import os
# print("현재 작업 디렉토리:", os.getcwd())


# 이미지 로드
from pathlib import Path
import numpy as np
image_path = Path(__file__).parent.parent / 'data/origin_img/HR_img1_origin.jpg'
# print("경로 확인:", image_path)
# print("존재 여부:", image_path.exists())  # True여야 정상
img_array = np.fromfile(image_path, np.uint8)


import cv2
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
# img = cv2.imread(str(image_path))
print("이미지 로딩 성공 여부:", img is not None)


import easyocr

# ✅ OCR 모델 초기화 (이 부분이 반드시 먼저 있어야 함)
reader = easyocr.Reader(['ko', 'en'])

# 텍스트 인식
results = reader.readtext(img)
for bbox, text, confidence in results:
    print(f"인식된 텍스트: {text} (신뢰도: {confidence:.2f})")
