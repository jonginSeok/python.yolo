from ultralytics import YOLO
from pathlib import Path
import cv2

# 모델 경로
root = Path(__file__).parent.resolve()
model_path = root / "runs/bottle/cls2/weights/best.pt"
images_dir = Path(root / "data/images")
crop_base_dir = Path(root / "data/images/crops")

# 모델 로드
model = YOLO(model_path)

# 이미지마다 감지 후 crop
for image_file in images_dir.glob("*.jpg"):
    results = model(image_file)

    for i, r in enumerate(results):
        im = cv2.imread(str(image_file))

        for j, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 클래스 이름 가져오기
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # 클래스별 디렉터리 생성
            class_dir = crop_base_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # crop 이미지 저장
            cropped = im[y1:y2, x1:x2]
            crop_filename = class_dir / f"{image_file.stem}_crop{j}.jpg"
            cv2.imwrite(str(crop_filename), cropped)

print("클래스별 crop 완료!")
