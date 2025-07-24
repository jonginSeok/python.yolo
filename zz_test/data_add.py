from pathlib import Path
import cv2
import albumentations as A
import os

# 클래스 정의
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']
AUGMENT_CLASSES = set(i for i, name in enumerate(CLASS_NAMES) if name != 'bottle-good')

# 증강 파이프라인 정의
transform = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.RandomBrightnessContrast(p=1.0),
    A.HueSaturationValue(p=1.0)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 경로
root = Path(__file__).parent.resolve()
img_path = root / 'data/images'
label_path = root / 'data/labels'

# 증강 처리
for img_file in img_path.glob('*.jpg'):
    label_file = label_path / f"{img_file.stem}.txt"
    if not label_file.exists():
        continue

    # 이미지 로딩
    img = cv2.imread(str(img_file))
    height, width = img.shape[:2]

    # 라벨 로딩
    bboxes = []
    class_labels = []
    with open(label_file, 'r') as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            if int(cls) in AUGMENT_CLASSES:
                bboxes.append([x, y, w, h])
                class_labels.append(int(cls))

    if not bboxes:
        continue  # 증강 대상 없음

    # 증강
    augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
    aug_img = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']

    # 파일 저장 (같은 폴더, _aug 접미어 추가)
    aug_img_file = img_path / f"{img_file.stem}_aug.jpg"
    aug_label_file = label_path / f"{img_file.stem}_aug.txt"
    cv2.imwrite(str(aug_img_file), aug_img)
    with open(aug_label_file, 'w') as f:
        for cls, bbox in zip(aug_labels, aug_bboxes):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")
