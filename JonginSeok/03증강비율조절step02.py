import os
import shutil
import random

# 결과물 출력 구조(data_path 하위구조)
# dataset/
# ├── test/ [6%]
# │   ├── images/
# │   └── labels/
# ├── train/ [78%]
# │   ├── images/
# │   └── labels/
# ├── valid/ [16%]
# │   ├── images/
# │   └── labels/
# └── data.yaml

# 경로 설정
image_dir = "JonginSeok/dataset/images"
label_dir = "JonginSeok/dataset/labels"
data_path = "JonginSeok/dataset"
rate_img = [78, 16, 6]

# 라벨 맵 정의
label_map = {
    "bad-broken_large": 0,
    "bad-broken_small": 1,
    "bad-contamination": 2,
    "bottle-good": 3,
}

# 분할 폴더 생성
splits = ["train", "valid", "test"]
for split in splits:
    os.makedirs(os.path.join(data_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(data_path, split, "labels"), exist_ok=True)

# 이미지-라벨 매칭 리스트 생성
paired_files = []
for img_file in os.listdir(image_dir):
    name, ext = os.path.splitext(img_file)
    label_file = name + ".txt"
    label_path = os.path.join(label_dir, label_file)
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            line = f.readline().strip()
            if line:
                label_idx = int(line.split()[0])
                if label_idx in label_map.values():
                    paired_files.append((img_file, label_file))

# 셔플 후 분할
random.shuffle(paired_files)
total = len(paired_files)
train_end = int(total * rate_img[0] / 100)
valid_end = train_end + int(total * rate_img[1] / 100)
test_end = total - train_end - int(total * rate_img[1] / 100)

print(f"⚠️  total:{total}  train:{train_end} valid:{int(total * rate_img[1] / 100)} test:{test_end}")

splits_data = {
    "train": paired_files[:train_end],
    "valid": paired_files[train_end:valid_end],
    "test": paired_files[valid_end:],
}

# 파일 복사
for split, files in splits_data.items():
    for img_file, label_file in files:
        shutil.copy(
            os.path.join(image_dir, img_file),
            os.path.join(data_path, split, "images", img_file),
        )
        shutil.copy(
            os.path.join(label_dir, label_file),
            os.path.join(data_path, split, "labels", label_file),
        )

print("✅ 이미지 및 라벨 파일 분할 완료!")
