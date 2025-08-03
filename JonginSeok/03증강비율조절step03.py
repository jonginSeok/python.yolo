import os
import shutil
import random

# 결과물 출력 구조(data_path 하위구조)
# dataset/
# ├── test/ [6%]
# │   ├── bad-broken_large/
# │   │   ├── images/
# │   │   └── labels/
# │   ├── bad-broken_small/
# │   │   ├── images/
# │   │   └── labels/
# │   ├── bad-contamination/
# │   │   ├── images/
# │   │   └── labels/
# │   └── bottle-good/
# │       ├── images/
# │       └── labels/

# 주석
# │   ├── images/
# │   └── labels/

# ├── train/ [78%]
# │   ├── bad-broken_large/
# │   │   ├── images/
# │   │   └── labels/
# │   ├── bad-broken_small/
# │   │   ├── images/
# │   │   └── labels/
# │   ├── bad-contamination/
# │   │   ├── images/
# │   │   └── labels/
# │   └── bottle-good/
# │       ├── images/
# │       └── labels/
# ├── valid/ [16%]
# │   ├── bad-broken_large/
# │   │   ├── images/
# │   │   └── labels/
# │   ├── bad-broken_small/
# │   │   ├── images/
# │   │   └── labels/
# │   ├── bad-contamination/
# │   │   ├── images/
# │   │   └── labels/
# │   └── bottle-good/
# │       ├── images/
# │       └── labels/
# └── data.yaml

# 설정
image_dir = "JonginSeok/dataset/images"
label_dir = "JonginSeok/dataset/labels"
output_base = "JonginSeok/dataset/cnn"

rate_img = [78, 16, 6]  # train, valid, test 비율

# 클래스 매핑
label_map = {
    "bad-broken_large": 0,
    "bad-broken_small": 1,
    "bad-contamination": 2,
    "bottle-good": 3,
}
id_to_class = {v: k for k, v in label_map.items()}
valid_classes = set(label_map.values())

# 출력 폴더 생성
splits = ["train", "valid", "test"]
for split in splits:

    # if split == 'test':
    #     os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
    #     os.makedirs(os.path.join(output_base, split, 'labels'), exist_ok=True)
    # else:
    #     for class_name in label_map.keys():
    #         os.makedirs(os.path.join(output_base, split, class_name, 'images'), exist_ok=True)
    #         os.makedirs(os.path.join(output_base, split, class_name, 'labels'), exist_ok=True)

    for class_name in label_map.keys():
        os.makedirs(os.path.join(output_base, split, class_name, ""), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, class_name, ""), exist_ok=True)

# 이미지 파일 목록 수집
image_files = [
    f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))
]
random.shuffle(image_files)

# 유효한 이미지만 필터링
valid_image_files = []
for img_file in image_files:
    base_name = os.path.splitext(img_file)[0]
    label_path = os.path.join(label_dir, base_name + ".txt")
    if not os.path.exists(label_path):
        print(f"⚠️ 라벨 파일 없음: {base_name}.txt")
        continue

    with open(label_path, "r") as f:
        line = f.readline().strip()
        if not line:
            print(f"⚠️ 라벨 내용 없음: {base_name}.txt")
            continue
        class_id = line.split()[0]
        if not class_id.isdigit() or int(class_id) not in valid_classes:
            print(f"⚠️ 유효하지 않은 클래스 ID: {class_id} in {base_name}.txt")
            continue

    valid_image_files.append(img_file)

# 분할 계산
total = len(valid_image_files)
train_count = int(total * rate_img[0] / 100)
valid_count = int(total * rate_img[1] / 100)
test_count = total - train_count - valid_count

print(
    f"⚠️ total:{total}  train_count:{train_count} valid_count:{valid_count} test_count:{test_count}"
)

split_counts = {"train": train_count, "valid": valid_count, "test": test_count}

# 분할 및 복사
start = 0
for split in splits:
    count = split_counts[split]
    subset = valid_image_files[start : start + count]
    for img_file in subset:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, "r") as f:
            class_id = int(f.readline().strip().split()[0])
            class_name = id_to_class[class_id]

        # 경로 설정
        # if split == 'test':
        #     dst_img = os.path.join(output_base, split, 'images', img_file)
        #     dst_label = os.path.join(output_base, split, 'labels', label_file)
        # else:
        #     dst_img = os.path.join(output_base, split, class_name, 'images', img_file)
        #     dst_label = os.path.join(output_base, split, class_name, 'labels', label_file)

        dst_img = os.path.join(output_base, split, class_name, "", img_file)
        dst_label = os.path.join(output_base, split, class_name, "", label_file)

        # 복사
        shutil.copy2(os.path.join(image_dir, img_file), dst_img)
        shutil.copy2(label_path, dst_label)

    start += count

print("✅ 클래스별 데이터셋 분할 및 정리 완료!")
