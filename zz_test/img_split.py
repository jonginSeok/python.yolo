import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# 현재 경로
root = Path(__file__).parent.resolve()
img_path = root / 'data/images'
label_path = root / 'data/labels'

# 클래스 정의
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 나눌 비율
split_ratio = {'train': 8.0, 'val': 1.5, 'test': 0.5}
total_ratio = sum(split_ratio.values())

# 목적지 폴더 생성
for split in split_ratio.keys():
    for folder in ['images', 'labels']:
        dest_dir = root / f'{split}/{folder}'
        dest_dir.mkdir(parents=True, exist_ok=True)

# 라벨별 파일 목록 수집
class_to_files = defaultdict(list)

for label_file in label_path.glob('*.txt'):
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # class_id = int(line.split()[0])
            class_id = int(float(line.split()[0]))
            class_name = CLASS_NAMES[class_id]
            if class_name in CLASS_NAMES:
                class_to_files[class_name].append(label_file.name)
                break  # 한 이미지에 여러 클래스가 있어도 첫 번째 기준으로 분류

# 각 클래스에서 파일 분할
for class_name, files in class_to_files.items():
    random.shuffle(files)
    num_files = len(files)
    train_end = int(num_files * split_ratio['train'] / total_ratio)
    val_end = train_end + int(num_files * split_ratio['val'] / total_ratio)

    splits = {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

    for split, split_files in splits.items():
        for file in split_files:
            img_file = img_path / file.replace('.txt', '.jpg')  # 이미지 확장자에 맞게 변경
            lbl_file = label_path / file

            # 복사
            if img_file.exists():
                shutil.copy(img_file, root / f'{split}/images' / img_file.name)
            shutil.copy(lbl_file, root / f'{split}/labels' / lbl_file.name)
