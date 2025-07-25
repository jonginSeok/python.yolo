import os
import shutil
from pathlib import Path
import random

# 원본 데이터 경로
root = Path(__file__).parent.resolve()
source_dir = Path(root / "data/images/crops")
classes = os.listdir(source_dir)

# Split 비율
split_ratios = [0.8, 0.15, 0.05]  # Train, Val, Test

# 분할 대상 디렉토리
target_root = Path(root / "data/images/split_crops")
subsets = ["train", "val", "test"]

for cls in classes:
    cls_path = source_dir / cls
    images = list(cls_path.glob("*"))  # 이미지 리스트 가져오기
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for subset, img_list in splits.items():
        target_dir = target_root / subset / cls
        target_dir.mkdir(parents=True, exist_ok=True)
        for img_path in img_list:
            shutil.copy(img_path, target_dir / img_path.name)