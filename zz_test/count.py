from pathlib import Path
import os
from collections import Counter

# 경로 설정
root = Path(__file__).parent.resolve()
label_path = root / 'data/labels'

# 클래스 정의
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 클래스 카운트 초기화
class_count = Counter()

# 라벨 파일들을 하나씩 확인하면서 클래스 개수 세기
for label_file in os.listdir(label_path):
    if label_file.endswith('.txt'):
        with open(label_path / label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                # class_id = int(line.strip().split()[0])  # 첫 번째 값이 클래스 ID라고 가정
                class_id = int(float(line.strip().split()[0]))
                class_name = CLASS_NAMES[class_id]
                class_count[class_name] += 1

# 결과 출력
for class_name in CLASS_NAMES:
    print(f"{class_name}: {class_count[class_name]} images")
