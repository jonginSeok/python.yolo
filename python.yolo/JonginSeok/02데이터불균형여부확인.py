"""
train 데이터 분석 결과:
- 총 209개의 라벨이 수집되었습니다.

🔢 클래스별 샘플 수:
- bad-broken_large: 0개
- bad-broken_small: 0개
- bad-contamination: 0개
- bottle-good: 209개

valid 데이터 분석 결과:
- 총 63개의 라벨이 수집되었습니다.

🔢 클래스별 샘플 수:
- bad-broken_large: 20개
- bad-broken_small: 22개
- bad-contamination: 21개
- bottle-good: 0개
"""

import os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# 클래스 이름과 인덱스 매핑
label_map = {
    "bad-broken_large": 0,
    "bad-broken_small": 1,
    "bad-contamination": 2,
    "bottle-good": 3,
}

# 인덱스 → 클래스 이름 역매핑
index_to_name = {v: k for k, v in label_map.items()}


def collect_label_indices(label_dir):
    class_indices = []
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_indices.append(int(parts[0]))  # index 0: class label
    return class_indices


def analyze_class_distribution(base_path):
    train_labels = os.path.join(base_path, "train", "labels")
    valid_labels = os.path.join(base_path, "valid", "labels")

    train_classes = collect_label_indices(train_labels)
    valid_classes = collect_label_indices(valid_labels)

    total_classes = train_classes + valid_classes
    class_counts = Counter(total_classes)

    # 클래스 이름으로 변환
    named_counts = {
        index_to_name[idx]: count
        for idx, count in class_counts.items()
        if idx in index_to_name
    }
    return named_counts


def visualize_distribution(named_counts):
    classes = list(named_counts.keys())
    counts = list(named_counts.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=classes, y=counts, palette="mako", hue=classes, legend=False)
    plt.title("Class Distribution by Name")
    plt.xlabel("Class Name")
    plt.ylabel("Number of Instances")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def check_imbalance(named_counts, threshold=0.1):
    total = sum(named_counts.values())
    proportions = {cls: count / total for cls, count in named_counts.items()}
    max_p = max(proportions.values())
    min_p = min(proportions.values())

    print("\n📊 Class Proportions:")
    for cls, p in proportions.items():
        print(f"{cls}: {p:.2%}")

    if max_p - min_p > threshold:
        print("\n⚠️ 데이터 불균형이 존재합니다.")
    else:
        print("\n✅ 데이터는 비교적 균형 잡혀 있습니다.")


# 실행
base_path = "dataset/origin"
named_counts = analyze_class_distribution(base_path)
# visualize_distribution(named_counts)
check_imbalance(named_counts)
