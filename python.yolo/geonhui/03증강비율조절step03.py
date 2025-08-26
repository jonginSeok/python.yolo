import os
import shutil
import random

# --- 경로 설정 (스크립트 위치 기준으로 동적 설정) ---
# 현재 스크립트 파일의 절대 경로를 가져옴
script_path = os.path.abspath(__file__)
# 스크립트가 있는 디렉토리 경로를 가져옴 (예: .../geonhui)
script_dir = os.path.dirname(script_path)

# 스크립트 디렉토리를 기준으로 데이터 경로를 설정합니다.
# 가정: 'dataset' 폴더는 스크립트와 같은 폴더 안에 있습니다.
base_data_dir = os.path.join(script_dir, "dataset")

image_dir = os.path.join(base_data_dir, "images")
label_dir = os.path.join(base_data_dir, "labels")
output_base = os.path.join(base_data_dir, "cnn")

print(f"✅ 기준 데이터 경로: {base_data_dir}")
print(f"📂 분할 대상 이미지 경로: {image_dir}")
print(f"📂 분할 대상 라벨 경로: {label_dir}")
print(f"🚀 결과물 저장 경로: {output_base}")
# ----------------------------------------------------

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
    for class_name in label_map.keys():
        # 'images'와 'labels' 하위 폴더를 명시적으로 생성
        os.makedirs(os.path.join(output_base, split, class_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, class_name, 'labels'), exist_ok=True)

# 이미지 파일 목록 수집
try:
    image_files = [
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
except FileNotFoundError:
    print(f"❌ 경로를 찾을 수 없습니다: '{image_dir}'")
    print("스크립트와 같은 위치에 'dataset/images' 폴더가 있는지 확인해주세요.")
    exit() # 경로가 없으면 스크립트 중단

random.shuffle(image_files)

# 유효한 이미지만 필터링
print("\n데이터 유효성 검사를 시작합니다...")
valid_image_files = []
for img_file in image_files:
    base_name = os.path.splitext(img_file)[0]
    label_path = os.path.join(label_dir, base_name + ".txt")
    if not os.path.exists(label_path):
        # print(f"⚠️ 라벨 파일 없음: {base_name}.txt")
        continue

    with open(label_path, "r") as f:
        line = f.readline().strip()
        if not line:
            # print(f"⚠️ 라벨 내용 없음: {base_name}.txt")
            continue
        class_id_str = line.split()[0]
        if not class_id_str.isdigit() or int(class_id_str) not in valid_classes:
            # print(f"⚠️ 유효하지 않은 클래스 ID: {class_id_str} in {base_name}.txt")
            continue

    valid_image_files.append(img_file)

# 분할 계산
total = len(valid_image_files)
if total == 0:
    print("❌ 처리할 유효한 이미지가 없습니다. 라벨 파일을 확인해주세요.")
    exit()

train_count = int(total * rate_img[0] / 100)
valid_count = int(total * rate_img[1] / 100)
test_count = total - train_count - valid_count

print(
    f"\n✅ 데이터 분할 -> 총 {total}개 (Train: {train_count}개, Valid: {valid_count}개, Test: {test_count}개)"
)

split_counts = {"train": train_count, "valid": valid_count, "test": test_count}

# 분할 및 복사
print("파일 복사를 시작합니다...")
start = 0
for split in splits:
    count = split_counts[split]
    subset = valid_image_files[start : start + count]
    for img_file in subset:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        
        src_img_path = os.path.join(image_dir, img_file)
        src_label_path = os.path.join(label_dir, label_file)

        with open(src_label_path, "r") as f:
            class_id = int(f.readline().strip().split()[0])
            class_name = id_to_class[class_id]

        # 목적지 경로 설정
        dst_img_path = os.path.join(output_base, split, class_name, 'images', img_file)
        dst_label_path = os.path.join(output_base, split, class_name, 'labels', label_file)

        # 복사
        shutil.copy2(src_img_path, dst_img_path)
        shutil.copy2(src_label_path, dst_label_path)

    start += count

print("\n🎉 모든 작업이 완료되었습니다! 클래스별 데이터셋 분할 및 정리를 성공했습니다.")