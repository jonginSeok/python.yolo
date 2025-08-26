import os
import shutil
import glob
# import cv2  # pip install opencv-python
from PIL import Image #, ImageOps

"""
🔢 클래스별 샘플 수:

valid   - bad-broken_large : 20 개
        - bad-broken_small : 22 개
        - bad-contamination: 21 개
train   - bottle-good      : 209 개

- bad-broken_large : 180 개
- bad-broken_small : 198 개
- bad-contamination: 189 개
- bottle-good      : 209 개

"""


rotation_angle = 40

rotation = 0  # 회전 비율 rotation_angle
# valid 파일 처리
for i in range(0, int(360 // rotation_angle)):
    ratio = f"rot{rotation}"  # 파일명에 추가할 회전 비율
    # print(f"🗂️  회전 비율: {rotation}도, 파일명에 회전비율: _{ratio}")

    # 🗂️ 설정: 원본 이미지 폴더 및 저장 위치
    input_image_folder = "dataset/origin/valid/images/"  # 원본 이미지 폴더 경로
    output_image_folder = "JonginSeok/dataset/images/"  # 저장할 폴더 경로

    input_labels_folder = "dataset/origin/valid/labels/"
    output_labels_folder = "JonginSeok/dataset/labels/"

    # 폴더가 없으면 생성
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    # 이미지 확장자 필터링
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_image_folder, ext)))

    # print(f"📝 총 {len(image_files)}개의 이미지 파일을 찾았습니다.")

    def rotate_images_fill_white(image_files, rotation_angle, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGBA")
                file_name = os.path.basename(img_path)

                # 확장자 분리
                basename, extension = os.path.splitext(file_name)

                # 이미지 중심 기준으로 회전 (투명 배경 포함)
                rotated = img.rotate(
                    rotation_angle, resample=Image.BICUBIC, expand=True
                )

                # 새로운 흰색 배경 이미지 생성
                white_bg = Image.new("RGBA", rotated.size, (255, 255, 255, 255))
                merged = Image.alpha_composite(white_bg, rotated)

                # 원본 크기로 잘라서 비율 유지
                center_x, center_y = merged.size[0] // 2, merged.size[1] // 2
                original_size = img.size
                left = center_x - original_size[0] // 2
                top = center_y - original_size[1] // 2
                right = left + original_size[0]
                bottom = top + original_size[1]
                cropped = merged.crop((left, top, right, bottom)).convert(
                    "RGB"
                )  # 최종 RGB 변환

                if rotation == 0:
                    output_path = os.path.join(output_dir, f"{basename}.jpg")
                else:
                    output_path = os.path.join(output_dir, f"{basename}_{ratio}.jpg")

                cropped.save(output_path)
                # print(f"✅ Saved: {output_path}")
            except Exception as e:
                print(f"⚠️ Error with {img_path}: {e}")

    rotate_images_fill_white(image_files, rotation, output_image_folder)

    # 라벨 파일도 같은 방식으로 가져오기
    label_extensions = ["*.txt"]
    label_files = []
    for ext in label_extensions:
        label_files.extend(glob.glob(os.path.join(input_labels_folder, ext)))

    for label_file in label_files:

        try:
            with open(label_file, "r") as f:
                lines = f.readlines()

            # 라벨 파일에서 클래스와 좌표 추출
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # 잘못된 형식의 라벨은 무시
                class_id = parts[0]
                x_center, y_center, width, height = map(float, parts[1:5])
                new_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

            # 새 라벨 파일 저장
            label_name = os.path.basename(label_file)

            # 확장자 분리
            basename, extension = os.path.splitext(label_name)

            if rotation == 0:
                new_label_file = os.path.join(output_labels_folder, f"{basename}.txt")
            else:
                new_label_file = os.path.join(
                    output_labels_folder, f"{basename}_{ratio}.txt"
                )

            with open(new_label_file, "w") as f:
                f.writelines(new_lines)

            # print(f"✅ Saved: {new_label_file}")
        except Exception as e:
            print(f"⚠️ Error with {new_label_file}: {e}")

    rotation += rotation_angle

# train 파일 처리
# 경로 설정
input_image_folder = "dataset/origin/train/images/"
output_image_folder = "JonginSeok/dataset/images/"

input_labels_folder = "dataset/origin/train/labels/"
output_labels_folder = "JonginSeok/dataset/labels/"

# 출력 폴더 생성
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)


# 이미지 파일 복사
for file_name in os.listdir(input_image_folder):
    src = os.path.join(input_image_folder, file_name)
    dst = os.path.join(output_image_folder, file_name)
    if os.path.isfile(src):
        shutil.copy2(src, dst)

# 라벨 파일 복사
for file_name in os.listdir(input_labels_folder):
    src = os.path.join(input_labels_folder, file_name)
    dst = os.path.join(output_labels_folder, file_name)
    if os.path.isfile(src):
        shutil.copy2(src, dst)

# print(f"📝 총 {count}개의 이미지 파일을 찾았습니다.")
print("✅ 이미지와 라벨 파일 복사가 완료되었습니다.")
