# 
'''
🔢 클래스별 샘플 수:
- bad-broken_large: 180개 + 63개 = 243개
- bad-broken_small: 198개 + 63개 = 356개
- bad-contamination: 189개 + 63개 = 252개
- bottle-good: 209개
'''

import cv2 # pip install opencv-python
import os
import glob
from PIL import Image, ImageOps

# train_valid = 'train'
train_valid = 'valid'
rottn_angle = 0  # 회전 비율 (90도 회전)

for i in range(1, 8):
    rottn_angle += 36  # 36도씩 증가
    ratio = f'ratio{rottn_angle}'  # 파일명에 추가할 회전 비율
    print(f"회전 비율: {rottn_angle}도, 파일명에 추가: {ratio}")

    # 🗂️ 설정: 원본 이미지 폴더 및 저장 위치
    input_image_folder = '/Users/ngins/Git/python.yolo/dataset/ngins751206/'+train_valid+'/images/'        # 원본 이미지 폴더 경로
    output_image_folder = '/Users/ngins/Git/python.yolo/dataset/ngins751206/'+train_valid+'/images/' +ratio+'/' #'your_output_folder_path'  # 저장할 폴더 경로

    input_labels_folder = '/Users/ngins/Git/python.yolo/dataset/ngins751206/'+train_valid+'/labels/'
    output_labels_folder = '/Users/ngins/Git/python.yolo/dataset/ngins751206/'+train_valid+'/labels/'  +ratio+'/'

    # 폴더가 없으면 생성
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    # 이미지 확장자 필터링
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_image_folder, ext)))

    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")

    # # 🌀 회전 함수 정의
    # def rotate_image(img, angle):
    #     h, w = img.shape[:2]
    #     center = (w // 2, h // 2)
    #     M = cv2.getRotationMatrix2D(center, angle, 1.0)

    #     # 여유 공간을 주기 위해 회전 후 전체 크기 고려
    #     cos = abs(M[0, 0])
    #     sin = abs(M[0, 1])
    #     new_w = int((h * sin) + (w * cos))
    #     new_h = int((h * cos) + (w * sin))

    #     M[0, 2] += (new_w / 2) - center[0]
    #     M[1, 2] += (new_h / 2) - center[1]

    #     return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    def rotate_images_fill_white(image_files, rotation_angle, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGBA")
                file_name = os.path.basename(img_path)

                # 이미지 중심 기준으로 회전 (투명 배경 포함)
                rotated = img.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)

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
                cropped = merged.crop((left, top, right, bottom)).convert("RGB")  # 최종 RGB 변환

                output_path = os.path.join(output_dir, f"{file_name}_{ratio}.jpg" )
                cropped.save(output_path)
                print(f"✅ Saved: {output_path}")
            except Exception as e:
                print(f"⚠️ Error with {img_path}: {e}")

    rotate_images_fill_white(image_files, rottn_angle, output_image_folder)

    # 라벨 파일도 같은 방식으로 가져오기
    label_extensions = ['*.txt']
    label_files = []
    for ext in label_extensions:
        label_files.extend(glob.glob(os.path.join(input_labels_folder, ext)))


    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # 라벨 파일에서 클래스와 좌표 추출
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # 잘못된 형식의 라벨은 무시
            class_id = parts[0]
            x_center, y_center, width, height = map(float, parts[1:5])
            new_lines.append(f"{class_id} {x_center} {y_center} {width} {height}/n")

        # 새 라벨 파일 저장
        label_name = os.path.basename(label_file)
    #    new_label_file = os.path.join(output_folder, label_name)
        new_label_file = os.path.join(output_labels_folder, f"{label_name}_{ratio}.txt")
        
        with open(new_label_file, 'w') as f:
            f.writelines(new_lines)

    # # 🔁 이미지 하나씩 처리 및 저장
    # for file in image_files:
    #     img = cv2.imread(file)
    #     if img is None:
    #         print(f"이미지 로딩 실패: {file}")
    #         continue

    #     # rotated = rotate_image(img, angleX)  # angleX도 회전


    #     # 이미지 파일 경로 및 출력 경로 설정
    #     image_path = "input.jpg"  # 예시: 현재 경로의 input.jpg 파일을 사용
    #     output_path = "rotated_image.jpg" # 예시: 회전된 이미지를 rotated_image.jpg로 저장

    #     # 회전 각도 설정
    #     rotation_angle = 45  # 예시: 45도 회전

    #     # 이미지 회전 및 저장
    #     rotate_image(input_folder, rotation_angle, output_folder)

    #     # rotated = rotate_image(img, angleX)  # angleX도 회전

    #     # 🔤 저장 파일명 생성
    #     base = os.path.basename(file)
    #     name, ext = os.path.splitext(base)
    #     print(f"처리 중: {file} -> {os.path} {name}_{ratio}{ext}")
    #     output_file = os.path.join(output_folder, f"{name}_{ratio}{ext}")
    #     cv2.imwrite(output_file, rotated)
    #     print(f"저장 완료: {output_file}")
    
    

print("모든 이미지가 처리되었습니다.")
    