#!/usr/bin/env python3
"""
geonhui 데이터셋 실전 증강 도구
소수 클래스(bad-broken_large, bad-broken_small, bad-contamination) 집중 증강
"""

import os
import shutil
import glob
import cv2
import numpy as np
from PIL import Image
import math
import random
from collections import Counter

# =============================================================================
# 설정 섹션 - geonhui 폴더 기준
# =============================================================================

# 기존 데이터셋 폴더 (geonhui)
SOURCE_DATASET_PATH = "geonhui"

# 클래스별 증강 배수 설정 (불균형 해결)
CLASS_AUGMENTATION_MULTIPLIERS = {
    0: 5,    # bad-broken_large: 20개 → 100개 (5배)
    1: 5,    # bad-broken_small: 22개 → 110개 (5배)  
    2: 5,    # bad-contamination: 21개 → 105개 (5배)
    3: 1     # bottle-good: 209개 → 그대로 (증강 안함)
}

# 클래스 이름 매핑
CLASS_NAMES = {
    0: "bad-broken_large",
    1: "bad-broken_small", 
    2: "bad-contamination",
    3: "bottle-good"
}

# 증강 기법별 설정
AUGMENTATION_SETTINGS = {
    "rotation_angles": [15, 30, 45, -15, -30],  # 회전 각도
    "brightness_factors": [0.7, 0.8, 1.2, 1.3],  # 밝기 조절
    "noise_levels": [10, 20, 30],  # 노이즈 강도
    "blur_kernels": [3, 5],  # 블러 커널 크기
}

def rotate_bounding_box(x, y, w, h, angle_degrees, img_width, img_height):
    """바운딩박스를 회전시키는 정확한 계산"""
    
    # 각도를 라디안으로 변환
    angle_rad = math.radians(angle_degrees)
    
    # 이미지 중심점
    cx, cy = 0.5, 0.5
    
    # 바운딩박스 중심점을 이미지 중심 기준 좌표로 변환
    x_rel = x - cx
    y_rel = y - cy
    
    # 회전 변환
    x_new_rel = x_rel * math.cos(angle_rad) - y_rel * math.sin(angle_rad)
    y_new_rel = x_rel * math.sin(angle_rad) + y_rel * math.cos(angle_rad)
    
    # 다시 절대 좌표로 변환
    x_new = x_new_rel + cx
    y_new = y_new_rel + cy
    
    # 바운딩박스가 이미지 범위를 벗어나지 않도록 클리핑
    x_new = max(0, min(1, x_new))
    y_new = max(0, min(1, y_new))
    
    return x_new, y_new, w, h

def apply_brightness_augmentation(image, factor):
    """밝기 조절"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_noise_augmentation(image, noise_level):
    """노이즈 추가"""
    noise = np.random.randint(0, noise_level, image.shape, dtype=np.uint8)
    return cv2.add(image, noise)

def apply_blur_augmentation(image, kernel_size):
    """블러 효과"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_rotation_augmentation(image, angle):
    """회전 변환"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # 회전 행렬 생성
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 회전된 이미지의 새로운 크기 계산
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # 이미지 중심 이동
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]
    
    # 회전 적용
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return rotated

def analyze_class_distribution():
    """현재 클래스 분포 분석"""
    
    print("🔍 현재 클래스 분포 분석...")
    
    class_counts = {cls: 0 for cls in CLASS_NAMES.keys()}
    class_files = {cls: [] for cls in CLASS_NAMES.keys()}
    
    # train 폴더만 분석 (여기에 증강 적용)
    labels_dir = os.path.join(SOURCE_DATASET_PATH, "train", "labels")
    images_dir = os.path.join(SOURCE_DATASET_PATH, "train", "images")
    
    if not os.path.exists(labels_dir):
        print(f"❌ 라벨 폴더를 찾을 수 없습니다: {labels_dir}")
        return None, None
    
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        try:
            with open(label_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    class_counts[class_id] += 1
                    
                    # 이미지 파일 경로 확인
                    img_name = os.path.basename(label_file).replace('.txt', '.jpg')
                    img_path = os.path.join(images_dir, img_name)
                    if os.path.exists(img_path):
                        class_files[class_id].append((img_path, label_file))
        except:
            continue
    
    print(f"\n📊 현재 클래스 분포:")
    for class_id, count in class_counts.items():
        class_name = CLASS_NAMES[class_id]
        target_count = count * CLASS_AUGMENTATION_MULTIPLIERS[class_id]
        print(f"  {class_name}: {count}개 → {target_count}개 (×{CLASS_AUGMENTATION_MULTIPLIERS[class_id]})")
    
    return class_counts, class_files

def augment_single_class(class_id, files_list, target_multiplier):
    """특정 클래스의 데이터를 증강"""
    
    class_name = CLASS_NAMES[class_id]
    current_count = len(files_list)
    target_count = current_count * target_multiplier
    need_count = target_count - current_count
    
    print(f"\n🎯 {class_name} 증강 시작:")
    print(f"  현재: {current_count}개 → 목표: {target_count}개")
    print(f"  생성 필요: {need_count}개")
    
    if need_count <= 0:
        print("  ✅ 증강 불필요")
        return 0
    
    if current_count == 0:
        print("  ❌ 원본 데이터 없음")
        return 0
    
    # 출력 폴더 설정
    output_images_dir = os.path.join(SOURCE_DATASET_PATH, "train", "images")
    output_labels_dir = os.path.join(SOURCE_DATASET_PATH, "train", "labels")
    
    generated_count = 0
    
    while generated_count < need_count:
        # 랜덤하게 원본 파일 선택
        img_path, label_path = random.choice(files_list)
        
        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # 라벨 로드
        with open(label_path, 'r') as f:
            label_lines = f.readlines()
        
        if not label_lines:
            continue
        
        # 랜덤 증강 기법 선택
        aug_type = random.choice(['rotation', 'brightness', 'noise', 'blur', 'horizontal_flip'])
        
        # 증강 적용
        aug_image = image.copy()
        aug_labels = label_lines.copy()
        aug_suffix = ""
        
        if aug_type == 'rotation':
            angle = random.choice(AUGMENTATION_SETTINGS["rotation_angles"])
            aug_image = apply_rotation_augmentation(aug_image, angle)
            aug_suffix = f"rot{angle}"
            
            # 바운딩박스도 회전 (첫 번째 객체만)
            if aug_labels:
                parts = aug_labels[0].strip().split()
                if len(parts) >= 5:
                    cls_id, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x_new, y_new, w_new, h_new = rotate_bounding_box(x, y, w, h, angle, 
                                                                   image.shape[1], image.shape[0])
                    aug_labels[0] = f"{cls_id} {x_new:.6f} {y_new:.6f} {w_new:.6f} {h_new:.6f}\n"
        
        elif aug_type == 'brightness':
            factor = random.choice(AUGMENTATION_SETTINGS["brightness_factors"])
            aug_image = apply_brightness_augmentation(aug_image, factor)
            aug_suffix = f"bright{factor:.1f}".replace('.', '')
        
        elif aug_type == 'noise':
            level = random.choice(AUGMENTATION_SETTINGS["noise_levels"])
            aug_image = apply_noise_augmentation(aug_image, level)
            aug_suffix = f"noise{level}"
        
        elif aug_type == 'blur':
            kernel = random.choice(AUGMENTATION_SETTINGS["blur_kernels"])
            aug_image = apply_blur_augmentation(aug_image, kernel)
            aug_suffix = f"blur{kernel}"
        
        elif aug_type == 'horizontal_flip':
            aug_image = cv2.flip(aug_image, 1)
            aug_suffix = "hflip"
            
            # 바운딩박스 x 좌표 반전
            new_aug_labels = []
            for line in aug_labels:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x_flipped = 1.0 - x  # x 좌표 반전
                    new_aug_labels.append(f"{cls_id} {x_flipped:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                else:
                    new_aug_labels.append(line)
            aug_labels = new_aug_labels
        
        # 새 파일명 생성
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_img_name = f"{base_name}_aug_{class_name}_{generated_count:03d}_{aug_suffix}.jpg"
        new_label_name = f"{base_name}_aug_{class_name}_{generated_count:03d}_{aug_suffix}.txt"
        
        # 저장
        new_img_path = os.path.join(output_images_dir, new_img_name)
        new_label_path = os.path.join(output_labels_dir, new_label_name)
        
        try:
            cv2.imwrite(new_img_path, aug_image)
            
            with open(new_label_path, 'w') as f:
                f.writelines(aug_labels)
            
            generated_count += 1
            
            if generated_count % 10 == 0:
                print(f"    진행상황: {generated_count}/{need_count}")
        
        except Exception as e:
            print(f"    ⚠️ 저장 오류: {e}")
            continue
    
    print(f"  ✅ 완료! {generated_count}개 생성")
    return generated_count

def main():
    """메인 실행 함수"""
    
    print("🚀 geonhui 데이터셋 소수 클래스 증강 도구")
    print("=" * 60)
    
    # 1. 현재 분포 분석
    class_counts, class_files = analyze_class_distribution()
    
    if class_counts is None:
        print("❌ 데이터 분석 실패!")
        return
    
    # 2. 소수 클래스만 선별 증강
    total_generated = 0
    
    for class_id in [0, 1, 2]:  # bottle-good(3) 제외
        multiplier = CLASS_AUGMENTATION_MULTIPLIERS[class_id]
        if multiplier > 1:  # 증강이 필요한 클래스만
            files_list = class_files[class_id]
            generated = augment_single_class(class_id, files_list, multiplier)
            total_generated += generated
    
    # 3. 결과 요약
    print(f"\n🎉 증강 완료!")
    print(f"총 {total_generated}개 파일 생성")
    
    # 4. 증강 후 분포 다시 확인
    print(f"\n📊 증강 후 예상 분포:")
    for class_id, count in class_counts.items():
        class_name = CLASS_NAMES[class_id]
        multiplier = CLASS_AUGMENTATION_MULTIPLIERS[class_id]
        final_count = count * multiplier
        print(f"  {class_name}: {final_count}개")
    
    print(f"\n💡 이제 다음 단계를 진행하세요:")
    print(f"  1️⃣ 증강된 데이터로 재학습")
    print(f"  2️⃣ 클래스 가중치 적용 학습")
    print(f"  3️⃣ 높은 임계값으로 추론 테스트")

if __name__ == "__main__":
    main()