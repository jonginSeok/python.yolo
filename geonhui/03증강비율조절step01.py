#!/usr/bin/env python3
"""
데이터 불균형 해결 - geonhui 데이터셋 기준
실제 데이터 현황: bottle-good(209) vs 소수클래스(20-22개)
"""

import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
import glob
from collections import Counter

# =============================================================================
# 1단계: 소수 클래스 데이터 증강 (20개 → 100개)
# =============================================================================

def augment_image_with_bbox(image, bbox_line, augmentation_type):
    """이미지와 바운딩박스를 함께 증강"""
    
    height, width = image.shape[:2]
    
    # 바운딩박스 파싱
    parts = bbox_line.strip().split()
    if len(parts) < 5:
        return None, None
    
    class_id = int(parts[0])
    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
    
    # 증강 적용
    if augmentation_type == "horizontal_flip":
        image = cv2.flip(image, 1)
        x_center = 1.0 - x_center  # x 좌표 반전
        
    elif augmentation_type == "brightness_up":
        # 밝기 증가
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    elif augmentation_type == "brightness_down":
        # 밝기 감소
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.subtract(hsv[:, :, 2], 30)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    elif augmentation_type == "rotate_5":
        # 5도 회전
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
    elif augmentation_type == "rotate_-5":
        # -5도 회전
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -5, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
    elif augmentation_type == "noise":
        # 노이즈 추가
        noise = np.random.randint(0, 25, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
    elif augmentation_type == "blur":
        # 블러 효과
        image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # 새로운 바운딩박스 라인 생성
    new_bbox_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
    
    return image, new_bbox_line

def augment_minority_classes():
    """소수 클래스 데이터 증강"""
    
    print("🚀 소수 클래스 데이터 증강 시작!")
    print("목표: 각 클래스 100개씩")
    
    base_path = "geonhui"
    train_images_dir = os.path.join(base_path, "train", "images")
    train_labels_dir = os.path.join(base_path, "train", "labels")
    
    # 소수 클래스 정의 (bottle-good 제외)
    minority_classes = [0, 1, 2]  # bad-broken_large, bad-broken_small, bad-contamination
    target_count = 100  # 각 클래스별 목표 개수
    
    # 증강 기법들
    augmentations = [
        "horizontal_flip",
        "brightness_up", 
        "brightness_down",
        "rotate_5",
        "rotate_-5", 
        "noise",
        "blur"
    ]
    
    # 현재 클래스별 데이터 수집
    class_files = {cls: [] for cls in minority_classes}
    
    for label_file in glob.glob(os.path.join(train_labels_dir, "*.txt")):
        try:
            with open(label_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    if class_id in minority_classes:
                        img_name = os.path.basename(label_file).replace('.txt', '.jpg')
                        img_path = os.path.join(train_images_dir, img_name)
                        if os.path.exists(img_path):
                            class_files[class_id].append((img_path, label_file))
        except:
            continue
    
    # 각 클래스별 증강 수행
    for class_id in minority_classes:
        class_name = ["bad-broken_large", "bad-broken_small", "bad-contamination"][class_id]
        current_files = class_files[class_id]
        current_count = len(current_files)
        
        print(f"\n🎯 {class_name} 증강:")
        print(f"  현재: {current_count}개 → 목표: {target_count}개")
        
        if current_count == 0:
            print("  ❌ 원본 데이터가 없습니다!")
            continue
        
        # 필요한 증강 수 계산
        need_count = target_count - current_count
        if need_count <= 0:
            print("  ✅ 이미 충분합니다!")
            continue
        
        # 증강 생성
        generated = 0
        while generated < need_count:
            # 랜덤하게 원본 파일 선택
            img_path, label_path = random.choice(current_files)
            
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
            aug_type = random.choice(augmentations)
            
            # 증강 적용 (첫 번째 객체만)
            aug_image, aug_bbox = augment_image_with_bbox(image, label_lines[0], aug_type)
            
            if aug_image is None:
                continue
            
            # 새 파일명 생성
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            new_img_name = f"{base_name}_aug_{generated:03d}_{aug_type}.jpg"
            new_label_name = f"{base_name}_aug_{generated:03d}_{aug_type}.txt"
            
            # 저장
            new_img_path = os.path.join(train_images_dir, new_img_name)
            new_label_path = os.path.join(train_labels_dir, new_label_name)
            
            cv2.imwrite(new_img_path, aug_image)
            
            with open(new_label_path, 'w') as f:
                f.write(aug_bbox)
                # 나머지 객체들도 원본 그대로 추가
                for line in label_lines[1:]:
                    f.write(line)
            
            generated += 1
            
            if generated % 10 == 0:
                print(f"    진행상황: {generated}/{need_count}")
        
        print(f"  ✅ 완료! {generated}개 추가 생성")

# =============================================================================
# 2단계: 클래스 가중치 극대화 학습
# =============================================================================

def create_extreme_weighted_training():
    """극도로 높은 클래스 가중치 학습"""
    
    script_content = '''#!/usr/bin/env python3
"""
극도로 높은 클래스 가중치 학습 - 불균형 해결
"""

from ultralytics import YOLO
import torch

def train_with_extreme_weights():
    """극도로 높은 가중치로 학습"""
    
    print("🔥 극도 가중치 학습 시작!")
    
    # 클래스별 손실 가중치 (bottle-good 대비)
    # bottle-good: 209개 vs 소수클래스: ~20개 = 약 10배 차이
    class_weights = [
        10.0,  # bad-broken_large (20개)
        10.0,  # bad-broken_small (22개) 
        10.0,  # bad-contamination (21개)
        1.0    # bottle-good (209개)
    ]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"디바이스: {device}")
    
    model = YOLO('yolo11n.pt')
    
    # 1단계: 극도 가중치 학습
    print("\\n1단계: 극도 가중치 학습")
    results1 = model.train(
        data='geonhui/data.yaml',
        epochs=200,
        batch=16,
        imgsz=640,
        device=device,
        
        # 극도 가중치 설정
        cls=10.0,              # 분류 손실 가중치 10배
        box=1.0,               # 박스 손실
        dfl=2.0,               # Distribution Focal Loss
        
        # 학습률 설정
        lr0=0.01,
        lrf=0.001,
        
        # 정규화 강화 (오버피팅 방지)
        weight_decay=0.001,
        momentum=0.937,
        
        # 데이터 증강 강화
        mixup=0.3,             # 이미지 혼합
        copy_paste=0.5,        # 객체 복사-붙여넣기
        
        project='geonhui/result',
        name='extreme_weighted_v1',
        exist_ok=True,
        patience=50,
        save_period=20,
    )
    
    # 2단계: 세밀 조정
    print("\\n2단계: 세밀 조정")
    model_stage2 = YOLO('geonhui/result/extreme_weighted_v1/weights/best.pt')
    
    results2 = model_stage2.train(
        data='geonhui/data.yaml',
        epochs=100,
        batch=8,               # 작은 배치로 세밀 조정
        imgsz=640,
        device=device,
        
        # 더욱 극단적 설정
        cls=20.0,              # 분류 손실 가중치 20배!
        box=0.5,               # 박스 손실 감소
        
        # 매우 낮은 학습률
        lr0=0.0005,
        lrf=0.00005,
        
        project='geonhui/result',
        name='extreme_weighted_v2',
        exist_ok=True,
        patience=30,
    )
    
    return results1, results2

if __name__ == "__main__":
    print("=== 극도 가중치 학습 전략 ===")
    
    stage1, stage2 = train_with_extreme_weights()
    
    print("\\n🎉 학습 완료!")
    print("결과 확인:")
    print("  - geonhui/result/extreme_weighted_v1/")
    print("  - geonhui/result/extreme_weighted_v2/")
'''
    
    with open("geonhui/extreme_weighted_training.py", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("🔥 극도 가중치 학습 스크립트 생성완료!")

# =============================================================================
# 3단계: 매우 높은 임계값 추론
# =============================================================================

def create_high_threshold_inference():
    """매우 높은 임계값 추론 스크립트"""
    
    script_content = '''#!/usr/bin/env python3
"""
매우 높은 임계값 추론 - False Positive 최대 제거
"""

import os
from ultralytics import YOLO

# 매우 높은 임계값 (False Positive 최대 제거)
HIGH_THRESHOLDS = {
    0: 0.6,   # bad-broken_large
    1: 0.85,  # bad-broken_small (매우 높음)
    2: 0.9,   # bad-contamination (극도로 높음)
    3: 0.4,   # bottle-good (낮춤)
}

CLASS_NAMES = {
    0: 'bad-broken_large',
    1: 'bad-broken_small', 
    2: 'bad-contamination',
    3: 'bottle-good'
}

def high_threshold_test():
    """높은 임계값으로 테스트"""
    
    # 최신 모델 로드
    model_paths = [
        'geonhui/result/extreme_weighted_v2/weights/best.pt',
        'geonhui/result/extreme_weighted_v1/weights/best.pt',
        'geonhui/result/yolo11n_bottle_4class4/weights/best.pt'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ 모델을 찾을 수 없습니다!")
        return
    
    print(f"📦 모델 로드: {model_path}")
    model = YOLO(model_path)
    
    # 테스트 실행
    test_dir = "geonhui/test/images"
    if not os.path.exists(test_dir):
        print(f"❌ 테스트 디렉토리 없음: {test_dir}")
        return
    
    results_summary = {cls: 0 for cls in CLASS_NAMES.keys()}
    precision_stats = {cls: {'correct': 0, 'total': 0} for cls in CLASS_NAMES.keys()}
    
    for img_file in os.listdir(test_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_dir, img_file)
            
            # 예측 수행
            results = model(img_path, conf=0.1, save=False)
            
            print(f"\\n📷 {img_file}:")
            
            for result in results:
                if result.boxes is not None:
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for score, cls in zip(scores, classes):
                        threshold = HIGH_THRESHOLDS.get(cls, 0.5)
                        class_name = CLASS_NAMES[cls]
                        
                        if score >= threshold:
                            print(f"  ✅ {class_name}: {score:.3f} (임계값: {threshold})")
                            results_summary[cls] += 1
                        else:
                            print(f"  ❌ {class_name}: {score:.3f} (임계값: {threshold}) - 제거")
    
    print(f"\\n📊 높은 임계값 적용 결과:")
    for cls, count in results_summary.items():
        class_name = CLASS_NAMES[cls]
        print(f"  {class_name}: {count}개 탐지")
    
    print(f"\\n💡 False Positive가 크게 감소했는지 확인하세요!")

if __name__ == "__main__":
    high_threshold_test()
'''
    
    with open("geonhui/high_threshold_test.py", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("🎯 높은 임계값 테스트 스크립트 생성완료!")

# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    
    print("🎯 데이터 불균형 해결 - 실전 솔루션")
    print("=" * 50)
    print("현재 상황:")
    print("  - bottle-good: 209개 (10배 많음)")
    print("  - 소수 클래스: 20-22개 (부족)")
    print("  - 결과: False Positive 다발생")
    
    print(f"\n📋 해결 계획:")
    print(f"  1️⃣ 소수 클래스 데이터 증강 (20개 → 100개)")
    print(f"  2️⃣ 극도 클래스 가중치 학습 (10-20배)")
    print(f"  3️⃣ 매우 높은 임계값 적용 (0.85, 0.9)")
    
    # 1단계: 데이터 증강
    print(f"\n🚀 1단계: 데이터 증강 실행...")
    try:
        augment_minority_classes()
        print("✅ 데이터 증강 완료!")
    except Exception as e:
        print(f"❌ 데이터 증강 실패: {e}")
    
    # 2단계: 학습 스크립트 생성
    print(f"\n🔥 2단계: 극도 가중치 학습 스크립트 생성...")
    create_extreme_weighted_training()
    
    # 3단계: 추론 스크립트 생성
    print(f"\n🎯 3단계: 높은 임계값 테스트 스크립트 생성...")
    create_high_threshold_inference()
    
    print(f"\n🎉 모든 도구 준비 완료!")
    print(f"\n🚀 실행 순서:")
    print(f"  1️⃣ python geonhui/extreme_weighted_training.py  # 학습")
    print(f"  2️⃣ python geonhui/high_threshold_test.py       # 테스트")
    
    print(f"\n📈 예상 개선 효과:")
    print(f"  bad-broken_small: 67% → 85%+ (목표)")
    print(f"  bad-contamination: 81.7% → 90%+ (목표)")

if __name__ == "__main__":
    main()