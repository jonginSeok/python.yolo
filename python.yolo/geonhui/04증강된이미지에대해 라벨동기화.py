#!/usr/bin/env python3
"""
geonhui 데이터셋 라벨 동기화 도구 - 올바른 경로 버전
클래스별로 이미지와 라벨을 정리하여 분석하기 쉽게 만듭니다.
"""

import os
import shutil
import glob
from collections import Counter

# =============================================================================
# 설정 섹션
# =============================================================================

# 클래스 이름과 번호 매핑
LABEL_MAP = {
    "0": "bad-broken_large",
    "1": "bad-broken_small",
    "2": "bad-contamination", 
    "3": "bottle-good",
}

# 처리할 데이터셋 선택 ('train', 'valid', 'test')
DATASET_TYPE = 'train'  # 기본값: train

# 기본 경로 설정 (geonhui 폴더 기준)
BASE_PATH = "geonhui"

def check_folder_structure():
    """현재 폴더 구조 확인"""
    
    print("🔍 현재 폴더 구조 확인...")
    
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(BASE_PATH, split, 'images')
        labels_dir = os.path.join(BASE_PATH, split, 'labels')
        
        print(f"\n📁 {split} 폴더:")
        
        if os.path.exists(images_dir):
            img_count = len(glob.glob(os.path.join(images_dir, "*.jpg")))
            print(f"  ✅ images: {img_count}개 파일")
        else:
            print(f"  ❌ images 폴더 없음: {images_dir}")
        
        if os.path.exists(labels_dir):
            lbl_count = len(glob.glob(os.path.join(labels_dir, "*.txt")))
            print(f"  ✅ labels: {lbl_count}개 파일")
        else:
            print(f"  ❌ labels 폴더 없음: {labels_dir}")

def organize_by_class(dataset_type='train'):
    """클래스별로 이미지와 라벨 정리"""
    
    print(f"\n🚀 '{dataset_type}' 데이터셋 클래스별 정리 시작")
    
    # 경로 설정 (올바른 geonhui 구조)
    images_folder = os.path.join(BASE_PATH, dataset_type, 'images')
    labels_folder = os.path.join(BASE_PATH, dataset_type, 'labels')
    output_root = os.path.join(BASE_PATH, f'{dataset_type}_organized')
    
    print(f"📂 이미지 폴더: {images_folder}")
    print(f"📄 라벨 폴더: {labels_folder}")
    print(f"📤 출력 폴더: {output_root}")
    
    # 폴더 존재 확인
    if not os.path.exists(images_folder):
        print(f"❌ 이미지 폴더를 찾을 수 없습니다: {images_folder}")
        return
    
    if not os.path.exists(labels_folder):
        print(f"❌ 라벨 폴더를 찾을 수 없습니다: {labels_folder}")
        return
    
    # 클래스별 통계
    class_stats = Counter()
    processed_files = 0
    error_files = []
    
    # 라벨 파일을 기준으로 처리
    label_files = glob.glob(os.path.join(labels_folder, "*.txt"))
    
    print(f"\n📋 총 {len(label_files)}개 라벨 파일 처리 중...")
    
    for label_path in label_files:
        label_filename = os.path.basename(label_path)
        
        try:
            # 라벨 파일에서 클래스 ID 읽기
            with open(label_path, 'r') as f:
                first_line = f.readline().strip()
                
            if not first_line:
                print(f"   ⚠️ 빈 파일: {label_filename}")
                error_files.append(label_filename)
                continue
            
            class_id = first_line.split()[0]
            
            # 클래스 이름 확인
            if class_id not in LABEL_MAP:
                print(f"   ⚠️ 알 수 없는 클래스 ID '{class_id}': {label_filename}")
                error_files.append(label_filename)
                continue
            
            class_name = LABEL_MAP[class_id]
            class_stats[class_name] += 1
            
            # 클래스별 출력 폴더 생성
            class_image_dir = os.path.join(output_root, class_name, 'images')
            class_label_dir = os.path.join(output_root, class_name, 'labels')
            os.makedirs(class_image_dir, exist_ok=True)
            os.makedirs(class_label_dir, exist_ok=True)
            
            # 대응하는 이미지 파일 찾기
            base_filename = os.path.splitext(label_filename)[0]
            
            # 여러 이미지 확장자 시도
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_path = os.path.join(images_folder, base_filename + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path is None:
                print(f"   ⚠️ 이미지 파일 없음: {base_filename}")
                error_files.append(label_filename)
                continue
            
            # 파일 복사
            dst_image_path = os.path.join(class_image_dir, os.path.basename(image_path))
            dst_label_path = os.path.join(class_label_dir, label_filename)
            
            shutil.copy2(image_path, dst_image_path)
            shutil.copy2(label_path, dst_label_path)
            
            processed_files += 1
            
            if processed_files % 20 == 0:
                print(f"   진행상황: {processed_files}/{len(label_files)}")
            
        except Exception as e:
            print(f"   ❌ 오류 ({label_filename}): {e}")
            error_files.append(label_filename)
            continue
    
    # 결과 요약
    print(f"\n📊 처리 결과:")
    print(f"  총 처리된 파일: {processed_files}개")
    print(f"  오류 파일: {len(error_files)}개")
    
    print(f"\n📈 클래스별 분포:")
    for class_name, count in sorted(class_stats.items()):
        print(f"  {class_name}: {count}개")
    
    if error_files:
        print(f"\n⚠️ 오류 파일 목록:")
        for error_file in error_files[:10]:  # 처음 10개만 표시
            print(f"    - {error_file}")
        if len(error_files) > 10:
            print(f"    ... 외 {len(error_files)-10}개")
    
    print(f"\n✅ 정리 완료! 결과는 '{output_root}' 폴더에 저장되었습니다.")
    
    return class_stats

def analyze_class_distribution():
    """전체 데이터셋의 클래스 분포 분석"""
    
    print(f"\n📊 전체 데이터셋 클래스 분포 분석")
    print("=" * 50)
    
    total_stats = Counter()
    
    for split in ['train', 'valid', 'test']:
        print(f"\n📁 {split} 분석:")
        
        labels_folder = os.path.join(BASE_PATH, split, 'labels')
        
        if not os.path.exists(labels_folder):
            print(f"  ❌ 폴더 없음: {labels_folder}")
            continue
        
        split_stats = Counter()
        label_files = glob.glob(os.path.join(labels_folder, "*.txt"))
        
        for label_path in label_files:
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            class_id = line.split()[0]
                            if class_id in LABEL_MAP:
                                class_name = LABEL_MAP[class_id]
                                split_stats[class_name] += 1
                                total_stats[class_name] += 1
            except:
                continue
        
        # 분할별 결과 출력
        for class_name in LABEL_MAP.values():
            count = split_stats.get(class_name, 0)
            print(f"    {class_name}: {count}개")
    
    # 전체 요약
    print(f"\n📋 전체 요약:")
    total_objects = sum(total_stats.values())
    
    for class_name in LABEL_MAP.values():
        count = total_stats.get(class_name, 0)
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  {class_name}: {count}개 ({percentage:.1f}%)")
        
        # 문제점 표시
        if count < 30:
            print(f"    ⚠️ 데이터 부족")
        elif percentage < 5:
            print(f"    ⚠️ 비율 부족")
    
    # 불균형 분석
    if total_stats:
        max_count = max(total_stats.values())
        min_count = min(v for v in total_stats.values() if v > 0)
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        
        print(f"\n⚖️ 데이터 불균형 분석:")
        print(f"  최대 클래스: {max_count}개")
        print(f"  최소 클래스: {min_count}개")
        print(f"  불균형 비율: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 10:
            print("  ❌ 심각한 불균형!")
        elif imbalance_ratio > 5:
            print("  ⚠️ 불균형 있음")
        else:
            print("  ✅ 상대적으로 균형적")

def main():
    """메인 실행 함수"""
    
    print("🗂️ geonhui 데이터셋 라벨 동기화 및 분석 도구")
    print("=" * 60)
    
    # 1. 폴더 구조 확인
    check_folder_structure()
    
    # 2. 전체 클래스 분포 분석
    analyze_class_distribution()
    
    # 3. 사용자 선택
    print(f"\n🎯 클래스별 정리 옵션:")
    print(f"  1. train 데이터 정리")
    print(f"  2. valid 데이터 정리") 
    print(f"  3. test 데이터 정리")
    print(f"  4. 모든 데이터 정리")
    
    try:
        choice = input("\n선택하세요 (1-4, 기본값: 1): ").strip()
        
        if choice == '2':
            organize_by_class('valid')
        elif choice == '3':
            organize_by_class('test')
        elif choice == '4':
            for split in ['train', 'valid', 'test']:
                organize_by_class(split)
        else:  # 기본값 또는 '1'
            organize_by_class('train')
            
    except KeyboardInterrupt:
        print(f"\n\n❌ 사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        # 기본값으로 train 실행
        organize_by_class('train')

if __name__ == "__main__":
    main()