#!/usr/bin/env python3
"""
현재 geonhui 데이터셋 구조 및 클래스 분포 정확 분석
"""

import os
import glob
from collections import Counter, defaultdict

# 클래스 매핑
CLASS_MAP = {
    0: "bad-broken_large",
    1: "bad-broken_small", 
    2: "bad-contamination",
    3: "bottle-good"
}

def analyze_current_dataset():
    """현재 geonhui 데이터셋 구조 분석"""
    
    print("🔍 현재 데이터셋 구조 분석")
    print("=" * 50)
    
    base_path = "geonhui"
    
    if not os.path.exists(base_path):
        print(f"❌ {base_path} 폴더를 찾을 수 없습니다!")
        return
    
    # 전체 구조 확인
    splits = ["train", "valid", "test"]
    total_stats = {
        "images": defaultdict(int),
        "labels": defaultdict(int),
        "class_instances": defaultdict(lambda: defaultdict(int)),
        "class_images": defaultdict(lambda: defaultdict(int))
    }
    
    for split in splits:
        print(f"\n📁 {split.upper()} 폴더 분석:")
        
        images_dir = os.path.join(base_path, split, "images")
        labels_dir = os.path.join(base_path, split, "labels")
        
        # 이미지 파일 수 확인
        if os.path.exists(images_dir):
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            
            print(f"  📷 이미지 파일: {len(image_files)}개")
            total_stats["images"][split] = len(image_files)
            
            # 각 이미지 파일명 출력 (디버깅용)
            if len(image_files) <= 10:  # 10개 이하면 모두 출력
                for img_file in image_files:
                    print(f"    - {os.path.basename(img_file)}")
            else:
                for i, img_file in enumerate(image_files[:5]):  # 처음 5개만
                    print(f"    - {os.path.basename(img_file)}")
                print(f"    ... 외 {len(image_files)-5}개")
        else:
            print(f"  ❌ 이미지 폴더 없음: {images_dir}")
        
        # 라벨 파일 수 및 클래스 분포 확인  
        if os.path.exists(labels_dir):
            label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
            print(f"  🏷️ 라벨 파일: {len(label_files)}개")
            total_stats["labels"][split] = len(label_files)
            
            # 클래스별 분석
            class_counter = Counter()
            images_with_class = defaultdict(set)
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    image_classes = set()
                    
                    for line in lines:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_counter[class_id] += 1
                                image_classes.add(class_id)
                    
                    # 이 이미지에 포함된 클래스들 기록
                    for class_id in image_classes:
                        images_with_class[class_id].add(os.path.basename(label_file))
                
                except Exception as e:
                    print(f"    ⚠️ 라벨 파일 오류 {os.path.basename(label_file)}: {e}")
            
            # 클래스별 통계 출력
            print(f"  📊 클래스별 객체 인스턴스 수:")
            for class_id in sorted(class_counter.keys()):
                class_name = CLASS_MAP.get(class_id, f"Unknown-{class_id}")
                instance_count = class_counter[class_id]
                image_count = len(images_with_class[class_id])
                
                print(f"    {class_name}: {instance_count}개 객체 (이미지 {image_count}개)")
                
                total_stats["class_instances"][split][class_id] = instance_count
                total_stats["class_images"][split][class_id] = image_count
                
                # 어떤 이미지에 포함되어 있는지 출력 (소수인 경우)
                if image_count <= 5:
                    for img_name in sorted(images_with_class[class_id]):
                        print(f"      - {img_name}")
        else:
            print(f"  ❌ 라벨 폴더 없음: {labels_dir}")
    
    return total_stats

def print_summary(stats):
    """전체 요약 출력"""
    
    print(f"\n📋 전체 요약")
    print("=" * 50)
    
    # 전체 이미지/라벨 수
    total_images = sum(stats["images"].values())
    total_labels = sum(stats["labels"].values())
    
    print(f"전체 이미지: {total_images}개")
    print(f"전체 라벨: {total_labels}개")
    
    # 클래스별 전체 통계
    print(f"\n📊 클래스별 전체 통계:")
    
    for class_id in sorted(CLASS_MAP.keys()):
        class_name = CLASS_MAP[class_id]
        
        total_instances = sum(stats["class_instances"][split].get(class_id, 0) 
                            for split in ["train", "valid", "test"])
        total_images = sum(stats["class_images"][split].get(class_id, 0) 
                         for split in ["train", "valid", "test"])
        
        print(f"\n  {class_name} (ID: {class_id}):")
        print(f"    총 객체 인스턴스: {total_instances}개")
        print(f"    포함된 이미지: {total_images}개")
        
        # 분할별 세부 정보
        for split in ["train", "valid", "test"]:
            instances = stats["class_instances"][split].get(class_id, 0)
            images = stats["class_images"][split].get(class_id, 0)
            if instances > 0 or images > 0:
                print(f"      {split}: {instances}개 객체, {images}개 이미지")
        
        # 문제점 분석
        if total_instances < 10:
            print(f"    ❌ 심각한 데이터 부족! (< 10개)")
        elif total_instances < 30:
            print(f"    ⚠️ 데이터 부족 (< 30개)")
        elif total_images < 10:
            print(f"    ⚠️ 이미지 다양성 부족 (< 10개 이미지)")
        else:
            print(f"    ✅ 양호한 데이터 수")

def check_data_balance():
    """데이터 불균형 정도 확인"""
    
    print(f"\n⚖️ 데이터 불균형 분석")
    print("=" * 50)
    
    stats = analyze_current_dataset()
    
    if not stats:
        return
    
    # 전체 클래스별 인스턴스 수 계산
    class_totals = {}
    for class_id in CLASS_MAP.keys():
        total = sum(stats["class_instances"][split].get(class_id, 0) 
                   for split in ["train", "valid", "test"])
        class_totals[class_id] = total
    
    if not any(class_totals.values()):
        print("❌ 유효한 데이터를 찾을 수 없습니다!")
        return
    
    max_instances = max(class_totals.values())
    min_instances = min(v for v in class_totals.values() if v > 0)
    
    print(f"최대 클래스 인스턴스: {max_instances}개")
    print(f"최소 클래스 인스턴스: {min_instances}개")
    print(f"불균형 비율: {max_instances/min_instances:.1f}:1")
    
    if max_instances / min_instances > 10:
        print("❌ 심각한 데이터 불균형!")
    elif max_instances / min_instances > 5:
        print("⚠️ 데이터 불균형 있음")
    else:
        print("✅ 상대적으로 균형있는 데이터")
    
    print_summary(stats)
    
    return stats

def main():
    """메인 함수"""
    
    print("🔍 YOLO 데이터셋 구조 및 클래스 분포 분석기")
    print("=" * 60)
    
    # 현재 작업 디렉토리 확인
    current_dir = os.getcwd()
    print(f"현재 작업 디렉토리: {current_dir}")
    
    # geonhui 폴더 존재 확인
    if os.path.exists("geonhui"):
        print("✅ geonhui 폴더 발견")
        
        # 상세 분석 실행
        stats = check_data_balance()
        
        if stats:
            print(f"\n💡 권장사항:")
            
            # 각 클래스별 권장사항
            for class_id in CLASS_MAP.keys():
                total_instances = sum(stats["class_instances"][split].get(class_id, 0) 
                                    for split in ["train", "valid", "test"])
                total_images = sum(stats["class_images"][split].get(class_id, 0) 
                                 for split in ["train", "valid", "test"])
                
                class_name = CLASS_MAP[class_id]
                
                if total_instances < 30:
                    print(f"  🎯 {class_name}: 데이터 증강 또는 추가 수집 필요")
                    print(f"     현재 {total_instances}개 → 최소 50개 목표")
    else:
        print("❌ geonhui 폴더를 찾을 수 없습니다!")
        
        # 현재 디렉토리의 폴더들 확인
        folders = [f for f in os.listdir('.') if os.path.isdir(f)]
        print(f"현재 디렉토리의 폴더들: {folders}")

if __name__ == "__main__":
    main()