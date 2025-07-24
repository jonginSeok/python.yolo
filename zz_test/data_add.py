import os
import cv2
import glob

def flip_image_and_label(img_path, label_path, flip_mode):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[오류] 이미지를 불러올 수 없습니다: {img_path}")
        return

    h, w = img.shape[:2]
    flipped_img = cv2.flip(img, flip_mode)

    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path).split('.')[0]
    suffix = 'v' if flip_mode == 0 else 'h'
    new_img_name = f"{basename}_{suffix}.jpg"
    new_label_name = f"{basename}_{suffix}.txt"

    img_save_path = os.path.join(dirname, new_img_name)
    label_save_path = os.path.join(os.path.dirname(label_path), new_label_name)

    success = cv2.imwrite(img_save_path, flipped_img)
    if not success:
        print(f"[오류] 이미지 저장 실패: {img_save_path}")
        return
    else:
        print(f"[완료] 저장된 이미지: {img_save_path}")

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[오류] 라벨 파일 읽기 실패: {label_path} - {e}")
        return

    new_lines = []
    for line in lines:
        try:
            cls, x, y, w_box, h_box = map(float, line.strip().split())
            if flip_mode == 1:  # Horizontal
                x = 1.0 - x
            elif flip_mode == 0:  # Vertical
                y = 1.0 - y
            new_lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")
        except ValueError:
            print(f"[오류] 잘못된 라벨 형식: {line.strip()} (파일: {label_path})")
            return

    try:
        with open(label_save_path, 'w') as f:
            f.writelines(new_lines)
        print(f"[완료] 저장된 라벨: {label_save_path}")
    except Exception as e:
        print(f"[오류] 라벨 저장 실패: {label_save_path} - {e}")

def process_dir(images_dir, labels_dir):
    img_files = glob.glob(os.path.join(images_dir, "*.*"))  # 모든 확장자 지원
    if not img_files:
        print(f"[경고] 이미지 없음: {images_dir}")
        return

    print(f"[시작] 디렉토리 처리: {images_dir}")
    for img_path in img_files:
        # basename = os.path.basename(img_path).split('.')[0]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{basename}.txt")
        if not os.path.exists(label_path):
            print(f"[경고] 라벨 누락: {label_path}")
            continue

        print(f"→ 처리 중: {basename}")
        flip_image_and_label(img_path, label_path, 1)  # 좌우
        flip_image_and_label(img_path, label_path, 0)  # 상하

from pathlib import Path

root = Path(__file__).parent.resolve()

# 💡 여기에 디렉토리 경로를 설정하세요
train_img_dir = root / "data/train/images"
train_lbl_dir = root / "data/train/labels"
val_img_dir = root / "data/val/images"
val_lbl_dir = root / "data/val/labels"

# 🔄 실행
process_dir(train_img_dir, train_lbl_dir)
process_dir(val_img_dir, val_lbl_dir)
