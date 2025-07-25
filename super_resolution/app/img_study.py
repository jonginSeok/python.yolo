from PIL import ImageFont, ImageDraw, Image, ImageFilter, ImageTransform
import numpy as np
from pathlib import Path
import os
import random


### 1. 노이즈 및 변형 함수 정의
def add_noise(img, stddev=50):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, stddev, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)

def apply_affine(img):
    width, height = img.size
    coeffs = (1, 0.2, -10,   # a, b, c
              0.1, 1, -5)    # d, e, f
    return img.transform((width, height), Image.AFFINE, coeffs, resample=Image.BICUBIC)

def apply_rotation(img, angle=4):
    return img.rotate(angle, resample=Image.BICUBIC, expand=True).crop((0, 0, img.size[0], img.size[1]))

def apply_random_crop_resize(img, crop_ratio=0.9):
    w, h = img.size
    crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
    left = np.random.randint(0, w - crop_w + 1)
    top = np.random.randint(0, h - crop_h + 1)
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    # return cropped.resize((w // 16, h // 16), resample=Image.BICUBIC)
    return cropped # 최종 resize는 밖에서 처리


### 2. HR + LR 페어 이미지 생성 함수
def generate_lr_pairs(hr_path, lr_folder, paired_hr_folder):
    img_hr = Image.open(hr_path)
    hr_w, hr_h = img_hr.size
    filename_stem = hr_path.stem
    target_lr_size = (150, 41)  # ESPCN 요구 입력 사이즈

    def save_pair(img_lr, suffix):
        filename = f"{filename_stem}_{suffix}.jpg"
        img_lr.save(lr_folder / filename)
        img_hr.save(paired_hr_folder / filename)

    # Case 1: Bicubic
    lr1 = img_hr.resize(target_lr_size, resample=Image.BICUBIC)
    save_pair(lr1, "bicubic")

    # Case 2: Low quality
    lr2 = img_hr.resize(target_lr_size, resample=Image.LANCZOS)
    lr2 = lr2.convert("P", palette=Image.ADAPTIVE, colors=4).convert("RGB")
    save_pair(lr2, "low_quality")

    # Case 3: Blurred
    blurred = img_hr.filter(ImageFilter.GaussianBlur(radius=10))
    lr3 = blurred.resize(target_lr_size, resample=Image.BICUBIC)
    save_pair(lr3, "blurred")

    # Case 4: Noisy
    noisy = add_noise(img_hr, stddev=50).resize(target_lr_size, resample=Image.BICUBIC)
    save_pair(noisy, "noisy")

    # Case 5: JPEG artifact
    temp_path = lr_folder / f"{filename_stem}_temp.jpg"
    img_hr.resize(target_lr_size, resample=Image.BICUBIC).save(temp_path, quality=2)
    jpeg_img = Image.open(temp_path)
    save_pair(jpeg_img, "jpeg_artifact")
    temp_path.unlink()

    # Case 6: Affine 변형
    affine = apply_affine(img_hr).resize(target_lr_size, resample=Image.BICUBIC)
    save_pair(affine, "affine")

    # Case 7: Rotation
    rotated = apply_rotation(img_hr).resize(target_lr_size, resample=Image.BICUBIC)
    save_pair(rotated, "rotated")

    # Case 8: Random Crop + Resize
    crop_resized = apply_random_crop_resize(img_hr, crop_ratio=0.9)
    crop_resized = crop_resized.resize(target_lr_size, resample=Image.BICUBIC)
    save_pair(crop_resized, "crop_resize")


### 3. 번호판 HR 생성 + 1x LR 기본 이미지 생성
output_hr_folder = Path(__file__).parent.parent / 'data/train_HR_gen'
output_lr_folder = Path(__file__).parent.parent / 'data/train_LR_gen'
paired_hr_folder = Path(__file__).parent.parent / 'data/train_HR_gen'
font_path = Path(__file__).parent.parent / 'data/fonts/hy-m-yoond1004.ttf'

# 디렉토리 준비
for folder in [output_hr_folder, output_lr_folder, paired_hr_folder]:
    os.makedirs(folder, exist_ok=True)

font_size_digit = 92
font_size_korean = 86

# 랜덤 샘플 정의
front_numbers = random.sample([f"{i:03d}" for i in range(1000)], 10)
korean_chars = random.sample(["가", "나", "다", "라", "마",
    "거", "너", "더", "러", "머", "버", "서", "어", "저", "고",
    "노", "도", "로", "모", "보", "소", "오", "조", "구", "누",
    "두", "루", "무", "부", "수", "우", "주", "바", "사", "아",
    "자", "허", "하", "호"], 5)
back_numbers = random.sample([f"{i:04d}" for i in range(10000)], 20)

# 번호판 HR/LR 기본 이미지 생성
for front in front_numbers:
    for kor in korean_chars:
        for back in back_numbers:
            text = f"{front}{kor} {back}"
            filename = f"{front}{kor}{back}.jpg"

            img_hr = Image.new('RGB', (600, 165), color='white')
            draw = ImageDraw.Draw(img_hr)
            x, y = 10, 30

            for ch in text:
                font_size = font_size_digit if ch.isdigit() else font_size_korean
                font = ImageFont.truetype(str(font_path), font_size)
                draw.text((x, y), ch, font=font, fill='black')
                bbox = draw.textbbox((x, y), ch, font=font)
                x += bbox[2] - bbox[0]

            # HR 저장
            hr_path = output_hr_folder / filename
            img_hr.save(hr_path)

            # 기본 LR 저장 (1/4 축소)
            lr_img = img_hr.resize((150, 41), resample=Image.BICUBIC)
            lr_img.save(output_lr_folder / filename)

print("✅ Step 1 완료: HR + 기본 LR 번호판 이미지 생성")

### 4. HR 모든 이미지에 대해 8가지 LR + HR 쌍 생성

hr_images = list(output_hr_folder.glob("*.jpg"))
print(f"🔍 총 {len(hr_images)}개의 HR 이미지 처리 시작")

for hr_img_path in hr_images:
    generate_lr_pairs(hr_img_path, output_lr_folder, paired_hr_folder)

print("✅ Step 2 완료: 모든 HR 이미지에 대해 LR/HR 짝 8쌍씩 생성 완료")