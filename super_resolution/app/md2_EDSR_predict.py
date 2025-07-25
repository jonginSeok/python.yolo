import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageEnhance
from pathlib import Path
import easyocr
import numpy as np
import re
import cv2

from md2_EDSR import EDSR  # EDSR 모델 정의 import

# ────────────────────────────────
# 하이퍼파라미터 및 경로 설정
SCALE = 3
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'model/edsr.pth'
TEST_IMG_DIR = BASE_DIR / 'data/train_LR_gen'
target_prefix = "181모7661"  # 테스트할 번호판 시작 문자열

# ────────────────────────────────
# 장치 설정 및 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EDSR(scale_factor=SCALE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ────────────────────────────────
# OCR 리더 초기화
reader = easyocr.Reader(['ko', 'en'])

# OCR 혼동 보정용 매핑 테이블 - 후처리 성능 개선(노가다 필요)
ocr_correction = {
    'I': '1', 'J': '3', 'O': '0', 'U': '나',
    'T': '7', 'Z': '2', 'S': '5', 'Y': '9',
    'B': '8', 'A': '4', 'G': '9', 'L': '1',
    'N': '7',
}

# 한글 유사 문자 보정 매핑 테이블
hangul_correction = {
    '오': '모',
    '모': '오',  # 필요시 상호 변환도 가능
}

def apply_correction(text):
    # 숫자/영문 OCR 보정
    corrected = ''.join(ocr_correction.get(ch, ch) for ch in text.upper())
    # 한글 OCR 보정
    corrected = ''.join(hangul_correction.get(ch, ch) for ch in corrected)
    return corrected

# ────────────────────────────────
# 번호판 문자열 보정 함수
def correct_plate(texts):
    merged = ''.join(texts).replace(" ", "").replace("/", "")
    
    # 숫자/영문 OCR 보정
    merged = ''.join(ocr_correction.get(ch, ch) for ch in merged.upper())

    # 중간 한글 위치 보정: ambiguous한 경우 후보군 탐색
    pattern = r"(\d{3})([가-힣])(\d{4})"

    candidates = []
    # 먼저 원래 텍스트로 시도
    if re.match(pattern, merged):
        candidates.append(merged)
    
    # ambiguous한 경우 오/모 양쪽 대입
    if len(merged) == 8:
        for h in ['오', '모']:
            variant = merged[:3] + h + merged[3:]
            if re.match(pattern, variant):
                candidates.append(variant)
    
    # 중복 제거 및 우선순위 부여
    candidates = list(dict.fromkeys(candidates))  # 순서 유지하면서 중복 제거
    
    # 최종 선택
    if candidates:
        best = candidates[0]
        return best[:4] + " " + best[4:]
    
    return merged

# ────────────────────────────────
# affine 이미지 보정
def correct_affine(img):
    return img.rotate(-5, expand=True, fillcolor=(255, 255, 255))

# ────────────────────────────────
# blurred 이미지 전처리
def preprocess_for_blurred(image: Image.Image) -> Image.Image:
    img_cv = np.array(image.convert("RGB"))[..., ::-1]  # BGR

    # (1) Unsharp masking 방식 sharpening
    gaussian = cv2.GaussianBlur(img_cv, (9, 9), 10.0)
    sharpened = cv2.addWeighted(img_cv, 1.5, gaussian, -0.5, 0)

    # (2) CLAHE로 대비 향상
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    merged = cv2.merge((l_clahe, a, b))
    contrast_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return Image.fromarray(contrast_img[:, :, ::-1])  # BGR → RGB

# ────────────────────────────────
# 이미지 경로를 받아 SR + OCR + 보정까지 수행
def predict_text_from_image(img_path: Path) -> str:
    img = Image.open(img_path).convert("RGB")
    img_name = img_path.name

    # blurred 이미지 전처리
    if "blurred" in img_name.lower():
        img = preprocess_for_blurred(img)

    # affine 이미지 전처리
    if "affine" in img_name.lower():
        img = correct_affine(img)

    # SR 추론
    input_tensor = ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).clamp(0, 1)
    sr_img = ToPILImage()(output.cpu())

    # 추가 sharpness 보정
    sr_img = ImageEnhance.Sharpness(sr_img).enhance(1.1)

    # OCR
    result = reader.readtext(np.array(sr_img), detail=0)
    print(f"OCR 원본: {result}")

    # 번호판 문자열 보정
    corrected = correct_plate(result)
    print(f"보정된 결과: {corrected}")
    return corrected

# ────────────────────────────────
# 이미지 파일들 처리 루프
for img_path in TEST_IMG_DIR.glob(f"{target_prefix}*.jpg"):
    print(f"\n처리 중: {img_path.name}")
    predict_text_from_image(img_path)