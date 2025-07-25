from pathlib import Path
from PIL import Image

lr_folder = Path("D:/학생방/YOLO/[01] super_resolution/data/train_LR")
output_lr_folder = lr_folder

for lr_img in lr_folder.glob("*.*"):
    ext = lr_img.suffix.lower()
    filename_no_ext = lr_img.stem
    if ext == ".gif":
        try:
            print(f"처리 중: {lr_img.name}")
            with Image.open(lr_img) as img:
                try:
                    img.seek(0)
                except EOFError:
                    pass  # 단일 프레임 GIF
                img = img.convert("RGB")
                new_file = output_lr_folder / f"{filename_no_ext}.png"
                img.save(new_file)
                print(f"✅ 변환 완료: {new_file.name}")
        except Exception as e:
            print(f"❌ 오류 발생: {lr_img.name} → {type(e).__name__}: {e}")
    else:
        print(f"복사 생략: {lr_img.name} (같은 위치)")

print(f"저장 시도 경로: {new_file}")
# print("✅ 전체 GIF 처리 완료.")