# from torchvision import transforms
# 
# bottle_good_aug = transforms.Compose([
#     transforms.RandomRotation(degrees=180),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
#     transforms.CenterCrop(size=(400, 400)),
#     transforms.Resize(size=(640, 640)),
#     transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
#     transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
# ])

from ultralytics import YOLO
from torchvision import transforms
import os
from PIL import Image

# 1. bottle-good 클래스에 대한 데이터 증강
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    transforms.RandomRotation(40),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0))
])

input_dir = 'JonginSeok/dataset/bottle-good/images'
output_dir = 'JonginSeok/dataset/bottle-good-augmented'
os.makedirs(output_dir, exist_ok=True)

# 이미지 작업
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = Image.open(img_path)

    # 확장자 분리
    # basename, extension = os.path.splitext(img_name)

    for i in range(5):  # 5배 증강
        aug_img = augment(img)
        print(f"🟢 aug_img:({aug_img}) img_path:{img_path}")
        aug_img.save(os.path.join(output_dir, f"{img_name.split('.')[0]}_aug{i}.jpg"))
        # aug_img.save(os.path.join(os.path.join(output_dir, f"{basename}.jpg" )))

# 라벨 작업
label_input_dir = 'JonginSeok/dataset/bottle-good/labels'
label_output_dir = 'JonginSeok/dataset/bottle-good-augmented/labels'
os.makedirs(label_output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    label_name = img_name.replace('.jpg', '.txt')
    label_path = os.path.join(label_input_dir, label_name)
    print(f"🟢 label_name:({label_name}) label_path:{label_path}")
    for i in range(5):
        new_label_name = f"{img_name.split('.')[0]}_aug{i}.txt"
        new_label_path = os.path.join(label_output_dir, new_label_name)
        with open(label_path, 'r') as f:
            label_data = f.read()
        with open(new_label_path, 'w') as f:
            f.write(label_data)
