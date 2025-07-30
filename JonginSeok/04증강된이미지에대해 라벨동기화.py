import os
import shutil

# 라벨 매핑 딕셔너리
label_map = {
    'bad-broken_large': '0',
    'bad-broken_small': '1',
    'bad-contamination': '2',
    'bottle-good': '3',
}
reverse_map = {v: k for k, v in label_map.items()}

# 경로 설정
labels_folder = '/Users/ngins/Git/python.yolo/dataset/ngins751206/labels/'
image_folder = '/Users/ngins/Git/python.yolo/dataset/ngins751206/images/'
output_root = './organized_output'

os.makedirs(output_root, exist_ok=True)

# 이미지 파일명(확장자 제외) → 전체 경로 딕셔너리 생성
image_dict = {}
for fname in os.listdir(image_folder):
    if os.path.isfile(os.path.join(image_folder, fname)):
        name, ext = os.path.splitext(fname)
        image_dict[name] = fname  # 확장자 포함된 원본 이름 저장

print(f"⚠️ image_dict : {image_dict}")

# 라벨 파일 처리
for label_file in os.listdir(labels_folder):
    if not label_file.endswith('.txt'):
        continue
    
    print(f"⚠️ label_file: {label_file}")
    label_path = os.path.join(labels_folder, label_file)

    with open(label_path, 'r') as f:
        line = f.readline().strip()
        parts = line.split()
        if len(parts) < 2:
            continue

        image_key = parts[0]  # 확장자 없는 이미지 이름
        # label_value = int(parts[1])
        # print(f"⚠️ image_key: {image_key}")

        # if label_value not in reverse_map:
        #     continue
        # label_name = reverse_map[label_value]
        label_name = reverse_map[image_key]
        # print(f"⚠️ label_name: {label_name}")

        class_folder = os.path.join(output_root, label_name)
        image_subdir = os.path.join(class_folder, 'images')
        label_subdir = os.path.join(class_folder, 'labels')
        os.makedirs(image_subdir, exist_ok=True)
        os.makedirs(label_subdir, exist_ok=True)

        # 확장자 분리
        basename, extension = os.path.splitext(label_file)

        # 🖼 이미지 복사
        # if image_key in image_dict:
        img_ext = 'jpg'
        img_nm = f"{basename}.{img_ext}"
        
        if basename in image_dict:
            src_img = os.path.join(image_folder, image_dict[basename])
            dst_img = os.path.join(image_subdir, image_dict[basename])

            shutil.copy(src_img, dst_img)
            if label_name != 'bottle-good' :
                print(f"✅ 이미지 복사됨 → {label_name}/images/{image_dict[basename]}")
            else:
                print(f"✅ 이미지 복사됨 ")
        else:     
            print(f"⚠️ 이미지 없음: {img_nm} basename:{basename} → {label_name}/images/{image_dict[basename]}  src_img:{src_img} dst_img:{dst_img}    .... label_name: {label_name}")
            

        # 📄 라벨 복사
        dst_label = os.path.join(label_subdir, label_file)
        shutil.copy(label_path, dst_label)
        print(f"📝 라벨 복사됨 → {label_name}/labels/{label_file}")
