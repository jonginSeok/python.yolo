import os
import glob
import cv2

def load_boxes(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            boxes.append((cls, x, y, w, h))
    return boxes

def iou(box1, box2):
    _, x1, y1, w1, h1 = box1
    _, x2, y2, w2, h2 = box2

    # 중심 좌표 → 좌상단/우하단 좌표
    boxA = [x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2]
    boxB = [x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

def evaluate(label_file, pred_file, iou_threshold=0.5):

    # print(f'label_file:{label_file} pred_file:{pred_file} ')
    
    truths = load_boxes(label_file)
    preds = load_boxes(pred_file)

    matched = 0
    for truth in truths:
        for pred in preds:
            if truth[0] == pred[0] and iou(truth, pred) >= iou_threshold:
                matched += 1
                break

    return matched / len(truths) if truths else 0.0

def main():
    label_dir = 'JonginSeok/ngins7512/dataset/labels'
    image_dir = 'JonginSeok/ngins7512/dataset/images'
    pred_dir = 'JonginSeok/ngins7512/dataset/predictions'
    accuracy_threshold = 0.95 # 0.5  # 기준 정확도

    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        filename = os.path.basename(label_file)
        image_name = filename.replace('.txt', '.jpg')  # .png도 가능

        # print(f'filename:{filename} image_name:{image_name} ')

        pred_file = os.path.join(pred_dir, filename)
        image_path = os.path.join(image_dir, image_name)

        # print(f'pred_file:{pred_file} image_path:{image_path} ')

        if not os.path.exists(pred_file) or not os.path.exists(image_path):
            print(f"⚠️ 파일 없음: {filename}")
            continue

        score = evaluate(label_file, pred_file)
        
        if score < accuracy_threshold:
            print(f"🟥 제거됨 ({score:.2f}): {filename}, {image_name}")
            os.remove(label_file)
            os.remove(image_path)
        else:
            print(f"✅ 유지 ({score:.2f}): {filename}")

if __name__ == '__main__':
    main()