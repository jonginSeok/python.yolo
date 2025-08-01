import os
import glob

def load_boxes(file_path):
    # txt 파일에서 박스 좌표와 클래스 정보 읽기
    boxes = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])
            boxes.append((cls, x, y, w, h))
    return boxes

def iou(box1, box2):
    # 간단한 IoU 계산
    _, x1, y1, w1, h1 = box1
    _, x2, y2, w2, h2 = box2

    # convert to corner points
    boxA = [x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2]
    boxB = [x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou_score = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou_score

def evaluate_image(pred_path, label_path, iou_threshold=0.5):
    preds = load_boxes(pred_path)
    truths = load_boxes(label_path)

    matched = 0
    for truth in truths:
        for pred in preds:
            if truth[0] == pred[0] and iou(truth, pred) >= iou_threshold:
                matched += 1
                break

    if len(truths) == 0:
        return 1.0 if len(preds) == 0 else 0.0
    return matched / len(truths)

def main():
    prediction_dir = './predictions/'  # YOLO 예측 결과 저장 폴더
    label_dir = './labels/'            # 실제 정답 라벨 폴더

    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        image_name = os.path.basename(label_file)
        pred_file = os.path.join(prediction_dir, image_name)

        if not os.path.exists(pred_file):
            print(f"⚠️ Prediction missing for {image_name}")
            continue

        score = evaluate_image(pred_file, label_file)
        if score < 0.5:
            print(f"🔴 Low Accuracy ({score:.2f}): {image_name}")
        else:
            print(f"🟢 OK ({score:.2f}): {image_name}")

if __name__ == '__main__':
    main()