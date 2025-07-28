import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 📁 경로 설정
root = Path(__file__).parent.resolve()
model_dir = root / "model"
test_dir = root / "data/images/split_crops/test"

# 🏷️ 클래스 정의
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 📦 이미지 전처리 (학습과 동일해야 함!)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Letterbox 대신 Resize 사용 가능
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 📂 테스트셋 로드
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 📡 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 모델 정의 (학습과 동일 구조!)
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, len(CLASS_NAMES))
)
model = model.to(device)

# 🔐 학습된 파라미터 로드
model.load_state_dict(torch.load(model_dir / "best_val.pth"))
model.eval()

# 🧪 테스트 평가
all_preds, all_labels = [], []
test_correct = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(1)

        test_correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_correct / len(test_dataset)
print(f"\n🏁 Test Accuracy: {test_acc:.4f}")

# 📊 분석 저장 및 시각화
# df_results = pd.DataFrame({
#     "True Label": all_labels,
#     "Predicted Label": all_preds
# })
# df_results.to_csv(model_dir / "test_predictions.csv", index=False)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(model_dir / "test_confusion_matrix.png")
plt.show()

print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))