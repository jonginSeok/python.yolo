import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 클래스 정의
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 디렉토리 경로 설정
root = Path(__file__).parent.resolve()
base_dir = root / "data/images/split_crops"
train_dir = base_dir / "train"
val_dir = base_dir / "val"
test_dir = base_dir / "test"

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 데이터 로더 구성
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 모델 정의 (ResNet18 + Dropout)
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, len(CLASS_NAMES))
)
model = model.to(device)

# 손실 함수, 옵티마이저, 스케줄러
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# 학습 파라미터
EPOCHS = 30
best_val_acc = 0.0

# 경로 설정
model_dir = root / "model"
model_dir.mkdir(parents=True, exist_ok=True)

# 💪 학습 시작
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct = 0.0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    scheduler.step()
    train_acc = correct / len(train_dataset)
    print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 📊 검증 단계
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)
    print(f"Validation Accuracy: {val_acc:.4f}")

    # 🧠 최적 성능 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_dir / "best_model.pth")
        print("✅ Saved best_model.pth")

    # 마지막 epoch에 최종 모델 저장
    if epoch == EPOCHS - 1:
        torch.save(model.state_dict(), model_dir / "last_model.pth")
        print("💾 Saved last_model.pth")

# 🧪 테스트 평가
model.load_state_dict(torch.load(model_dir / "best_model.pth"))  # best 모델 로드
model.eval()
test_correct, all_preds, all_labels = 0, [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(1)

        test_correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_correct / len(test_dataset)
print(f"\n🏁 Test Accuracy (best model): {test_acc:.4f}")

# 🔍 결과 저장 및 시각화
df_results = pd.DataFrame({
    "True Label": all_labels,
    "Predicted Label": all_preds
})
df_results.to_csv(model_dir / "predictions.csv", index=False)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(model_dir / "confusion_matrix.png")
plt.show()

print("\n📢 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
