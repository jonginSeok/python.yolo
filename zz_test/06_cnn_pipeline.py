import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image

# 📌 클래스 정의
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 📁 디렉토리 경로 설정
root = Path(__file__).parent.resolve()
base_dir = root / "data/images/split_crops"
train_dir, val_dir, test_dir = base_dir / "train", base_dir / "val", base_dir / "test"

# 🧱 Letterbox 이미지 리사이즈 함수
def letterbox_image(image, target_size=(256, 256)):
    iw, ih = image.size
    w, h = target_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (128, 128, 128))
    new_image.paste(image_resized, ((w - nw) // 2, (h - nh) // 2))
    return new_image

# 🔍 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.Lambda(lambda img: letterbox_image(img, (256, 256))),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])     # 너무 튀는 데이터의 정규화 과정
])

# 📦 데이터셋 & 로더
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 📡 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📌 Using device:", device)

# 🧠 모델 정의 및 출력층 수정
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(0.3),  # 과적합 방지를 위해 뉴런 30% 무작위 정지
    nn.Linear(model.fc.in_features, len(CLASS_NAMES))
)
model = model.to(device)

# 🎯 손실함수, 옵티마이저, 스케줄러
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)   # 조금씩 천천히 학습하며 안정화를 위한 L2 정규화
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)   # 정교하게 수렴하여 점진적으로 조절해 성능 향상 도모

# ⏱ 하이퍼파라미터
EPOCHS = 40
# patience = 10
counter = 0
best_val_acc = 0.0
best_val_loss = float('inf')
best_overall_score = -float('inf')

# 📁 모델 저장 디렉토리
model_dir = root / "model"
model_dir.mkdir(parents=True, exist_ok=True)

# 🚀 학습 루프
for epoch in range(EPOCHS):
    # 🔧 훈련
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total

    # 🧪 검증
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    val_loss /= len(val_loader)

    # 📊 성능 출력
    print(f"\n📅 Epoch {epoch+1}")
    print(f"🔧 Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"🧪 Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

    # 💾 저장 조건: 정확도 & 손실 둘 다 개선된 경우에만 저장
    improved = False
    if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
        best_val_acc = val_acc
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_dir / "best_val.pth")
        print(f"✅ Saved best_val.pth (Epoch {epoch+1})")
        improved = True
    else:
        print("❌ Not improved on both accuracy and loss. Skipping save.")

    # 💡 "모든 지표 기반 점수" 계산
    overall_score = (val_acc * 2.0) - (val_loss * 1.0) + (train_acc * 1.0) - (train_loss * 0.5)
    
    # 💾 best_all.pth 저장 조건
    if overall_score > best_overall_score:
        best_overall_score = overall_score
        torch.save(model.state_dict(), model_dir / "best_all.pth")
        print(f"💎 Saved best_all.pth (Epoch {epoch+1}) with overall score {overall_score:.4f}")

    # ⛔ EarlyStopping
    # if not improved:
    #     counter += 1
    #     print(f"🕒 EarlyStopping counter: {counter}/{patience}")
    #     if counter >= patience:
    #         print("⛔ Training stopped due to no improvement.")
    #         break
    # else:
    #     counter = 0

    scheduler.step()

# 💾 최종 모델 저장
torch.save(model.state_dict(), model_dir / "last_model.pth")
print("📦 Final model saved as last_model.pth")
