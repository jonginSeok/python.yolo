import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from pathlib import Path

# 클래스 이름 정의 (YOLO와 동일하게 유지)
CLASS_NAMES = ['bad-broken_large', 'bad-broken_small', 'bad-contamination', 'bottle-good']

# 데이터 경로
crop_base_dir = Path(__file__).parent.resolve() / "data/images/crops"

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 크기 표준화
    transforms.ToTensor()
])

# 데이터 로딩
train_dataset = datasets.ImageFolder(root=str(crop_base_dir), transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = datasets.ImageFolder(root=str(crop_base_dir).replace("train", "val"), transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# 사전학습 모델 로드 (ResNet18)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))  # 클래스 수에 맞게 출력 조정

# 손실 함수 & 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 학습 루프
for epoch in range(20):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/10, Loss: {total_loss:.4f}")
