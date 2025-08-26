# 사용자 정의 Dataset 클래스 추가
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 하이퍼파라미터 및 설정
BATCH_SIZE = 4
EPOCHS = 100
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("True여야 GPU 사용 가능 :", torch.cuda.is_available())  # True여야 GPU 사용 가능
print(f"사용 가능한 GPU({DEVICE}) 수:", torch.cuda.device_count())  # 사용 가능한 GPU 수

# 라벨 맵 정의
label_map = {
    "bad-broken_large": 0,
    "bad-broken_small": 1,
    "bad-contamination": 2,
    "bottle-good": 3,
}
class_names = list(label_map.keys())

data_path = "JonginSeok/dataset/cnn"


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.samples = []
        self.transform = transform
        self.label_map = label_map  # 예: {'bad-broken_large':0,'bad-broken_small':1,'bad-contamination':2,'bottle-good':3}

        for class_name, label in label_map.items():
            class_path = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".png")):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(
        self, idx
    ):  # DataLoader가 호출(직접호출하지 않음), 데이터 1set(문제, 정답)를 조회할 때
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return (
            image,
            label,
        )  # 가로, 세로, 넓이 ... 등을 추가하여 더  많은 정보를 전달한다.


# 2. 데이터 전처리
# 이미지를 전처리(Preprocessing) 하기 위한 연속된 변환 작업(transform pipeline)을 정의
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),  # 이미지를 고정 크기로 설정
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
        transforms.Normalize(
            [0.5], [0.5]
        ),  # 빠르고 안정적인 학습을 위한 정규화(0~1 -> -1~1), (x-0.5)/0.5
    ]
)

# 커스텀 Dataset 적용
train_dataset = CustomImageDataset(
    root_dir=data_path + "/train", label_map=label_map, transform=transform
)
valid_dataset = CustomImageDataset(
    root_dir=data_path + "/valid", label_map=label_map, transform=transform
)

# shuffle=True : 모델이 순서에 영향을 받지 않도록 매 epoch마다 무작위로 섞는다
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False
)  # 데이터 순서 고정


# 3. 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(3채널(RGB), 필터수, 필터크기, stride=1, padding=0)
            nn.Conv2d(
                3, 16, 3, padding=1
            ),  # 128x128x3 -> 128x128x16, padding=1은 1픽셀 추가하여 출력크기 유지
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 64x64x16, 이미지 크기를 1/2로 축소(국소적 특징 요약)
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 32x32x32
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                32 * 32 * 32, 128
            ),  # 입력은 CNN에서 전달된 크기, 출력은 보통 64, 128, 256, 512 등
            nn.ReLU(),
            # nn.Linear(128, 2)        # 최종 출력이 1이면 Sigmoid연결, 2이면 Softmax연결
            nn.Linear(
                128, 4
            ),  # BCEWithLogitsLoss() (또는 BCELoss + Sigmoid),	CrossEntropyLoss() (Softmax 포함)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()  # Softmax 포함
optimizer = optim.Adam(model.parameters(), lr=LR)

# 4. 학습 및 시각화용 리스트
train_acc_list, valid_acc_list = [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total, loss_total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)
    train_acc = correct / total
    train_acc_list.append(train_acc)

    # 검증
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(DEVICE), y.to(
                DEVICE
            )  # Tensor 데이터를 지정된 디바이스(CPU 또는 GPU)로 이동시키고, 새 참조 리턴
            outputs = model(x)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    valid_acc = correct / total
    valid_acc_list.append(valid_acc)

    print(
        f"Epoch({DEVICE}) {epoch+1} | Loss: {loss_total:.4f} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}"
    )
    # if (epoch + 1) % 10 == 0:
    #     print(f"Epoch({DEVICE}) {epoch+1} | Loss: {loss_total:.4f} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")

# 5. 학습 시각화
plt.plot(loss_total, label="Loss_total")
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(valid_acc_list, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Progress")
plt.legend()
plt.grid(True)
plt.show()

# 6. 모델 저장
torch.save(model.state_dict(), "bottle_v01_cnn.pth")

# 7. 모델 로드 (예시)
model.load_state_dict(torch.load("bottle_v01_cnn.pth", map_location=DEVICE))
model.eval()


# 8. 실제 이미지 예측 함수
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    output = model(image_tensor)
    pred = output.argmax(1).item()
    plt.imshow(np.array(image))
    plt.title(f"Prediction: {class_names[pred]}")
    plt.axis("off")
    plt.show()


# 9. 예측 실행 예시
predict_image(
    data_path
    + "/test/bad-contamination/012_png.rf.649902ef1adcb13616394303e1fd0bdd_ratio80.jpg"
)  # 실제 파일 경로 지정
