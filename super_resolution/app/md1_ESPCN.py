# 빠르게 실험하고 싶다면 → SRCNN or ESPCN
# 정확도 높이고 싶다면 → EDSR or RCAN
# 실전 왜곡에 강한 복원 원하면 → Real-ESRGAN

# ESPCN	Real-time 가능, 속도 빠름	⭐⭐ 빠른 실험용. 변형 포함 시 900~2,700쌍(10, 5, 6)


from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# 하이퍼파라미터
SCALE = 3
BATCH_SIZE = 16
EPOCHS = 5
LR_SIZE = (41, 150)     # HR 크기(165, 600)을 scale=3으로 줄인 것
HR_SIZE = (123, 450)
LEARNING_RATE = 1e-3

# 디렉토리 설정
BASE_DIR = Path(__file__).parent.parent
HR_DIR = BASE_DIR / 'data/train_HR_gen'
LR_DIR = BASE_DIR / 'data/train_LR_gen'
MODEL_DIR = BASE_DIR / 'model'
MODEL_PATH = MODEL_DIR / 'espcn.pth'


# 1. 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_images = sorted(list(Path(lr_dir).glob("*.jpg")))
        self.hr_images = sorted(list(Path(hr_dir).glob("*.jpg")))
        assert len(self.lr_images) == len(self.hr_images), "Image pair count mismatch!"

        self.lr_transform = transforms.Compose([
            transforms.Resize(LR_SIZE),
            transforms.ToTensor()
        ])

        self.hr_transform = transforms.Compose([
            transforms.Resize(HR_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr = Image.open(self.lr_images[idx]).convert("RGB")
        hr = Image.open(self.hr_images[idx]).convert("RGB")
        return self.lr_transform(lr), self.hr_transform(hr)
    

# 2. ESPCN 모델 정의
class ESPCN(nn.Module):
    def __init__(self, scale_factor):
        super(ESPCN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 3 * (scale_factor ** 2), kernel_size=3, padding=1),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.features(x)
        x = self.pixel_shuffle(x)
        return x


# 3. 학습 함수
def train(model, dataloader, optimizer, criterion):
    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
        total_loss = 0

        for lr, hr in loop:
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} completed, Avg Loss: {total_loss / len(dataloader):.6f}")


# 4. 실행 코드
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomDataset(LR_DIR, HR_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ESPCN(scale_factor=SCALE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train(model, dataloader, optimizer, criterion)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")