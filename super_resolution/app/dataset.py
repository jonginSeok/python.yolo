from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as T


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.transform = transform or T.ToTensor()
        self.lr_images = sorted(list(self.lr_dir.glob("*.*")))
        self.hr_images = sorted(list(self.hr_dir.glob("*.*")))

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_images[idx]).convert("RGB")
        hr_img = Image.open(self.hr_images[idx]).convert("RGB")

        # HR 이미지를 LR 이미지에 맞게 크기 조정 후 scale 적용
        expected_hr_size = (lr_img.width * 3, lr_img.height * 3)
        hr_img = hr_img.resize(expected_hr_size, Image.BICUBIC)

        return self.transform(lr_img), self.transform(hr_img)


# from PIL import Image
# from pathlib import Path

# BASE_DIR = Path(__file__).parent.parent
# HR_DIR = BASE_DIR / 'data/train_HR_gen'
# LR_DIR = BASE_DIR / 'data/train_LR_gen'

# for lr_img_path, hr_img_path in zip(sorted(LR_DIR.glob("*.*")), sorted(HR_DIR.glob("*.*"))):
#     lr_img = Image.open(lr_img_path)
#     hr_img = Image.open(hr_img_path)
#     print(f"LR: {lr_img.size}, HR: {hr_img.size}")