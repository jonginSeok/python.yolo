if __name__ == "__main__":
    import yaml
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from dataset import SRDataset
    from md2_EDSR import EDSR
    from pathlib import Path
    from torchvision.transforms import ToTensor
    from tqdm import tqdm

    # ───────────────────────
    # 설정 파일 로드
    with open("C:/Users/602-17/YOLO/super_resolution/app/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # ───────────────────────
    # 하이퍼파라미터
    scale = config["scale"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["learning_rate"]
    num_workers = config["num_workers"]

    # 디렉토리 설정
    BASE_DIR = Path(__file__).parent.parent
    HR_DIR = BASE_DIR / 'data/train_HR_gen'
    LR_DIR = BASE_DIR / 'data/train_LR_gen'
    MODEL_DIR = BASE_DIR / 'model'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / 'edsr.pth'                 # 최종 저장용
    checkpoint_path = MODEL_DIR / 'edsr_checkpoint.pth' # 중간 저장용

    # ───────────────────────
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ───────────────────────
    # 데이터 로더
    train_dataset = SRDataset(LR_DIR, HR_DIR, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # ───────────────────────
    # 모델, 손실 함수, 옵티마이저
    model = EDSR(scale_factor=scale).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ───────────────────────
    # 체크포인트 이어 학습
    start_epoch = 1
    if checkpoint_path.exists():
        print("🔁 Checkpoint 불러오는 중...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ {checkpoint['epoch']} epoch 이후부터 이어서 학습합니다.\n")

    # ───────────────────────
    # 학습 루프
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            preds = model(lr_imgs)
            loss = criterion(preds, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch} 완료 | 평균 Loss: {avg_loss:.6f}")

        # ───────────────────────
        # Checkpoint 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"💾 Checkpoint 저장됨: {checkpoint_path}")

    # ───────────────────────
    # 최종 모델 저장
    torch.save(model.state_dict(), model_path)
    print(f"🎉 학습 완료! 최종 모델 저장됨: {model_path}")
