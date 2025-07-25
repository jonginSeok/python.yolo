from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. CNN에서 feature만 추출
model.eval()
features, labels = [], []

with torch.no_grad():
    for images, lbls in train_loader:
        x = images.to(device)
        output = model.features(x)  # Conv features만
        flat = output.view(output.size(0), -1).cpu().numpy()
        features.extend(flat)
        labels.extend(lbls.numpy())

# 2. SVM 학습
scaler = StandardScaler()
X = scaler.fit_transform(features)
svm = SVC(kernel='rbf')
svm.fit(X, labels)
