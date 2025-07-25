import torch
import torch.nn as nn
import torch.nn.functional as F

class BBoxCNN(nn.Module):
    def __init__(self):
        super(BBoxCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 4)  # Output: [x, y, w, h]

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.conv1(x))       # [B, 16, H, W]
        x = F.max_pool2d(x, 2)          # [B, 16, H/2, W/2]
        x = F.relu(self.conv2(x))       # [B, 32, H/2, W/2]
        x = F.max_pool2d(x, 2)          # [B, 32, H/4, W/4]
        x = F.relu(self.conv3(x))       # [B, 64, H/4, W/4]
        x = F.max_pool2d(x, 2)          # [B, 64, H/8, W/8]

        # Flatten and regression output
        x = x.view(x.size(0), -1)       # [B, 64 * H/8 * W/8]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                 # Bounding box coordinates
        return x


model = BBoxCNN()
dummy_input = torch.randn(1, 3, 64, 64)  # (Batch, Channels, Height, Width)
output = model(dummy_input)

print("Predicted bounding box:", output)  # 예: [x_center, y_center, width, height]