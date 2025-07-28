import pandas as pd
from pathlib import Path

root = Path(__file__).parent.resolve()

csv_path = Path(root / 'runs/bottle/cls/results.csv')
df = pd.read_csv(csv_path)
last_epoch = df.tail(1)

train_loss = last_epoch[['train/box_loss', 'train/cls_loss', 'train/dfl_loss']]
val_loss = last_epoch[['val/box_loss', 'val/cls_loss', 'val/dfl_loss']]
accuracy = last_epoch[['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']]
learning_rate = last_epoch[['lr/pg0', 'lr/pg1', 'lr/pg2']]

print()
print("Train Loss")
print(train_loss.to_string(index=False), "\n")

print("Validation Loss")
print(val_loss.to_string(index=False), "\n")

print("Accuracy Metrics")
print(accuracy.to_string(index=False), "\n")

print("Learning Rate")
print(learning_rate.to_string(index=False))