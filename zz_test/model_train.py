import pandas as pd
from pathlib import Path

root = Path(__file__).parent.resolve()

csv_path = Path(root / 'runs/bottle/cls2/results.csv')
df = pd.read_csv(csv_path)
print(df.tail(1))  # 마지막 에포크 성능 출력
