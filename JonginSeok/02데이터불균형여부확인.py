import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# train_valid = 'train'
'''
train 데이터 분석 결과:
- 총 209개의 라벨이 수집되었습니다.

🔢 클래스별 샘플 수:
- bad-broken_large: 0개
- bad-broken_small: 0개
- bad-contamination: 0개
- bottle-good: 209개
'''
train_valid = 'valid'
'''
valid 데이터 분석 결과:
- 총 63개의 라벨이 수집되었습니다.

🔢 클래스별 샘플 수:
- bad-broken_large: 20개
- bad-broken_small: 22개
- bad-contamination: 21개
- bottle-good: 0개
'''

# 📂 라벨 파일 경로 지정
# label_dir = '/Users/ngins/Git/python.yolo/dataset/origin/'+train_valid+'/labels'
label_dir = '/Users/ngins/Git/python.yolo/JonginSeok/ngins7512/labels/'
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

# 📊 시각화 설정 한글 방법1
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]

# # 📊 시각화 설정 한글 방법2
# # 나눔글꼴 경로 설정
# font_path = 'C:/Windows/Fonts/NanumGothic.ttf'
# # 폰트 이름 가져오기
# font_name = fm.FontProperties(fname=font_path).get_name()
# # 폰트 설정
# plt.rc('font', family=font_name) #출처: https://giveme-happyending.tistory.com/168 [소연의_개발일지:티스토리]

# 라벨 맵 정의
label_map = {
    'bad-broken_large': 0,
    'bad-broken_small': 1,
    'bad-contamination': 2,
    'bottle-good': 3,
}

reverse_map = {v: k for k, v in label_map.items()}
class_names = list(label_map.keys())

# 라벨 수집
labels = []

for file in label_files:
    file_path = os.path.join(label_dir, file)
    try:
        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()  # 공백 기준 분리
                if len(tokens) >= 1 and tokens[0].isdigit():
                    labels.append(int(tokens[0]))
                    print(f"[✅] '{file}'에서 라벨 '{tokens[0]}' 수집 완료")
                else:
                    print(f"[⚠️] '{file}'에서 비정상 라벨 구조 발견: {line}")
    except Exception as e:
        print(f"[🚫] '{file}' 읽기 실패: {e}")

print(f"총 {len(labels)}개의 라벨이 수집되었습니다.")

# 분석 및 시각화
label_series = pd.Series(labels)
label_named = label_series.map(reverse_map)
class_counts = label_named.value_counts().reindex(class_names, fill_value=0)

print("🔢 클래스별 샘플 수:")
for class_name, count in class_counts.items():
    print(f"- {class_name}: {count}개")

sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="muted", legend=False)
plt.title('클래스별 샘플 수 분포')
plt.xlabel('클래스명')
plt.ylabel('샘플 수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 불균형 판단(?)
max_count = class_counts.max()
min_count = class_counts.min()
imbalance_ratio = round(max_count / (min_count + 1e-5), 2)
print(f"📊 최대/최소 클래스 비율: {imbalance_ratio}")

if imbalance_ratio > 1.5:
    print("❗ 데이터 불균형이 존재할 가능성이 있습니다.")
else:
    print("✅ 데이터가 비교적 균형 잡혀 있습니다.")