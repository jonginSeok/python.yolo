import os


def clean_txt_files(root_dir):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".txt"):
                file_path = os.path.join(foldername, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                    cleaned_content = content.replace("/n", "")  # '/n' 제거
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(cleaned_content)
                    print(
                        f"✅ Cleaned. content:[{content}] cleaned_content:[{cleaned_content}]"
                    )  # {file_path} {filename}
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")


# 사용 예시
target_directory = r"JonginSeok/dataset"
clean_txt_files(target_directory)
print("✅ 모든 라벨정보 텍스트 파일에서 /n 삭제 완료!")
