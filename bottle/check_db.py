import os
import django
from django.db import connection
from django.db.utils import OperationalError

# Django 환경 설정
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "config.settings"
)  # 프로젝트 이름으로 변경
django.setup()


def check_database_connection():
    try:
        connection.ensure_connection()
        if connection.is_usable():
            print("✅ PostgreSQL 데이터베이스 연결 성공!")
        else:
            print("⚠️ 연결은 되었지만 사용 불가능한 상태입니다.")
    except OperationalError as e:
        print("❌ 데이터베이스 연결 실패:", e)


if __name__ == "__main__":
    check_database_connection()
