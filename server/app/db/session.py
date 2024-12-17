from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# Base 클래스 선언
Base = declarative_base()

# SQLAlchemy 엔진 생성
engine = create_engine(settings.DATABASE_URL)

# 세션 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 데이터베이스 세션을 각 요청마다 생성 및 반환
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
