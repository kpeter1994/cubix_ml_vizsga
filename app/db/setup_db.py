from database import engine
from app.models import Base

# Base.metadata.drop_all(bind=engine)

Base.metadata.create_all(bind=engine)

print("✅ Creating tables in the database...")
