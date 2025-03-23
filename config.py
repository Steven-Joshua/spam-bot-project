import os
from dotenv import load_dotenv

# Load environment variables from .env file (only in local development)
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")  
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("Missing required database environment variables")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# Keep the optimal threshold unchanged
OPTIMAL_THRESHOLD = 0.24
