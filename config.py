import os

# Store PostgreSQL database credentials in environment variables
DB_HOST = os.getenv("DB_HOST", "dpg-cvfsg6hopnds73bcr2gg-a")
DB_PORT = os.getenv("DB_PORT", "5432")  # Default PostgreSQL port
DB_USER = os.getenv("DB_USER", "spam_database_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "WY3goEw3okNjULxVNPJGhgbzZSWUk2ZF")
DB_NAME = os.getenv("DB_NAME", "spam_database")

# Construct the PostgreSQL connection URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Keep the optimal threshold unchanged
OPTIMAL_THRESHOLD = 0.24
