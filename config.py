import os

# Render automatically provides the database URL as an environment variable
DATABASE_URL = os.getenv("DATABASE_URL")  # Fetches PostgreSQL connection string

# Keep the optimal threshold unchanged
OPTIMAL_THRESHOLD = 0.24
