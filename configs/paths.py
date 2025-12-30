"""
Path configurations for BharatTrend
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data subdirectories
EXTERNAL_DATA = DATA_DIR / "external"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"
VISUALIZATIONS = DATA_DIR / "visualizations"

# Model files
PRICE_MODEL = MODELS_DIR / "price_predictor.pkl"
CLUSTERING_MODEL = MODELS_DIR / "clustering_model.pkl"

# Log file
LOG_FILE = LOGS_DIR / "bharattrend.log"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, EXTERNAL_DATA, 
                  RAW_DATA, PROCESSED_DATA, VISUALIZATIONS]:
    directory.mkdir(parents=True, exist_ok=True)
