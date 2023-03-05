from pathlib import Path
import os

SRC_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PATH = SRC_PATH.parents[0]

DATA_PATH = PROJECT_PATH / "data"
SPLIT_DATA_PATH = DATA_PATH / "split"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
RAW_DATA_PATH = DATA_PATH / "raw"
MODELS_PATH = PROJECT_PATH / "models"