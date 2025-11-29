from pathlib import Path

# Root of the project (assumes this file is in loan_mlops/utils/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Filenames for our two datasets
APPROVAL_DATA_FILE = RAW_DATA_DIR / "loan_approval.csv"
DEFAULT_DATA_FILE = RAW_DATA_DIR / "loan_default.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

APPROVAL_MODEL_FILE = MODELS_DIR / "approval_model.pkl"
DEFAULT_MODEL_FILE = MODELS_DIR / "default_model.pkl"