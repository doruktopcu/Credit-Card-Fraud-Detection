"""Project configuration, paths, and hyperparameters."""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
DATA_FILE = DATA_DIR / "creditcard.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Train-Test Split Settings
TEST_SIZE = 0.20  # 80/20 stratified split
CV_FOLDS = 5

# Cost-Matrix Parameters for Fraud Detection (in USD / Business Units)
# Cost of False Negative (Missed Fraud): Estimated average fraud loss (~$120)
COST_FN = 120.0
# Cost of False Positive (False Alert / Manual Review / Customer Friction): ~$5
COST_FP = 5.0
