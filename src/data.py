"""Data loading, validation, and stratified partitioning module."""
import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_FILE, RANDOM_SEED, TEST_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(data_path=None) -> pd.DataFrame:
    """Load the raw credit card transaction dataset, supporting single file or split parts."""
    from pathlib import Path
    path = Path(data_path) if data_path else DATA_FILE
    
    if path.exists():
        logger.info("Loading dataset from %s ...", path)
        df = pd.read_csv(path)
    else:
        # Check for split parts (part1 and part2)
        part1 = path.parent / f"{path.stem}_part1.csv"
        part2 = path.parent / f"{path.stem}_part2.csv"
        if part1.exists() and part2.exists():
            logger.info("Loading dataset from split parts: %s and %s ...", part1.name, part2.name)
            df1 = pd.read_csv(part1)
            df2 = pd.read_csv(part2)
            df = pd.concat([df1, df2], ignore_index=True)
        else:
            raise FileNotFoundError(f"Neither {path} nor parts ({part1.name}, {part2.name}) found.")

    logger.info("Dataset loaded successfully. Shape: %s", df.shape)
    
    # Validation checks
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        logger.warning("Found %d missing values! Imputing or checking required.", null_counts)
    else:
        logger.info("Validation passed: 0 missing values detected.")
        
    class_counts = df["Class"].value_counts().to_dict()
    fraud_pct = (class_counts.get(1, 0) / len(df)) * 100
    logger.info(
        "Class Distribution: Non-Fraud (0): %d, Fraud (1): %d (Fraud Rate: %.4f%%)",
        class_counts.get(0, 0),
        class_counts.get(1, 0),
        fraud_pct,
    )
    return df


def get_stratified_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into stratified training and testing sets.
    
    CRITICAL: Stratification ensures the exact empirical fraud ratio (~0.172%)
    is preserved across both training and test partitions without bias.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Stratified Split: Train Shape=%s (Frauds=%d, %.4f%%) | Test Shape=%s (Frauds=%d, %.4f%%)",
        X_train.shape,
        y_train.sum(),
        (y_train.sum() / len(y_train)) * 100,
        X_test.shape,
        y_test.sum(),
        (y_test.sum() / len(y_test)) * 100,
    )
    return X_train, X_test, y_train, y_test
