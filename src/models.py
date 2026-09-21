"""Model factory providing baseline, cost-weighted, and state-of-the-art classifiers."""
import logging
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.config import RANDOM_SEED
from src.features import FraudFeaturePreprocessor
from src.resampling import build_resampling_pipeline

logger = logging.getLogger(__name__)


def get_model_portfolio() -> Dict[str, Any]:
    """
    Returns a dictionary of competitive models across linear baselines,
    bagging ensembles, modern gradient boosters, and resampling pipelines.
    """
    models = {}

    # 1. Calibrated Cost-Sensitive Logistic Regression (with Feature Preprocessing)
    models["Logistic Regression (Balanced)"] = build_resampling_pipeline(
        preprocessor=FraudFeaturePreprocessor(),
        classifier=LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        ),
        resampling_strategy=None,
    )

    # 2. Balanced Random Forest (100 Trees with balanced subsample)
    models["Random Forest (Balanced)"] = build_resampling_pipeline(
        preprocessor=FraudFeaturePreprocessor(),
        classifier=RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        resampling_strategy=None,
    )

    # 3. XGBoost (Extreme Gradient Boosting with depth control & mild weight)
    models["XGBoost (Cost-Weighted)"] = build_resampling_pipeline(
        preprocessor=FraudFeaturePreprocessor(),
        classifier=XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            scale_pos_weight=5.0,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        resampling_strategy=None,
    )

    # 4. CatBoost (Categorical & Numeric Gradient Boosting with symmetric trees)
    models["CatBoost (Balanced)"] = build_resampling_pipeline(
        preprocessor=FraudFeaturePreprocessor(),
        classifier=CatBoostClassifier(
            iterations=250,
            learning_rate=0.08,
            depth=5,
            auto_class_weights="Balanced",
            random_seed=RANDOM_SEED,
            verbose=0,
            thread_count=-1,
        ),
        resampling_strategy=None,
    )

    # 5. SMOTE + LightGBM (Leak-free in-pipeline synthetic oversampling)
    models["SMOTE + LightGBM"] = build_resampling_pipeline(
        preprocessor=FraudFeaturePreprocessor(),
        classifier=LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        resampling_strategy="smote",
        sampling_ratio=0.10,  # 10% minority ratio
    )

    # 6. SMOTE + Random Forest
    models["SMOTE + Random Forest"] = build_resampling_pipeline(
        preprocessor=FraudFeaturePreprocessor(),
        classifier=RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        resampling_strategy="smote",
        sampling_ratio=0.10,
    )

    return models
