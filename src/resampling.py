"""Leak-free resampling pipelines for imbalanced learning."""
from typing import Optional
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from src.config import RANDOM_SEED


def build_resampling_pipeline(
    preprocessor,
    classifier,
    resampling_strategy: Optional[str] = None,
    sampling_ratio: float = 0.1,  # resample minority to 10% of majority (stable ratio)
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """
    Construct an imblearn Pipeline ensuring that any resampling is strictly
    applied ONLY during fit() on training instances, avoiding test leakage.
    
    Strategies supported:
    - None / "none": standard pipeline without synthetic resampling
    - "smote": standard SMOTE oversampling
    - "borderline_smote": Borderline-SMOTE focusing on borderline instances
    - "random_under": Random undersampling of majority class
    - "hybrid": SMOTE followed by moderate undersampling
    """
    steps = [("preprocessor", preprocessor)]

    if resampling_strategy == "smote":
        steps.append(
            ("sampler", SMOTE(sampling_strategy=sampling_ratio, random_state=random_state))
        )
    elif resampling_strategy == "borderline_smote":
        steps.append(
            ("sampler", BorderlineSMOTE(sampling_strategy=sampling_ratio, random_state=random_state))
        )
    elif resampling_strategy == "random_under":
        # Undersample majority class to a 1:10 ratio with minority class
        steps.append(
            ("sampler", RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=random_state))
        )
    elif resampling_strategy == "hybrid":
        steps.append(
            ("oversampler", SMOTE(sampling_strategy=0.05, random_state=random_state))
        )
        steps.append(
            ("undersampler", RandomUnderSampler(sampling_strategy=0.20, random_state=random_state))
        )
    elif resampling_strategy in [None, "none"]:
        pass
    else:
        raise ValueError(f"Unsupported resampling strategy: {resampling_strategy}")

    steps.append(("classifier", classifier))
    return Pipeline(steps=steps)
