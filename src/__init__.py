"""Credit Card Fraud Detection package."""
import warnings

# Suppress benign future warnings from sklearn/seaborn/lightgbm/xgboost
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Compatibility patch for scikit-learn 1.6+ and XGBoost/LightGBM with imblearn pipelines
try:
    import sklearn.utils._tags as _tags_mod
    import sklearn.utils.validation as _val_mod

    _orig_get_tags = _tags_mod.get_tags

    def _safe_get_tags(estimator):
        try:
            return _orig_get_tags(estimator)
        except Exception:
            return _tags_mod.default_tags(estimator)

    _tags_mod.get_tags = _safe_get_tags
    _val_mod.get_tags = _safe_get_tags
except Exception:
    pass
