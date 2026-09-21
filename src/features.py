"""Feature engineering and preprocessing module for Credit Card Fraud Detection."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler


class FraudFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature transformer.
    
    Transforms:
    1. 'Amount': Scaled using RobustScaler (resistant to large financial outliers)
       and supplemented with log1p(Amount).
    2. 'Time': Transformed into cyclical time-of-day features (sin_hour, cos_hour)
       plus scaled continuous time.
    3. 'V1' through 'V28': Retained as standardized PCA components.
    """

    def __init__(self, include_cyclical_time: bool = True, include_log_amount: bool = True):
        self.include_cyclical_time = include_cyclical_time
        self.include_log_amount = include_log_amount
        self.amount_scaler = RobustScaler()
        self.time_scaler = StandardScaler()
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        X_df = X.copy()
        if "Amount" in X_df.columns:
            self.amount_scaler.fit(X_df[["Amount"]])
        if "Time" in X_df.columns:
            self.time_scaler.fit(X_df[["Time"]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        output = pd.DataFrame(index=X_df.index)

        # 1. Amount transformations
        if "Amount" in X_df.columns:
            output["scaled_amount"] = self.amount_scaler.transform(X_df[["Amount"]]).flatten()
            if self.include_log_amount:
                output["log_amount"] = np.log1p(np.maximum(X_df["Amount"].values, 0))

        # 2. Time transformations
        if "Time" in X_df.columns:
            output["scaled_time"] = self.time_scaler.transform(X_df[["Time"]]).flatten()
            if self.include_cyclical_time:
                # 48-hour window in dataset: compute hour of day
                hour = (X_df["Time"].values // 3600) % 24
                output["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
                output["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

        # 3. PCA Features (V1 - V28)
        pca_cols = [col for col in X_df.columns if col.startswith("V") and col[1:].isdigit()]
        for col in pca_cols:
            output[col] = X_df[col].values

        self.feature_names_out_ = list(output.columns)
        return output

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
