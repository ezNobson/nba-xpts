"""Gradient boosting (LightGBM) shot make model."""
import lightgbm as lgb
import pandas as pd

# Row-wise features from Stage 2; action_group is passed as a pandas categorical.
GBM_FEATURES = ["dist_ft", "angle_deg", "LOC_X", "LOC_Y", "is_three", "is_corner3", "is_heave",
                "is_overtime", "PERIOD", "secs_left_period", "action_group"]
CAT_COLS = ["action_group"]

DEFAULT_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def gbm_matrix(X: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    """Select GBM_FEATURES; categoricals use the training categories (unseen values -> NaN)."""
    out = X[GBM_FEATURES].copy()
    for c in CAT_COLS:
        out[c] = pd.Categorical(out[c].astype(str), categories=None if categories is None else categories[c])
    return out


class GBMModel:
    """LGBMClassifier on gbm_matrix; categories are fixed on the training data."""

    def __init__(self, **params):
        self.params = {**DEFAULT_PARAMS, **params}

    def fit(self, X: pd.DataFrame, y):
        M = gbm_matrix(X)
        self.categories_ = {c: M[c].cat.categories for c in CAT_COLS}
        self.model_ = lgb.LGBMClassifier(**self.params).fit(M, y)
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(gbm_matrix(X, self.categories_))
