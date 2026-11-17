"""Gradient boosting (LightGBM) shot make model."""
import lightgbm as lgb
import pandas as pd

# Row-wise features from Stage 2; action_group is passed as a pandas categorical.
GBM_FEATURES = ["dist_ft", "angle_deg", "LOC_X", "LOC_Y", "is_three", "is_corner3", "is_heave",
                "is_overtime", "PERIOD", "secs_left_period", "action_group"]
# Game-context features from play-by-play (scripts/02_features/06_build_pbp_features.py), pre-shot events only.
CONTEXT_FEATURES = ["is_home", "prev_event", "secs_since_prev", "score_margin"]
CAT_COLS = ["action_group", "prev_event"]

DEFAULT_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def gbm_matrix(X: pd.DataFrame, categories: dict | None = None, features=GBM_FEATURES) -> pd.DataFrame:
    """Select features; categoricals use the training categories (unseen values -> NaN)."""
    out = X[list(features)].copy()
    for c in [c for c in CAT_COLS if c in out]:
        out[c] = pd.Categorical(out[c].astype(str), categories=None if categories is None else categories[c])
    return out


class GBMModel:
    """LGBMClassifier on gbm_matrix; categories are fixed on the training data."""

    def __init__(self, features=GBM_FEATURES, **params):
        self.features = list(features)
        self.params = {**DEFAULT_PARAMS, **params}

    def fit(self, X: pd.DataFrame, y, X_val: pd.DataFrame | None = None, y_val=None, patience: int = 50):
        """With X_val / y_val: early stopping, trees are added until the validation log loss stops improving."""
        M = gbm_matrix(X, features=self.features)
        self.categories_ = {c: M[c].cat.categories for c in CAT_COLS if c in M}
        kw = {}
        if X_val is not None:
            kw = dict(eval_X=gbm_matrix(X_val, self.categories_, self.features), eval_y=y_val,
                      callbacks=[lgb.early_stopping(patience, verbose=False)])
        self.model_ = lgb.LGBMClassifier(**self.params).fit(M, y, **kw)
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(gbm_matrix(X, self.categories_, self.features))


def tune_gbm(X_tr, y_tr, X_val, y_val, grid: list[dict], max_trees: int = 2000) -> pd.DataFrame:
    """Fit every parameter set of grid on the training part with early stopping on the validation part.

    Returns one row per set (parameters, number of trees, validation log loss), best first.
    """
    rows = []
    for params in grid:
        m = GBMModel(**params, n_estimators=max_trees).fit(X_tr, y_tr, X_val, y_val)
        rows.append({**params, "n_trees": m.model_.best_iteration_,
                     "val_log_loss": m.model_.best_score_["valid_0"]["binary_logloss"]})
    return pd.DataFrame(rows).sort_values("val_log_loss", ignore_index=True)
