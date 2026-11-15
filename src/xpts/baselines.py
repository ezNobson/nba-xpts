"""Baseline shot make models."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

ZONE_COLS = ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA"]


class ZoneBaseline(ClassifierMixin, BaseEstimator):
    """P(make) = historical FG% of the shot's court zone, smoothed towards the league FG%.

    p_zone = (makes + alpha * p_league) / (attempts + alpha)
    Zones not seen in training get p_league.
    """

    def __init__(self, cols=ZONE_COLS, alpha: float = 20.0):
        self.cols = cols
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.classes_ = np.array([0, 1])
        self.p_league_ = y.mean()
        g = y.groupby([X[c] for c in self.cols], observed=True).agg(["sum", "size"])
        self.table_ = (g["sum"] + self.alpha * self.p_league_) / (g["size"] + self.alpha)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(X[self.cols])
        p = self.table_.reindex(idx).fillna(self.p_league_).to_numpy()
        return np.column_stack([1 - p, p])


# Knots of the distance splines (ft): dense near the rim and around the 3PT line (22 / 23.75 ft).
# Fixed by court geometry, not fitted on outcomes.
DIST_KNOTS = [0, 1, 2, 3, 5, 8, 12, 16, 20, 22, 24, 26, 30, 40]
FLAG_COLS = ["is_three", "is_corner3", "is_heave", "is_overtime"]
# Knots of the optional angle spline (deg, 0 = straight on, +-90 = along the baseline), every 15 deg.
ANGLE_KNOTS = list(range(-90, 91, 15))


class ShotFeatures(TransformerMixin, BaseEstimator):
    """Sparse design matrix for the logistic regression.

    - B-splines of dist_ft (clipped to the last knot), one separate curve per action_group
      (spline basis x one-hot group, so each shot type gets its own distance effect and intercept);
    - optionally B-splines of angle_deg (clipped to +-90), one curve shared by all groups;
    - binary flags: FLAG_COLS and last_second (secs_left_period <= 1).
    Groups not seen in training get no spline terms (only the flags and the intercept).
    """

    def __init__(self, knots=DIST_KNOTS, degree: int = 3, group_col: str = "action_group", angle_knots=None):
        self.knots = knots
        self.degree = degree
        self.group_col = group_col
        self.angle_knots = angle_knots

    def fit(self, X: pd.DataFrame, y=None):
        self.spline_ = SplineTransformer(knots=np.asarray(self.knots, dtype=float).reshape(-1, 1),
                                         degree=self.degree, extrapolation="constant", sparse_output=True)
        self.spline_.fit(self._dist(X))
        self.groups_ = np.array(sorted(X[self.group_col].astype(str).unique()))
        self.n_basis_ = self.spline_.n_features_out_
        if self.angle_knots is not None:
            self.angle_spline_ = SplineTransformer(knots=np.asarray(self.angle_knots, dtype=float).reshape(-1, 1),
                                                   degree=self.degree, extrapolation="constant", sparse_output=True)
            self.angle_spline_.fit(self._angle(X))
        return self

    def _angle(self, X):
        return X[["angle_deg"]].clip(self.angle_knots[0], self.angle_knots[-1]).to_numpy()

    def _dist(self, X):
        return X[["dist_ft"]].clip(upper=self.knots[-1]).to_numpy()

    def transform(self, X: pd.DataFrame):
        s = self.spline_.transform(self._dist(X)).tocoo()
        g = pd.Categorical(X[self.group_col].astype(str), categories=self.groups_).codes.astype(np.int64)[s.row]
        keep = g >= 0
        by_group = sparse.csr_matrix(
            (s.data[keep], (s.row[keep], g[keep] * self.n_basis_ + s.col[keep])),
            shape=(len(X), len(self.groups_) * self.n_basis_))
        flags = np.column_stack([X[FLAG_COLS].to_numpy(dtype=float),
                                 (X["secs_left_period"] <= 1).to_numpy(dtype=float)])
        parts = [by_group, sparse.csr_matrix(flags)]
        if self.angle_knots is not None:
            parts.append(self.angle_spline_.transform(self._angle(X)))
        return sparse.hstack(parts, format="csr")

    def get_feature_names_out(self, input_features=None):
        spl = [f"{g}:B{j + 1}" for g in self.groups_ for j in range(self.n_basis_)]
        ang = [f"angle:B{j + 1}" for j in range(self.angle_spline_.n_features_out_)] if self.angle_knots is not None else []
        return np.array(spl + FLAG_COLS + ["last_second"] + ang)


def make_logit(C: float = 1.0, group_col: str = "action_group", angle_knots=None, knots=DIST_KNOTS) -> Pipeline:
    """Logistic regression on ShotFeatures (L2 penalty, strength 1 / C)."""
    return Pipeline([("features", ShotFeatures(knots=knots, group_col=group_col, angle_knots=angle_knots)),
                     ("logit", LogisticRegression(C=C, max_iter=2000))])
