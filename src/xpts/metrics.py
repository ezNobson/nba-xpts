"""Probability metrics for shot make models."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def evaluate(y, p) -> dict:
    """Log loss, Brier score and AUC for predicted make probabilities p."""
    y, p = np.asarray(y), np.asarray(p)
    return {
        "n": len(y),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
    }


def calibration_table(y, p, n_bins: int = 20) -> pd.DataFrame:
    """Mean predicted vs observed make rate in quantile bins of p."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    return (df.groupby("bin", observed=True)
              .agg(p_mean=("p", "mean"), y_mean=("y", "mean"), n=("y", "size"))
              .reset_index(drop=True))
