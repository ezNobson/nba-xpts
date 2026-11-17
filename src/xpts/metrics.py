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


def paired_bootstrap(y, p_a, p_b, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean per-shot log loss difference (a - b) with a 95% bootstrap CI.

    Both models are scored on the same shots (paired). Whole groups (games) are resampled,
    because shots in one game are not independent. diff < 0 means model a is better.
    """
    y = np.asarray(y)
    p_a = np.clip(np.asarray(p_a), 1e-15, 1 - 1e-15)
    p_b = np.clip(np.asarray(p_b), 1e-15, 1 - 1e-15)
    d = y * (np.log(p_b) - np.log(p_a)) + (1 - y) * (np.log(1 - p_b) - np.log(1 - p_a))
    g = pd.DataFrame({"d": d, "g": np.asarray(groups)}).groupby("g")["d"].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    idx = np.random.default_rng(seed).integers(len(s), size=(n_boot, len(s)))
    boot = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return {"diff": d.mean(), "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975),
            "share_a_better": (boot < 0).mean()}
