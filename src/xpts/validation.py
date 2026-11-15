"""Time-based validation: walk-forward with an expanding window."""
import numpy as np
import pandas as pd

HOLDOUT_SEASON = "2025-26"


def walk_forward(seasons: pd.Series, include_holdout: bool = False, window: int | None = None):
    """Yield (test_season, train_mask, test_mask): train on all seasons before S, test on S.

    window=None is the expanding window; window=k trains on the k seasons before S only (sliding window).

    Season labels like "2014-15" sort chronologically as strings.
    The hold-out season is skipped unless include_holdout=True (final evaluation only).
    """
    order = sorted(seasons.unique())
    values = seasons.to_numpy()
    for i in range(1, len(order)):
        test_season = order[i]
        if test_season == HOLDOUT_SEASON and not include_holdout:
            continue
        yield test_season, np.isin(values, order[:i][-window:] if window else order[:i]), values == test_season
