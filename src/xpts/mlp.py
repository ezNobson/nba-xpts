"""Multilayer perceptron (PyTorch) shot make model, the neural comparison model."""
import numpy as np
import pandas as pd
import torch
from torch import nn

from xpts.gbm import CAT_COLS, CONTEXT_FEATURES, GBM_FEATURES

MLP_FEATURES = GBM_FEATURES + CONTEXT_FEATURES
DEFAULT_PARAMS = dict(hidden=128, layers=2, dropout=0.0, lr=1e-3, weight_decay=1e-5, batch_size=2048, epochs=30)


class _Net(nn.Module):
    """Embeddings of the categorical features + numeric features -> hidden layers -> one logit."""

    def __init__(self, n_num: int, cat_sizes: list[int], hidden: int, layers: int, dropout: float):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n, min(8, n)) for n in cat_sizes])
        width = n_num + sum(e.embedding_dim for e in self.emb)
        blocks = []
        for _ in range(layers):
            blocks += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        self.body = nn.Sequential(*blocks, nn.Linear(width, 1))

    def forward(self, num, cat):
        x = torch.cat([num] + [e(cat[:, i]) for i, e in enumerate(self.emb)], dim=1)
        return self.body(x).squeeze(1)


class MLPModel:
    """MLP with the same fit / predict_proba interface as GBMModel.

    Numeric features are standardised with the training mean and std; missing values become 0 (the mean) and
    get a 0/1 indicator. Categories are fixed on the training data, index 0 is "unseen / missing".
    With X_val / y_val the validation log loss is recorded after every epoch (history_) and the weights of
    the best epoch are kept (best_epoch_).
    """

    def __init__(self, features=MLP_FEATURES, seed: int = 0, device: str | None = None, **params):
        self.features = list(features)
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.params = {**DEFAULT_PARAMS, **params}

    def _tensors(self, X: pd.DataFrame, fit: bool = False):
        cat_cols = [c for c in CAT_COLS if c in self.features]
        num_cols = [c for c in self.features if c not in cat_cols]
        num = X[num_cols].to_numpy(dtype=np.float32)
        if fit:
            self.mean_, self.std_ = np.nanmean(num, axis=0), np.nanstd(num, axis=0) + 1e-6
            self.nan_cols_ = np.isnan(num).any(axis=0)
            self.categories_ = {c: pd.Index(sorted(X[c].dropna().astype(str).unique())) for c in cat_cols}
        missing = np.isnan(num[:, self.nan_cols_]).astype(np.float32)
        num = np.nan_to_num((num - self.mean_) / self.std_, nan=0.0)
        cat = np.column_stack([self.categories_[c].get_indexer(X[c].astype(object).astype(str)) + 1 for c in cat_cols])
        return (torch.from_numpy(np.hstack([num, missing])).to(self.device),
                torch.from_numpy(cat.astype(np.int64)).to(self.device))

    def fit(self, X: pd.DataFrame, y, X_val: pd.DataFrame | None = None, y_val=None):
        p = self.params
        torch.manual_seed(self.seed)
        num, cat = self._tensors(X, fit=True)
        target = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)
        self.net_ = _Net(num.shape[1], [len(c) + 1 for c in self.categories_.values()],
                         p["hidden"], p["layers"], p["dropout"]).to(self.device)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss()
        if X_val is not None:
            num_val, cat_val = self._tensors(X_val)
            target_val = torch.from_numpy(np.asarray(y_val, dtype=np.float32)).to(self.device)
        self.history_, best_state = [], None
        for _ in range(p["epochs"]):
            self.net_.train()
            for idx in torch.randperm(len(target), device=self.device).split(p["batch_size"]):
                opt.zero_grad()
                loss_fn(self.net_(num[idx], cat[idx]), target[idx]).backward()
                opt.step()
            if X_val is not None:
                self.net_.eval()
                with torch.no_grad():
                    self.history_.append(loss_fn(self.net_(num_val, cat_val), target_val).item())
                if self.history_[-1] == min(self.history_):
                    best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
        if best_state is not None:
            self.net_.load_state_dict(best_state)
            self.best_epoch_ = int(np.argmin(self.history_)) + 1
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(*self._tensors(X))).cpu().numpy().astype(np.float64)
        return np.column_stack([1 - p, p])


def tune_mlp(X_tr, y_tr, X_val, y_val, grid: list[dict], max_epochs: int = 40) -> pd.DataFrame:
    """Fit every parameter set of grid on the training part and score it on the validation part.

    Returns one row per set (parameters, best epoch, validation log loss), best first.
    """
    rows = []
    for params in grid:
        m = MLPModel(**params, epochs=max_epochs).fit(X_tr, y_tr, X_val, y_val)
        rows.append({**params, "epochs": m.best_epoch_, "val_log_loss": min(m.history_)})
    return pd.DataFrame(rows).sort_values("val_log_loss", ignore_index=True)
