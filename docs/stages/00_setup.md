# Stage 0: Project setup

Completed: 2026-10-17. This document records how the working environment and
repository were set up, and the conventions used in every later stage.

## 1. Environment

| Item | Value |
|---|---|
| OS | Windows with WSL (Ubuntu) |
| Project location | `~/nba-xpts` inside the WSL file system (not `/mnt/c`, which is much slower for large data files) |
| Python | 3.12.3 |
| Virtual environment | `.venv` created with `python3 -m venv .venv` |
| Editor | VS Code connected to WSL, notebooks run on the `.venv` kernel |

Main libraries (exact versions are pinned in `requirements-lock.txt`):
`nba_api` 1.11, `pandas` 3.0, `pyarrow` 26, `numpy` 2.5, `tqdm`, `matplotlib`,
`ipykernel`, `kaggle`.

Two dependency files are kept:
- `requirements.txt`: direct dependencies with minimum versions, extended in the
  stage that first needs a library.
- `requirements-lock.txt`: output of `pip freeze`, the exact environment used
  to produce the results, for reproducibility.

## 2. Repository

Private GitHub repository `ezNobson/nba-xpts`, branch `main`, SSH authentication.

```
nba-xpts/
├── data/
│   ├── raw/          # data exactly as downloaded, never edited
│   ├── interim/      # merged / technically aligned data, no feature engineering
│   └── processed/    # model-ready features
├── docs/
│   └── stages/       # one document per stage (this file, 01_data.md, ...)
├── notebooks/        # exploration and figures only, no pipeline logic
├── reports/figures/  # thesis figures (PDF)
├── scripts/
│   └── 00_data/      # numbered pipeline scripts per stage, run from repo root
├── src/xpts/         # reusable code as a Python package
├── tests/
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

### What is not committed (`.gitignore`)

- All data files (`data/*/*`, only `.gitkeep` placeholders are tracked). The full
  shot data is several hundred MB and GitHub rejects files over 100 MB.
  Reproducibility comes from the download scripts, not from committed data.
- The virtual environment, `__pycache__`, notebook checkpoints.
- Secrets: `.env`, `kaggle.json` (the Kaggle API token lives in `~/.kaggle/`).
- Windows `*:Zone.Identifier` files created when copying into WSL.

## 3. Conventions

- Code, comments, file names, figures and repository documentation in English.
- Pipeline scripts live in one folder per stage (`scripts/00_data/`,
  `scripts/01_features/`, ...) and are numbered in the order they are run.
- Raw data is never modified; every transformation is a script that writes to
  `interim/` or `processed/`.
- Thesis figures have a title with the data range, labelled axes with units and
  a legend, and are saved as PDF to `reports/figures/`.
- Each stage ends with its own document in `docs/stages/`.

## 4. Scope decisions made in this stage

- Seasons extended from the original plan (2014-15 to 2023-24) to **2014-15 to
  2025-26**, because 2024-25 and 2025-26 had already finished. The most recent
  season (2025-26) is reserved as the final test set.
- Regular season only; playoffs are excluded.
